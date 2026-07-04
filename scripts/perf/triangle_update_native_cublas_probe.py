#!/usr/bin/env python3
"""Screen a native exact-group cuBLAS contraction/assembly slice.

This is the first executable step after the native triangle-update design note.
It consumes the same exact-group tensors produced by
``triangle_update_producer_owned_direct_probe.py``, but replaces

    Python torch.bmm(groups) + Triton assemble_update(groups)

with one C++/CUDA extension call that runs cuBLAS on each exact-length group and
assembles row-major compact update rows inside native code.

This is not the final CuTe kernel.  It is a cheap acceptance test for whether
moving the contraction/assembly boundary out of PyTorch helps before attempting
a custom SM90 mainloop.  The report deliberately separates the *actual* path
that still pays the current Python/Triton producer from the post-producer
native boundary.  That avoids confusing a useful kernel-boundary signal with a
real full-update speedup.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from scripts.perf.triangle_contraction_cublas_probe import (
    _cuda_include_paths,
    _cuda_library_flags,
)
from scripts.perf.triangle_update_compact_segmented_probe import (
    C_Z,
    ContractConfig,
    baseline_update,
    cached_weights,
    compare,
    compact_finish_from_update,
    cuda_time,
    make_inputs,
    make_modules,
    parse_ints,
    valid_from_dense,
)
from scripts.perf.triangle_update_producer_owned_direct_probe import (
    direct_owned_groups,
    direct_owned_update,
    grouped_dense_offsets,
    lengths_from_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT = None


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    _EXT = load(
        name="protenix_triangle_update_native_cublas_probe",
        sources=[str(REPO_ROOT / "scripts/perf/triangle_update_native_cublas_probe.cu")],
        extra_include_paths=_cuda_include_paths(),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=_cuda_library_flags(),
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def native_contract_assemble(groups, rows: int, direction: str) -> torch.Tensor:
    return extension().contract_assemble(
        [group.lhs for group in groups],
        [group.rhs for group in groups],
        [group.start for group in groups],
        rows,
        C_Z,
        direction == "outgoing",
    )


def native_cublas_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    dense_offsets: torch.Tensor,
    lengths: list[int],
    direction: str,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    x_norm, groups = direct_owned_groups(
        module,
        z,
        dense_offsets,
        lengths,
        weights,
        row_stats="tiled",
        row_stat_block_rows=8,
    )
    update = native_contract_assemble(groups, dense_offsets.numel(), direction)
    return compact_finish_from_update(module, z, x_norm, update, dense_offsets, weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="both")
    parser.add_argument(
        "--config",
        type=ContractConfig.parse,
        default=ContractConfig.parse("32x32x64x1x4x3"),
        help="Accepted for consistency with the other triangle-update probes.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    lengths = lengths_from_args(args)
    modules = make_modules(args)
    z, mask = make_inputs(lengths, args.seed)
    dense_offsets = grouped_dense_offsets(lengths)
    directions = ["outgoing", "incoming"] if args.direction == "both" else [args.direction]

    results: dict[str, Any] = {}
    for direction in directions:
        module = modules[direction]
        weights = cached_weights(module)
        base_out, base_ms = cuda_time(
            lambda module=module: baseline_update(module, z, mask),
            args.warmup,
            args.iters,
        )
        current_out, current_ms = cuda_time(
            lambda: direct_owned_update(
                module,
                z,
                dense_offsets,
                lengths,
                direction,
                weights,
                row_stats="tiled",
                row_stat_block_rows=8,
            ),
            args.warmup,
            args.iters,
        )
        producer_out, producer_ms = cuda_time(
            lambda: direct_owned_groups(
                module,
                z,
                dense_offsets,
                lengths,
                weights,
                row_stats="tiled",
                row_stat_block_rows=8,
            ),
            args.warmup,
            args.iters,
        )
        x_norm, groups = producer_out
        _update_out, update_ms = cuda_time(
            lambda: native_contract_assemble(groups, dense_offsets.numel(), direction),
            args.warmup,
            args.iters,
        )
        native_boundary_out, native_boundary_ms = cuda_time(
            lambda: compact_finish_from_update(
                module,
                z,
                x_norm,
                native_contract_assemble(groups, dense_offsets.numel(), direction),
                dense_offsets,
                weights,
            ),
            args.warmup,
            args.iters,
        )
        native_full_out, native_full_ms = cuda_time(
            lambda: native_cublas_update(
                module,
                z,
                dense_offsets,
                lengths,
                direction,
                weights,
            ),
            args.warmup,
            args.iters,
        )

        ref_valid = valid_from_dense(base_out, dense_offsets)
        results[direction] = {
            "baseline_cueq": {"ms": base_ms},
            "current_direct_full": {
                "ms": current_ms,
                "speedup_vs_baseline": base_ms / current_ms,
                "parity_valid": compare(ref_valid, valid_from_dense(current_out, dense_offsets)),
            },
            "direct_producer_only": {"ms": producer_ms},
            "native_contract_assemble_only": {"ms": update_ms},
            "native_contract_finish_after_reused_producer": {
                "ms": native_boundary_ms,
                "note": "Excludes direct producer cost; use this only to judge the native post-producer boundary.",
                "parity_valid": compare(
                    ref_valid, valid_from_dense(native_boundary_out, dense_offsets)
                ),
            },
            "native_full_with_current_producer": {
                "ms": native_full_ms,
                "speedup_vs_baseline": base_ms / native_full_ms,
                "speedup_vs_current_direct": current_ms / native_full_ms,
                "parity_valid": compare(
                    ref_valid, valid_from_dense(native_full_out, dense_offsets)
                ),
            },
        }

    n_max = max(lengths)
    payload = {
        "args": {**vars(args), "config": args.config.label(), "lengths": lengths},
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": len(lengths),
            "n_max": n_max,
            "c_z": C_Z,
            "valid_pair_efficiency": sum(length * length for length in lengths)
            / (len(lengths) * n_max * n_max),
            "valid_cubic_efficiency": sum(length**3 for length in lengths)
            / (len(lengths) * n_max**3),
        },
        "results": results,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
