#!/usr/bin/env python3
"""CUTLASS 3 record-L-batched screen for v2 triangle contraction.

Prior contraction screens failed for two different reasons:

* one exact cuBLAS call per record had too much launch/small-GEMM overhead;
* CUTLASS grouped GEMM used one problem per ``(feature, record)`` pair, i.e.
  4096 grouped problems for Protenix-v2 ``c_z=256, B=16``.

This benchmark tests the missing stock-CUTLASS point between those failures:
launch one SM90 GEMM per sequence record, but put the 256 independent pair
channels in GEMM's rank-4 ``L`` dimension.  If this is still not competitive
with dense PyTorch, the next step really is a custom CuTe scheduler rather than
another library wrapper.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from scripts.perf.transition_output_epilogue_cutlass_candidate import (
    _header_include_paths,
)
from scripts.perf.triangle_contraction_cutlass3_grouped_probe import (
    SHORT_LENGTHS,
    LONG_LENGTHS,
    cuda_time,
    efficiency,
    make_operands,
    pad_for_cutlass,
    parse_ints,
    tflops,
    torch_contract,
    valid_error,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT = None


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    elif "9.0a" not in arch_list and "90a" not in arch_list:
        raise RuntimeError("CUTLASS GMMA probes require TORCH_CUDA_ARCH_LIST=9.0a")

    cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
    if not cutlass_include:
        raise RuntimeError("CUTLASS_INCLUDE_DIR must point at CUTLASS/CuTe headers")

    _EXT = load(
        name="protenix_triangle_contraction_cutlass3_record_lbatched_probe",
        sources=[
            str(REPO_ROOT / "scripts/perf/triangle_contraction_cutlass3_record_lbatched_probe.cu")
        ],
        extra_include_paths=_header_include_paths(cutlass_include),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def record_lbatched_contract(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lengths: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    return extension().forward(lhs, rhs, lengths, direction == "outgoing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _time_case(fn: Callable[[], torch.Tensor], args: argparse.Namespace):
    return cuda_time(fn, warmup=args.warmup, iters=args.iters)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True

    lengths = lengths_from_args(args)
    lengths_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    lhs, rhs = make_operands(lengths, features=args.features, seed=args.seed)
    lhs_cutlass, rhs_cutlass, n_cutlass = pad_for_cutlass(lhs, rhs)
    n_max = max(lengths)
    dense_work = len(lengths) * n_max**3
    exact_work = sum(length**3 for length in lengths)

    ref, torch_ms = _time_case(lambda: torch_contract(lhs, rhs, args.direction), args)
    candidate, candidate_ms = _time_case(
        lambda: record_lbatched_contract(
            lhs_cutlass,
            rhs_cutlass,
            lengths_tensor,
            args.direction,
        ),
        args,
    )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "features": args.features,
            "batch": len(lengths),
            "n_max": n_max,
            "n_cutlass": n_cutlass,
            "num_cutlass_launches": len(lengths),
            "lengths": lengths,
            **efficiency(lengths),
        },
        "dense_torch_einsum": {
            "ms": torch_ms,
            "logical_tflops_dense": tflops(args.features, dense_work, torch_ms),
        },
        "record_lbatched_cutlass3": {
            "ms": candidate_ms,
            "logical_tflops_exact": tflops(args.features, exact_work, candidate_ms),
            "speedup_vs_dense_torch": torch_ms / candidate_ms,
            "parity": valid_error(ref, candidate, lengths),
            "invalid_region": "not materialized by this contraction-core screen",
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
