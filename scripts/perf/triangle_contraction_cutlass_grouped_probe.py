#!/usr/bin/env python3
"""Grouped-CUTLASS screen for the v2 triangle contraction core.

The cuBLAS wrapper probe showed that one launch per sequence record loses.  This
probe keeps the same padded CUEQ-style operands but calls CUTLASS grouped GEMM so
all exact-length ``(feature, record)`` GEMMs run in one kernel launch family.
If this loses, the next credible path is a custom Hopper/CuTe segmented
scheduler or a fuller fused triangle-multiplication update.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPE = torch.bfloat16
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]
_EXT = None


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
    if not cutlass_include:
        raise RuntimeError("CUTLASS_INCLUDE_DIR must point at CUTLASS/CuTe headers")

    _EXT = load(
        name="protenix_triangle_contraction_cutlass_grouped_probe",
        sources=[str(REPO_ROOT / "scripts/perf/triangle_contraction_cutlass_grouped_probe.cu")],
        extra_include_paths=_header_include_paths(cutlass_include),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def cuda_time(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def make_operands(lengths: list[int], *, features: int, seed: int):
    n_max = max(lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lhs = torch.zeros(features, len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    rhs = torch.zeros_like(lhs)
    for batch_index, length in enumerate(lengths):
        shape = (features, length, length)
        lhs[:, batch_index, :length, :length] = torch.randn(
            shape, device="cuda", dtype=DTYPE, generator=generator
        )
        rhs[:, batch_index, :length, :length] = torch.randn(
            shape, device="cuda", dtype=DTYPE, generator=generator
        )
    return lhs.contiguous(), rhs.contiguous()


def torch_contract(lhs: torch.Tensor, rhs: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "outgoing":
        return torch.einsum("dbik,dbjk->dbij", lhs, rhs)
    return torch.einsum("dbki,dbkj->dbij", lhs, rhs)


def cutlass_grouped_contract(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lengths: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    return extension().forward(lhs, rhs, lengths, direction == "outgoing")


def valid_error(ref: torch.Tensor, candidate: torch.Tensor, lengths: list[int]) -> dict[str, float]:
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    for batch_index, length in enumerate(lengths):
        diff = (
            ref[:, batch_index, :length, :length].float()
            - candidate[:, batch_index, :length, :length].float()
        ).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        total_abs += float(diff.sum().item())
        count += diff.numel()
    return {"max_abs": max_abs, "mean_abs": total_abs / max(count, 1)}


def invalid_abs_sum(candidate: torch.Tensor, lengths: list[int]) -> float:
    invalid = 0.0
    for batch_index, length in enumerate(lengths):
        invalid += float(candidate[:, batch_index, length:, :].float().abs().sum().item())
        invalid += float(candidate[:, batch_index, :length, length:].float().abs().sum().item())
    return invalid


def efficiency(lengths: list[int]) -> dict[str, float]:
    n_max = max(lengths)
    valid_cubic = sum(length**3 for length in lengths)
    padded_cubic = len(lengths) * n_max**3
    return {
        "valid_cubic_efficiency": valid_cubic / padded_cubic,
        "ideal_contraction_work_speedup": padded_cubic / valid_cubic,
    }


def tflops(features: int, work_units: int, ms: float) -> float:
    return (2 * features * work_units) / (ms * 1e9)


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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True

    lengths = lengths_from_args(args)
    lengths_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    lhs, rhs = make_operands(lengths, features=args.features, seed=args.seed)
    n_max = max(lengths)
    dense_work = len(lengths) * n_max**3
    exact_work = sum(length**3 for length in lengths)
    timing = {"warmup": args.warmup, "iters": args.iters}

    ref, torch_ms = cuda_time(lambda: torch_contract(lhs, rhs, args.direction), **timing)
    grouped, grouped_ms = cuda_time(
        lambda: cutlass_grouped_contract(lhs, rhs, lengths_tensor, args.direction),
        **timing,
    )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "features": args.features,
            "batch": len(lengths),
            "n_max": n_max,
            "lengths": lengths,
            **efficiency(lengths),
        },
        "dense_torch_einsum": {
            "ms": torch_ms,
            "logical_tflops_dense": tflops(args.features, dense_work, torch_ms),
        },
        "segmented_cutlass_grouped": {
            "ms": grouped_ms,
            "logical_tflops_exact": tflops(args.features, exact_work, grouped_ms),
            "speedup_vs_dense_torch": torch_ms / grouped_ms,
            "parity": valid_error(ref, grouped, lengths),
            "invalid_abs_sum": invalid_abs_sum(grouped, lengths),
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
