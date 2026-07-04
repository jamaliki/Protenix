#!/usr/bin/env python3
"""cuBLAS screen for the v2 triangle-multiplication contraction core.

This probe tests the first production-shaped ragged boundary:

    out[d,b,i,j] = sum_k lhs[d,b,i,k] * rhs[d,b,j,k]

The earlier CUTLASS screen copied exact-length groups before timing.  This probe
keeps the padded CUEQ layout, passes a ``lengths`` vector to a native extension,
and asks cuBLAS to run exact-length GEMMs directly from the padded storage.  The
output remains padded, so this is the smallest realistic test of a segmented
contraction boundary for the Protenix-v2 pairformer.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


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


def _cuda_include_paths() -> list[str]:
    paths: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        root = Path(cuda_home)
        paths.extend([root / "include", root / "targets/x86_64-linux/include"])
    for root in [Path(sys.prefix), Path(torch.__file__).resolve().parents[1]]:
        paths.extend(root.glob("lib/python*/site-packages/nvidia/cu*/include"))

    include_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in seen:
                include_paths.append(resolved)
                seen.add(resolved)
    return include_paths


def _cuda_library_flags() -> list[str]:
    paths: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        root = Path(cuda_home)
        paths.extend([root / "lib64", root / "targets/x86_64-linux/lib"])
    for root in [Path(sys.prefix), Path(torch.__file__).resolve().parents[1]]:
        paths.extend(root.glob("lib/python*/site-packages/nvidia/cu*/lib"))
        paths.extend(root.glob("lib/python*/site-packages/nvidia/cublas/lib"))

    flags: list[str] = []
    seen: set[str] = set()
    cublas_candidates: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in seen:
                flags.append(f"-L{resolved}")
                flags.append(f"-Wl,-rpath,{resolved}")
                seen.add(resolved)
            cublas_candidates.extend(path.glob("libcublas.so*"))
    if not cublas_candidates:
        flags.append("-lcublas")
    else:
        # CUDA wheel installs often provide only versioned libraries.  Prefer the
        # CUDA-13 cuBLAS because grouped GEMM is a new API and may be absent from
        # the older CUDA-12 compatibility library in the same Python env.
        cublas_candidates.sort(
            key=lambda p: (
                "cu13" not in str(p),
                p.name != "libcublas.so",
                p.name,
            )
        )
        flags.append(str(cublas_candidates[0].resolve()))
    return flags


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    source = REPO_ROOT / "scripts/perf/triangle_contraction_cublas_grouped_probe.cu"
    _EXT = load(
        name="protenix_triangle_contraction_cublas_grouped_probe",
        sources=[str(source)],
        extra_include_paths=_cuda_include_paths(),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=_cuda_library_flags(),
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


def cublas_record_strided_contract(
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
        lambda: cublas_record_strided_contract(lhs, rhs, lengths_tensor, args.direction),
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
        "segmented_cublas_record_strided": {
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
