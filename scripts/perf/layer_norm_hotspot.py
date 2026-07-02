"""Benchmark the opt-in Triton LayerNorm inference path.

The full Protenix profiler shows PyTorch ``native_layer_norm`` as the largest
raw CUDA kernel class once the C++ fast LayerNorm extension cannot build.  This
hotspot screen compares the exact fallback expression used by
``FusedLayerNorm`` with the Triton replacement on representative channel sizes.

This is only a filter.  A win here must still be promoted by a full inference
gate, because faster LayerNorm can expose a different model bottleneck.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Callable

import torch
import torch.nn.functional as F

from protenix.model.layer_norm.layer_norm_triton import triton_layer_norm


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        rows, cols = item.lower().split("x", maxsplit=1)
        shapes.append((int(rows), int(cols)))
    return shapes


def _cuda_time(fn: Callable[[], torch.Tensor], repeat: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        out = fn()
        if out.numel() == 0:
            raise RuntimeError("empty benchmark output")
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if out.numel() == 0:
            raise RuntimeError("empty benchmark output")
        times.append(start.elapsed_time(end))
    return times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _screen(rows: int, cols: int, args: argparse.Namespace) -> dict[str, object]:
    dtype = getattr(torch, args.dtype)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32)
    normalized_shape = torch.Size([cols])

    def torch_path() -> torch.Tensor:
        # This matches FusedLayerNorm's fallback: parameters are cast to the
        # activation dtype before PyTorch native_layer_norm runs.
        return F.layer_norm(
            x,
            normalized_shape,
            weight.to(dtype=dtype),
            bias.to(dtype=dtype),
            args.eps,
        )

    def triton_path() -> torch.Tensor:
        out = triton_layer_norm(x, normalized_shape, weight, bias, args.eps)
        if out is None:
            raise RuntimeError("Triton LayerNorm path was not selected")
        return out

    with torch.no_grad():
        ref = torch_path()
        cand = triton_path()
        diff = (ref.float() - cand.float()).abs()
        torch_times = _cuda_time(torch_path, args.repeat, args.warmup)
        triton_times = _cuda_time(triton_path, args.repeat, args.warmup)

    torch_stats = _stats(torch_times)
    triton_stats = _stats(triton_times)
    return {
        "rows": rows,
        "cols": cols,
        "dtype": args.dtype,
        "torch": torch_stats,
        "triton": triton_stats,
        "speedup": torch_stats["median_ms"] / triton_stats["median_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="32768x128,8192x384,4096x768",
        help="Comma-separated ROWSxCOLS shapes. COLS is the normalized dim.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("layer_norm_hotspot requires CUDA")

    os.environ["PROTENIX_TRITON_LAYER_NORM"] = "1"
    results = [_screen(rows, cols, args) for rows, cols in _parse_shapes(args.shapes)]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
