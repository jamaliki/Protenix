#!/usr/bin/env python3
"""Isolate CUTLASS mainloop speed for the transition output GEMM.

The rejected fused CUTLASS candidate kept the gate load affine by treating each
token as a separate GEMM batch: ``M=N_sample, L=N_token``.  PyTorch/cuBLAS uses
one much larger flattened GEMM: ``M=N_sample * N_token``.  This probe removes
the gate/residual epilogue entirely and times only ``b @ weight.T`` so we can
tell whether the loss came from CUTLASS itself or from that token-batched shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.perf.transition_output_epilogue_cutlass_candidate import (  # noqa: E402
    _header_include_paths,
)
from scripts.perf.transition_output_epilogue_hotspot import (  # noqa: E402
    cuda_autocast,
    cuda_time,
    make_block,
    make_fixture,
    max_mean_abs,
)

_EXT = None
_WEIGHT_CACHE_KEY = None
_WEIGHT_CACHE = None


def _require_sm90a() -> None:
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    elif "9.0a" not in arch_list and "90a" not in arch_list:
        raise RuntimeError("CUTLASS GMMA probes require TORCH_CUDA_ARCH_LIST=9.0a")


def _extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    _require_sm90a()
    cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
    if not cutlass_include:
        raise RuntimeError("CUTLASS_INCLUDE_DIR must point at CUTLASS/CuTe headers")

    source = REPO_ROOT / "scripts/perf/transition_output_cutlass_mainloop_probe_sm90.cu"
    _EXT = load(
        name="protenix_transition_output_cutlass_mainloop_probe_sm90",
        sources=[str(source)],
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


def _bf16_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    global _WEIGHT_CACHE_KEY, _WEIGHT_CACHE
    key = (weight.data_ptr(), weight._version, dtype)
    if key != _WEIGHT_CACHE_KEY:
        _WEIGHT_CACHE = weight.to(dtype=dtype).contiguous()
        _WEIGHT_CACHE_KEY = key
    return _WEIGHT_CACHE


def cutlass_flat(b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight_bf16 = _bf16_weight(weight, b.dtype)
    return _extension().forward_flat(b.contiguous(), weight_bf16)


def cutlass_token_batched(b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight_bf16 = _bf16_weight(weight, b.dtype)
    return _extension().forward_token_batched(b.contiguous(), weight_bf16)


def tflops(samples: int, tokens: int, hidden: int, out_channels: int, ms: float) -> float:
    flops = 2 * samples * tokens * hidden * out_channels
    return flops / (ms * 1e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--input-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--gate-batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    block = make_block(args)
    with cuda_autocast(args.compute_dtype):
        fixture = make_fixture(block, args)
        b = fixture["b"]
        weight = fixture["weight"]
        hidden = b.shape[-1]
        out_channels = weight.shape[0]

        reference, baseline_ms = cuda_time(
            lambda: F.linear(b, weight),
            warmup=args.warmup,
            iters=args.iters,
        )
        flat_out, flat_ms = cuda_time(
            lambda: cutlass_flat(b, weight),
            warmup=args.warmup,
            iters=args.iters,
        )
        batched_out, batched_ms = cuda_time(
            lambda: cutlass_token_batched(b, weight),
            warmup=args.warmup,
            iters=args.iters,
        )

    result: dict[str, Any] = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "samples": args.samples,
            "tokens": args.tokens,
            "m_flat": args.samples * args.tokens,
            "m_token_batched": args.samples,
            "l_token_batched": args.tokens,
            "n_output": out_channels,
            "k_hidden": hidden,
        },
        "baseline_cublas": {
            "ms": baseline_ms,
            "tflops": tflops(args.samples, args.tokens, hidden, out_channels, baseline_ms),
        },
        "cutlass_flat": {
            "ms": flat_ms,
            "tflops": tflops(args.samples, args.tokens, hidden, out_channels, flat_ms),
            "speedup_vs_cublas": baseline_ms / flat_ms,
            "parity": max_mean_abs(reference, flat_out),
        },
        "cutlass_token_batched": {
            "ms": batched_ms,
            "tflops": tflops(args.samples, args.tokens, hidden, out_channels, batched_ms),
            "speedup_vs_cublas": baseline_ms / batched_ms,
            "parity": max_mean_abs(reference, batched_out),
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
