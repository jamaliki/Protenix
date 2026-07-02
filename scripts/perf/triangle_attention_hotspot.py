#!/usr/bin/env python3
"""Profile the current TriangleAttention forward path on H100.

Use this before designing custom CUDA/CuTe kernels.  The older breakdown script
times a hand-expanded path, which is useful for understanding the math, but it
does not exactly match the default fused q/k/v layout producer and fused
attention epilogue.  This script times the real module and can bracket one
warmed forward pass for Nsight Compute.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager, nullcontext

import torch

from protenix.model.modules.fused_elementwise_triton import (
    triton_fused_elementwise_available,
    triton_triangle_attention_epilogue_enabled,
)
from protenix.model.triangular.qkv_layout_triton import (
    triton_cueq_qkv_layout_available,
    triton_cueq_qkv_layout_enabled,
)
from protenix.model.triangular.triangular import TriangleAttention


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


@contextmanager
def cuda_autocast(dtype_name: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype else nullcontext()
    with ctx:
        yield


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
        starting=not args.ending,
    ).cuda()
    module.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 65537)
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(args: argparse.Namespace):
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_z,
        device="cuda",
        dtype=dtype,
    )
    mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)
    return x, mask


def cuda_time(fn, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        end.synchronize()
    return out, start.elapsed_time(end) / iters


def profile_cuda_ops(fn, warmup: int, row_limit: int):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            fn()
        torch.cuda.synchronize()

    rows = []
    for event in profiler.key_averages():
        cuda_total = float(
            getattr(event, "cuda_time_total", 0.0)
            or getattr(event, "device_time_total", 0.0)
            or 0.0
        )
        if cuda_total <= 0:
            continue
        rows.append(
            {
                "key": event.key,
                "count": int(getattr(event, "count", 0) or 0),
                "cuda_time_total_us": cuda_total,
                "self_cuda_time_total_us": float(
                    getattr(event, "self_cuda_time_total", 0.0)
                    or getattr(event, "self_device_time_total", 0.0)
                    or 0.0
                ),
                "cpu_time_total_us": float(
                    getattr(event, "cpu_time_total", 0.0) or 0.0
                ),
            }
        )
    rows.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
    return rows[:row_limit]


def cuda_profile_capture(name: str, fn, warmup: int, iters: int) -> torch.Tensor:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(name)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            out = fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
    return out


def estimate_flops(args: argparse.Namespace) -> dict[str, float]:
    batch, tokens = args.batch, args.tokens
    c_z, heads, head_dim = args.c_z, args.no_heads_pair, args.c_hidden_pair_att
    pair_sites = batch * tokens * tokens

    qkv = 2 * pair_sites * c_z * (3 * heads * head_dim)
    gate = 2 * pair_sites * c_z * (heads * head_dim)
    out = 2 * pair_sites * (heads * head_dim) * c_z
    triangle_bias = 2 * pair_sites * c_z * heads
    # Per (batch, row_i, head): QK^T and softmax-probabilities times V.
    attention_core = 4 * batch * tokens * heads * tokens * tokens * head_dim
    total = qkv + gate + out + triangle_bias + attention_core
    return {
        "qkv_tflop": qkv / 1e12,
        "gate_tflop": gate / 1e12,
        "output_projection_tflop": out / 1e12,
        "triangle_bias_tflop": triangle_bias / 1e12,
        "attention_core_tflop": attention_core / 1e12,
        "rough_total_tflop": total / 1e12,
    }


def feature_flags() -> dict[str, bool]:
    return {
        "triton_fused_elementwise_available": triton_fused_elementwise_available(),
        "triangle_attention_epilogue_enabled": triton_triangle_attention_epilogue_enabled(),
        "cueq_qkv_layout_available": triton_cueq_qkv_layout_available(),
        "cueq_qkv_layout_enabled": triton_cueq_qkv_layout_enabled(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--ending", type=str_bool, default=False)
    dtypes = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument("--profile", type=str_bool, default=False)
    parser.add_argument("--profile-row-limit", type=int, default=30)
    parser.add_argument("--ncu-target", choices=["none", "forward"], default="none")
    parser.add_argument("--ncu-iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    module = make_module(args)
    x, mask = make_inputs(args)

    def run_forward() -> torch.Tensor:
        with cuda_autocast(args.compute_dtype):
            return module(x, mask=mask, triangle_attention=args.triangle_attention)

    if args.ncu_target == "forward":
        out = cuda_profile_capture("triangle_attention_forward", run_forward, args.warmup, args.ncu_iters)
        print(
            json.dumps(
                {
                    "args": vars(args),
                    "device": torch.cuda.get_device_name(),
                    "out_shape": list(out.shape),
                    "out_all_finite": bool(torch.isfinite(out).all().item()),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "feature_flags": feature_flags(),
                    "timestamp": time.time(),
                    "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    out, forward_ms = cuda_time(run_forward, args.warmup, args.iters)
    flop_estimates = estimate_flops(args)
    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "out_shape": list(out.shape),
        "out_all_finite": bool(torch.isfinite(out).all().item()),
        "forward_ms": forward_ms,
        "estimated_flops": flop_estimates,
        "rough_effective_tflops": flop_estimates["rough_total_tflop"] / (forward_ms / 1000.0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "feature_flags": feature_flags(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    if args.profile:
        row["profile_top_cuda_ops"] = profile_cuda_ops(
            run_forward,
            args.warmup,
            args.profile_row_limit,
        )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
