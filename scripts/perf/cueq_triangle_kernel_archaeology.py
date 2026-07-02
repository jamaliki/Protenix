#!/usr/bin/env python3
"""Probe CUEQ triangle-attention boundaries on real GPU tensors.

This script is deliberately half benchmark, half readable note.  CUEQ's
high-level triangle-attention API is easy to call, but the performance-critical
question for Protenix is the *boundary* around it:

* CUEQ consumes q/k/v as physically contiguous ``[B, I, H, J, D]`` tensors on
  H100/A100-class GPUs.
* Protenix naturally gets a fused projection output in ``[B, I, J, 3*H*D]``.
* Passing q/k/v as views is mathematically correct, but CUEQ's Python wrapper
  materializes contiguous copies before its cuDNN FMHA launch.

The goal here is to verify that behavior with CUDA timings and profiler event
names, then compare it to our current producer-side layout kernel and the older
Triton triangle-attention implementation.  If we later write a CuTe kernel,
this file is the executable spec it should beat.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from protenix.model.tri_attention import TriAttention
from protenix.model.triangular.qkv_layout_triton import (
    triton_cueq_qkv_layout_available,
    triton_cueq_qkv_layout_enabled,
    triton_qkv_to_cueq_heads_first,
)


TensorFactory = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    packed_qkv: torch.Tensor
    bias: torch.Tensor
    mask_bool: torch.Tensor
    mask_additive: torch.Tensor
    scale: float


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def synchronize_time(fn: TensorFactory, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "dtype": str(x.dtype),
        "contiguous": x.is_contiguous(),
    }


def make_inputs(args: argparse.Namespace) -> Inputs:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    shape = (args.batch, args.rows, args.heads, args.tokens, args.head_dim)

    q = torch.randn(shape, device=device, dtype=dtype).contiguous()
    k = torch.randn(shape, device=device, dtype=dtype).contiguous()
    v = torch.randn(shape, device=device, dtype=dtype).contiguous()

    # The fused projection layout produced by the GEMM is [B, I, J, 3*H*D].
    # Building it from the contiguous CUEQ tensors gives us view-based q/k/v
    # with exactly matching values, so parity checks isolate layout effects.
    q_flat = q.transpose(2, 3).reshape(args.batch, args.rows, args.tokens, -1)
    k_flat = k.transpose(2, 3).reshape(args.batch, args.rows, args.tokens, -1)
    v_flat = v.transpose(2, 3).reshape(args.batch, args.rows, args.tokens, -1)
    packed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # CUEQ's triangle bias is [B, 1, H, query, key].  It is shared over the
    # fixed triangle row I.  This distinction matters: using [B, I, key, H]
    # as the bias is a different operation and can look falsely fast.
    bias = torch.randn(
        args.batch,
        1,
        args.heads,
        args.tokens,
        args.tokens,
        device=device,
        dtype=torch.float32,
    )
    mask_bool = torch.ones(
        args.batch,
        args.rows,
        1,
        1,
        args.tokens,
        device=device,
        dtype=torch.bool,
    )
    if args.mask_every:
        mask_bool[..., args.mask_every - 1 :: args.mask_every] = False
    mask_additive = torch.where(
        mask_bool,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), -1e9, device=device, dtype=dtype),
    )
    return Inputs(
        q=q,
        k=k,
        v=v,
        packed_qkv=packed_qkv,
        bias=bias,
        mask_bool=mask_bool,
        mask_additive=mask_additive,
        scale=1.0 / math.sqrt(args.head_dim),
    )


def packed_views(
    packed_qkv: torch.Tensor,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = packed_qkv.shape[-1] // 3
    head_dim = hidden // heads
    chunks = packed_qkv.split(hidden, dim=-1)
    return tuple(
        chunk.view(chunk.shape[:-1] + (heads, head_dim)).transpose(-2, -3)
        for chunk in chunks
    )  # type: ignore[return-value]


def cueq_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_bool: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    from cuequivariance_torch.primitives.triangle import triangle_attention

    out = triangle_attention(q, k, v, bias, mask_bool, scale=scale, return_aux=False)
    if isinstance(out, tuple):
        return out[0]
    return out


def triton_triattention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_additive: torch.Tensor,
) -> torch.Tensor:
    # The legacy Triton op uses [B, I, J, H, D].  Its forward kernel multiplies
    # qk by log2(e), but does not take an external scale, so we scale q here to
    # match CUEQ's q @ k / sqrt(D).
    q_legacy = (q * (1.0 / math.sqrt(q.shape[-1]))).transpose(2, 3).contiguous()
    k_legacy = k.transpose(2, 3).contiguous()
    v_legacy = v.transpose(2, 3).contiguous()
    out = TriAttention()(q_legacy, k_legacy, v_legacy, mask_additive, bias)
    return out.transpose(2, 3).contiguous()


def profile_one(name: str, fn: TensorFactory, warmup: int) -> list[dict[str, Any]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with nvtx_range(name):
            fn()
    torch.cuda.synchronize()

    rows = []
    for event in prof.key_averages():
        cuda_us = event.cuda_time_total
        if cuda_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "cuda_ms": cuda_us / 1000.0,
                "calls": event.count,
                "shapes": str(event.input_shapes)[:240],
            }
        )
    rows.sort(key=lambda row: row["cuda_ms"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rows", type=int, default=245)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mask-every", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)

    inputs = make_inputs(args)
    q_view, k_view, v_view = packed_views(inputs.packed_qkv, args.heads)
    helper = triton_qkv_to_cueq_heads_first(inputs.packed_qkv, args.heads)
    if helper is None:
        raise RuntimeError("triton_qkv_to_cueq_heads_first was unavailable")
    q_helper, k_helper, v_helper = helper

    variants: dict[str, TensorFactory] = {
        "cueq_contiguous_qkv": lambda: cueq_attention(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.bias,
            inputs.mask_bool,
            inputs.scale,
        ),
        "cueq_strided_qkv_views": lambda: cueq_attention(
            q_view,
            k_view,
            v_view,
            inputs.bias,
            inputs.mask_bool,
            inputs.scale,
        ),
        "qkv_triton_layout_plus_cueq": lambda: cueq_attention(
            *triton_qkv_to_cueq_heads_first(inputs.packed_qkv, args.heads),
            inputs.bias,
            inputs.mask_bool,
            inputs.scale,
        ),
        "qkv_torch_views_plus_cueq": lambda: cueq_attention(
            *packed_views(inputs.packed_qkv, args.heads),
            inputs.bias,
            inputs.mask_bool,
            inputs.scale,
        ),
        "legacy_triton_triattention": lambda: triton_triattention(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.bias,
            inputs.mask_additive,
        ),
    }

    # Validate values once.  We keep tolerances realistic for BF16+FMHA because
    # different softmax implementations can accumulate in different orders.
    with torch.no_grad():
        ref = variants["cueq_contiguous_qkv"]()
        parity = {}
        for name, fn in variants.items():
            out = fn()
            delta = (out.float() - ref.float()).abs()
            parity[name] = {
                "max_abs": float(delta.max().item()),
                "mean_abs": float(delta.mean().item()),
            }
        helper_delta = (q_helper - inputs.q).float().abs().max().item()
        view_delta = (q_view - inputs.q).float().abs().max().item()

    for _ in range(args.warmup):
        for fn in variants.values():
            fn()
    torch.cuda.synchronize()

    timings = {}
    with torch.no_grad():
        for name, fn in variants.items():
            timings[name] = synchronize_time(fn, args.iters)

    result: dict[str, Any] = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "rows": args.rows,
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
        },
        "triton_qkv_layout_enabled": triton_cueq_qkv_layout_enabled(),
        "triton_qkv_layout_available": triton_cueq_qkv_layout_available(),
        "layouts": {
            "q_contiguous": tensor_summary(inputs.q),
            "q_view_from_packed": tensor_summary(q_view),
            "q_helper": tensor_summary(q_helper),
        },
        "q_value_checks": {
            "view_vs_contiguous_max_abs": float(view_delta),
            "helper_vs_contiguous_max_abs": float(helper_delta),
        },
        "parity_vs_cueq_contiguous": parity,
        "timings_ms": timings,
        "speedups_vs_qkv_torch_views_plus_cueq": {
            name: timings["qkv_torch_views_plus_cueq"] / ms
            for name, ms in timings.items()
        },
    }

    if args.profile:
        result["profiler_top_cuda"] = {
            name: profile_one(name, fn, args.warmup)[: args.profile_top]
            for name, fn in variants.items()
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    with torch.no_grad():
        main()
