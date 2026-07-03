#!/usr/bin/env python3
"""Screen caching diffusion-transformer pair biases across denoising steps.

In low-sample campaign inference, the token diffusion transformer runs once per
denoising step, but its pair tensor `z` is step-invariant.  The current block
projects `z` to attention-head bias inside every block on every step.  This
probe asks whether storing the final `[B * samples, heads, tokens, tokens]`
bias for each block is fast enough to justify the extra memory.

This is only a hotspot screen.  A production change must still prove that the
one-time cache build plus higher live memory improves the full N_step=200 gate.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from diffusion_transformer_compile_probe import _compare, _dtype, _make_inputs
from protenix.model.modules.fused_elementwise_triton import (
    fused_sigmoid_mul,
    fused_sigmoid_mul_add,
)
from protenix.model.modules.transformer import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
    _repeat_bias_for_sample_lanes,
    fused_attention_residual_enabled,
    fused_transition_residual_enabled,
)


TensorFn = Callable[[], torch.Tensor]


def _autocast(dtype: torch.dtype):
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=dtype != torch.float32,
    )


def _pair_bias_for_block(
    block: DiffusionTransformerBlock,
    z: torch.Tensor,
    token_mask: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    """Project and expand the exact bias tensor consumed by rank-4 SDPA."""
    apb = block.attention_pair_bias
    weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[
        :, :, None, None
    ]
    bias = F.conv2d(z, weight)
    bias = _repeat_bias_for_sample_lanes(bias, samples)
    if token_mask is not None:
        key_bias = (token_mask.to(dtype=bias.dtype) - 1) * 1e4
        bias = bias + key_bias[..., None, None, :]
    return bias.contiguous()


def _precompute_biases(
    model: DiffusionTransformer,
    z: torch.Tensor,
    token_mask: torch.Tensor,
    samples: int,
) -> list[torch.Tensor]:
    return [_pair_bias_for_block(block, z, token_mask, samples) for block in model.blocks]


def _cached_attention_block(
    block: DiffusionTransformerBlock,
    a: torch.Tensor,
    s: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """DiffusionTransformerBlock.forward with pair-bias projection removed."""
    apb = block.attention_pair_bias
    q_x = apb.layernorm_a(a=a, s=s)
    attn = apb.attention(q_x=q_x, kv_x=q_x, attn_bias=bias, inplace_safe=False)
    gate = apb.linear_a_last(s)
    if fused_attention_residual_enabled() and isinstance(block.drop_path, torch.nn.Identity):
        attn_out = fused_sigmoid_mul_add(gate, attn, a)
    else:
        attn_out = fused_sigmoid_mul(gate, attn) + a

    if fused_transition_residual_enabled() and isinstance(block.drop_path, torch.nn.Identity):
        return block.conditioned_transition_block(a=attn_out, s=s, residual=attn_out)
    return block.conditioned_transition_block(a=attn_out, s=s) + attn_out


def _cached_transformer(
    model: DiffusionTransformer,
    a: torch.Tensor,
    s: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    for block, bias in zip(model.blocks, biases):
        a = _cached_attention_block(block, a, s, bias)
    return a


def _time(fn: TensorFn, warmup: int, iters: int) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = fn()
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
    return out, start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _profile_rows(fn: TensorFn, warmup: int, limit: int) -> list[dict[str, Any]]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            fn()
        torch.cuda.synchronize()

    rows = []
    for event in prof.key_averages():
        device_us = getattr(event, "device_time_total", None)
        if device_us is None:
            device_us = getattr(event, "cuda_time_total", 0.0)
        if device_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "cuda_ms": device_us / 1000.0,
                "calls": event.count,
                "shapes": str(event.input_shapes)[:220],
            }
        )
    rows.sort(key=lambda row: row["cuda_ms"], reverse=True)
    return rows[:limit]


def _case(
    name: str,
    fn: TensorFn,
    args: argparse.Namespace,
    reference: torch.Tensor | None,
    eager_ms: float,
) -> dict[str, Any]:
    out, ms, peak = _time(fn, args.warmup, args.iters)
    row: dict[str, Any] = {
        "name": name,
        "mean_ms": ms,
        "speedup_vs_eager": eager_ms / ms,
        "peak_allocated_mib": peak,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }
    if reference is not None:
        row["parity_vs_eager"] = _compare(out, reference)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_pair_bias_cache_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = _dtype(args.dtype)
    model = DiffusionTransformer(
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        n_blocks=args.blocks,
        n_heads=args.heads,
    ).cuda().eval()
    inputs = _make_inputs(args)

    def eager() -> torch.Tensor:
        with _autocast(dtype):
            return model(
                a=inputs["a"],
                s=inputs["s"],
                z=inputs["z"],
                enable_efficient_fusion=True,
                token_mask=inputs["token_mask"],
                z_sample_count=args.samples,
            )

    with _autocast(dtype), torch.inference_mode():
        bias_start = torch.cuda.Event(enable_timing=True)
        bias_end = torch.cuda.Event(enable_timing=True)
        bias_start.record()
        cached_biases = _precompute_biases(
            model, inputs["z"], inputs["token_mask"], args.samples
        )
        bias_end.record()
        bias_end.synchronize()
        bias_build_ms = bias_start.elapsed_time(bias_end)

    cached_bytes = sum(bias.numel() * bias.element_size() for bias in cached_biases)

    eager_out, eager_ms, eager_peak = _time(eager, args.warmup, args.iters)

    def cached_reuse() -> torch.Tensor:
        with _autocast(dtype):
            return _cached_transformer(model, inputs["a"], inputs["s"], cached_biases)

    def cached_rebuild() -> torch.Tensor:
        with _autocast(dtype):
            biases = _precompute_biases(model, inputs["z"], inputs["token_mask"], args.samples)
            return _cached_transformer(model, inputs["a"], inputs["s"], biases)

    cases = [
        {
            "name": "eager_recompute_pair_bias",
            "mean_ms": eager_ms,
            "speedup_vs_eager": 1.0,
            "peak_allocated_mib": eager_peak,
            "output_shape": list(eager_out.shape),
            "output_dtype": str(eager_out.dtype),
        },
        _case("cached_bias_reuse", cached_reuse, args, eager_out, eager_ms),
        _case("cached_bias_rebuild_each_call", cached_rebuild, args, eager_out, eager_ms),
    ]
    if args.profile:
        cases.append(
            {
                "name": "profiler_top_cuda",
                "eager": _profile_rows(eager, args.warmup, args.profile_top),
                "cached_bias_reuse": _profile_rows(
                    cached_reuse, args.warmup, args.profile_top
                ),
            }
        )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "cached_bias_build_ms": bias_build_ms,
        "cached_bias_mib": cached_bytes / 2**20,
        "cases": cases,
        "theoretical_e2e_note": (
            "Transformer-only screen. Full N_step gates must amortize the one-time "
            "bias build and include higher live memory before promotion."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
