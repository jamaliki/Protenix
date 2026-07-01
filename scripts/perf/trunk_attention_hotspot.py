#!/usr/bin/env python3
"""Benchmark diffusion trunk/token AttentionPairBias on representative H100 shapes."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from protenix.model.modules.transformer import AttentionPairBias, DiffusionTransformerBlock
from protenix.model.utils import permute_final_dims


def cuda_time(fn, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
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


def make_module(args: argparse.Namespace) -> AttentionPairBias:
    module = AttentionPairBias(
        has_s=True,
        create_offset_ln_z=False,
        n_heads=args.heads,
        c_a=args.c_token,
        c_s=args.c_s,
        c_z=args.c_z,
        cross_attention_mode=False,
    ).cuda()
    module.eval()
    return module


def make_block(args: argparse.Namespace) -> DiffusionTransformerBlock:
    block = DiffusionTransformerBlock(
        n_heads=args.heads,
        c_a=args.c_token,
        c_s=args.c_s,
        c_z=args.c_z,
        cross_attention_mode=False,
    ).cuda()
    block.eval()
    return block


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    a = torch.randn(
        args.samples,
        args.tokens,
        args.c_token,
        device=device,
        dtype=dtype,
    )
    s = torch.randn(
        args.samples,
        args.tokens,
        args.c_s,
        device=device,
        dtype=dtype,
    )
    if args.efficient_fusion:
        z = torch.randn(
            args.z_samples,
            args.c_z,
            args.tokens,
            args.tokens,
            device=device,
            dtype=dtype,
        )
    else:
        z = torch.randn(
            args.z_samples,
            args.tokens,
            args.tokens,
            args.c_z,
            device=device,
            dtype=dtype,
        )
    return a, s, z


def compute_bias(
    module: AttentionPairBias,
    z: torch.Tensor,
    *,
    efficient_fusion: bool,
) -> torch.Tensor:
    if efficient_fusion:
        weight = (module.linear_nobias_z.weight * module.layernorm_z.weight[None, :])[
            :, :, None, None
        ]
        return F.conv2d(z, weight)
    bias = module.linear_nobias_z(module.layernorm_z(z))
    return permute_final_dims(bias, [2, 0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument(
        "--z-samples",
        type=int,
        default=1,
        help="Leading sample/batch size for pair embedding z. Diffusion uses 1 and broadcasts across samples.",
    )
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--c-token", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--efficient-fusion", type=int, choices=[0, 1], default=1)
    parser.add_argument("--profile", type=int, choices=[0, 1], default=0)
    parser.add_argument("--profile-row-limit", type=int, default=25)
    parser.add_argument(
        "--ncu-target",
        choices=["none", "attention", "transition", "block"],
        default="none",
        help="Capture one warmed target path with CUDA profiler APIs for NCU.",
    )
    parser.add_argument(
        "--ncu-iters",
        type=int,
        default=1,
        help="Number of target calls inside the CUDA profiler capture range.",
    )
    return parser.parse_args()


def profile_cuda_ops(
    fn,
    warmup: int,
    row_limit: int,
) -> list[dict[str, float | int | str]]:
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
        cuda_total = float(getattr(event, "cuda_time_total", 0.0) or 0.0)
        if cuda_total <= 0:
            continue
        rows.append(
            {
                "key": event.key,
                "count": int(getattr(event, "count", 0) or 0),
                "cuda_time_total_us": cuda_total,
                "self_cuda_time_total_us": float(
                    getattr(event, "self_cuda_time_total", 0.0) or 0.0
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("trunk_attention_hotspot requires CUDA")
    torch.manual_seed(args.seed)
    args.efficient_fusion = bool(args.efficient_fusion)

    module = make_module(args)
    block = make_block(args)
    block.attention_pair_bias.load_state_dict(module.state_dict())
    a, s, z = make_inputs(args)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    def run_forward() -> torch.Tensor:
        with amp:
            a_norm = module.layernorm_a(a=a, s=s)
            return module.standard_multihead_attention(
                q=a_norm,
                kv=a_norm,
                z=z,
                enable_efficient_fusion=args.efficient_fusion,
            )

    def run_layernorm() -> torch.Tensor:
        with amp:
            return module.layernorm_a(a=a, s=s)

    def run_bias() -> torch.Tensor:
        with amp:
            return compute_bias(module, z, efficient_fusion=args.efficient_fusion)

    out, full_ms = cuda_time(run_forward, args.warmup, args.iters)
    a_norm = run_layernorm()
    bias = run_bias()
    attn_residual = out + a
    torch.cuda.synchronize()
    _, layernorm_ms = cuda_time(run_layernorm, args.warmup, args.iters)
    _, bias_ms = cuda_time(run_bias, args.warmup, args.iters)

    def run_attention() -> torch.Tensor:
        with amp:
            return module.attention(q_x=a_norm, kv_x=a_norm, attn_bias=bias)

    def run_transition() -> torch.Tensor:
        with amp:
            return block.conditioned_transition_block(a=attn_residual, s=s)

    def run_block() -> torch.Tensor:
        with amp:
            return block(
                a=a,
                s=s,
                z=z,
                enable_efficient_fusion=args.efficient_fusion,
            )[0]

    if args.ncu_target != "none":
        profile_fns = {
            "attention": run_attention,
            "transition": run_transition,
            "block": run_block,
        }
        ncu_out = cuda_profile_capture(
            args.ncu_target,
            profile_fns[args.ncu_target],
            args.warmup,
            args.ncu_iters,
        )
        row = {
            "args": vars(args),
            "device": torch.cuda.get_device_name(),
            "target": args.ncu_target,
            "out_shape": list(ncu_out.shape),
            "out_all_finite": bool(torch.isfinite(ncu_out).all().item()),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "timestamp": time.time(),
        }
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    _, attention_ms = cuda_time(
        run_attention,
        args.warmup,
        args.iters,
    )
    block_out, block_ms = cuda_time(run_block, args.warmup, args.iters)
    _, transition_ms = cuda_time(run_transition, args.warmup, args.iters)
    profiles = {}
    if args.profile:
        profiles = {
            "full_block": profile_cuda_ops(
                run_block, args.warmup, args.profile_row_limit
            ),
            "attention_pair_bias": profile_cuda_ops(
                run_forward, args.warmup, args.profile_row_limit
            ),
            "conditioned_transition": profile_cuda_ops(
                run_transition, args.warmup, args.profile_row_limit
            ),
        }

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "a_shape": list(a.shape),
        "s_shape": list(s.shape),
        "z_shape": list(z.shape),
        "out_shape": list(out.shape),
        "block_out_shape": list(block_out.shape),
        "full_block_ms": block_ms,
        "full_attention_pair_bias_ms": full_ms,
        "subrange_ms": {
            "adaptive_layernorm": layernorm_ms,
            "pair_bias_projection": bias_ms,
            "attention_qkv_sdpa_gate_out": attention_ms,
            "conditioned_transition": transition_ms,
        },
        "out_all_finite": bool(torch.isfinite(out).all().item()),
        "block_out_all_finite": bool(torch.isfinite(block_out).all().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "top_cuda_ops": profiles,
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
