#!/usr/bin/env python3
"""Benchmark one atom-transformer block on representative local-attention shapes."""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch

from protenix.model.modules.transformer import DiffusionTransformerBlock


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


def make_block(args: argparse.Namespace) -> DiffusionTransformerBlock:
    block = DiffusionTransformerBlock(
        n_heads=args.heads,
        c_a=args.c_atom,
        c_s=args.c_atom,
        c_z=args.c_atompair,
        cross_attention_mode=True,
    ).cuda()
    block.eval()
    return block


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    n_trunks = (args.atoms + args.n_queries - 1) // args.n_queries
    q = torch.randn(
        args.samples,
        args.atoms,
        args.c_atom,
        device=device,
        dtype=dtype,
    )
    c = torch.randn_like(q)
    p_samples = args.p_samples if args.p_samples is not None else args.samples
    p = torch.randn(
        p_samples,
        n_trunks,
        args.n_queries,
        args.n_keys,
        args.c_atompair,
        device=device,
        dtype=dtype,
    )
    return q, c, p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument(
        "--p-samples",
        type=int,
        default=None,
        help="Leading sample/batch size for atom pair bias. Use 1 to mimic the cached full inference path.",
    )
    parser.add_argument("--atoms", type=int, default=2529)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--c-atom", type=int, default=128)
    parser.add_argument("--c-atompair", type=int, default=16)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--profile-target",
        choices=["none", "attention", "transition", "block"],
        default="none",
        help="Capture one warmed target path with CUDA profiler APIs for NCU.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=1,
        help="Number of target calls inside the CUDA profiler capture range.",
    )
    parser.add_argument(
        "--transition-residual",
        type=int,
        choices=[0, 1],
        default=0,
        help="Pass the residual into ConditionedTransitionBlock, matching the promoted fused residual path.",
    )
    return parser.parse_args()


def cuda_profile_capture(
    name: str,
    fn,
    warmup: int,
    iters: int,
) -> torch.Tensor:
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
        raise RuntimeError("atom_transformer_hotspot requires CUDA")
    torch.manual_seed(args.seed)

    block = make_block(args)
    q, c, p = make_inputs(args)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    def run_attention() -> torch.Tensor:
        with amp:
            return block.attention_pair_bias(
                a=q,
                s=c,
                z=p,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
            )

    attn_out, attention_ms = cuda_time(run_attention, args.warmup, args.iters)
    residual = attn_out + q
    torch.cuda.synchronize()

    def run_transition() -> torch.Tensor:
        with amp:
            return block.conditioned_transition_block(
                a=residual,
                s=c,
                residual=residual if args.transition_residual else None,
            )

    def run_block() -> torch.Tensor:
        with amp:
            return block(
                a=q,
                s=c,
                z=p,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
            )[0]

    if args.profile_target != "none":
        profile_fns = {
            "attention": run_attention,
            "transition": run_transition,
            "block": run_block,
        }
        out = cuda_profile_capture(
            args.profile_target,
            profile_fns[args.profile_target],
            args.warmup,
            args.profile_iters,
        )
        row = {
            "args": vars(args),
            "device": torch.cuda.get_device_name(),
            "target": args.profile_target,
            "out_shape": list(out.shape),
            "out_all_finite": bool(torch.isfinite(out).all().item()),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "timestamp": time.time(),
        }
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    transition_out, transition_ms = cuda_time(
        run_transition, args.warmup, args.iters
    )
    block_out, block_ms = cuda_time(run_block, args.warmup, args.iters)

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "env": {
            name: os.environ.get(name)
            for name in (
                "PROTENIX_TRITON_LOCAL_ATTN",
                "PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE",
                "PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16",
                "PROTENIX_TRITON_FUSED_ATTENTION_RESIDUAL",
                "PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL",
                "PROTENIX_TRITON_FUSED_ELEMENTWISE",
                "PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION",
            )
        },
        "q_shape": list(q.shape),
        "p_shape": list(p.shape),
        "transition_residual": bool(args.transition_residual),
        "attention_out_shape": list(attn_out.shape),
        "transition_out_shape": list(transition_out.shape),
        "block_out_shape": list(block_out.shape),
        "full_block_ms": block_ms,
        "subrange_ms": {
            "attention_pair_bias_local_attention": attention_ms,
            "conditioned_transition": transition_ms,
        },
        "attention_out_all_finite": bool(torch.isfinite(attn_out).all().item()),
        "transition_out_all_finite": bool(torch.isfinite(transition_out).all().item()),
        "block_out_all_finite": bool(torch.isfinite(block_out).all().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
