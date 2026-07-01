#!/usr/bin/env python3
"""Benchmark one atom-transformer block on representative local-attention shapes."""

from __future__ import annotations

import argparse
import json
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
    p = torch.randn(
        args.samples,
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
    return parser.parse_args()


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
            return block.conditioned_transition_block(a=residual, s=c)

    def run_block() -> torch.Tensor:
        with amp:
            return block(
                a=q,
                s=c,
                z=p,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
            )[0]

    transition_out, transition_ms = cuda_time(
        run_transition, args.warmup, args.iters
    )
    block_out, block_ms = cuda_time(run_block, args.warmup, args.iters)

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "q_shape": list(q.shape),
        "p_shape": list(p.shape),
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
