#!/usr/bin/env python3
"""Benchmark fused atom local-attention pair-bias projection."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch

from protenix.model.modules.local_attention_bias_triton import (
    triton_project_local_attention_bias,
)
from protenix.model.modules.transformer import AttentionPairBias
from protenix.model.utils import permute_final_dims


@contextmanager
def env_flag(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


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
        c_a=args.c_atom,
        c_s=args.c_atom,
        c_z=args.c_atompair,
        cross_attention_mode=True,
    ).cuda()
    module.eval()
    return module


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
    kv = torch.randn_like(q)
    z = torch.randn(
        args.samples,
        n_trunks,
        args.n_queries,
        args.n_keys,
        args.c_atompair,
        device=device,
        dtype=dtype,
    )
    return q, kv, z


def reference_bias(module: AttentionPairBias, z: torch.Tensor) -> torch.Tensor:
    bias = module.linear_nobias_z(module.layernorm_z(z))
    return permute_final_dims(bias, [3, 0, 1, 2])


def fused_bias(
    module: AttentionPairBias,
    z: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    out = triton_project_local_attention_bias(
        z=z,
        layernorm_weight=module.layernorm_z.weight,
        linear_weight=module.linear_nobias_z.weight,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        eps=module.layernorm_z.eps,
    )
    if out is None:
        raise RuntimeError("fused local-attention bias projection was not selected")
    return out


def run_local_attention(
    module: AttentionPairBias,
    q: torch.Tensor,
    kv: torch.Tensor,
    z: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    return module.local_multihead_attention(
        q=q,
        kv=kv,
        z=z,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        inplace_safe=False,
        chunk_size=None,
    )


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs_error": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs_error": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=640)
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
        raise RuntimeError("local_attention_bias_hotspot requires CUDA")
    torch.manual_seed(args.seed)

    module = make_module(args)
    q, kv, z = make_inputs(args)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    os.environ["PROTENIX_TRITON_LOCAL_ATTN"] = "1"
    with amp, env_flag("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS", "0"):
        ref_bias, ref_bias_ms = cuda_time(
            lambda: reference_bias(module, z),
            warmup=args.warmup,
            iters=args.iters,
        )
        ref_out, ref_full_ms = cuda_time(
            lambda: run_local_attention(module, q, kv, z, args),
            warmup=args.warmup,
            iters=args.iters,
        )

    with amp, env_flag("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS", "1"):
        cand_bias, cand_bias_ms = cuda_time(
            lambda: fused_bias(module, z, args),
            warmup=args.warmup,
            iters=args.iters,
        )
        cand_out, cand_full_ms = cuda_time(
            lambda: run_local_attention(module, q, kv, z, args),
            warmup=args.warmup,
            iters=args.iters,
        )

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "q_shape": list(q.shape),
        "z_shape": list(z.shape),
        "ref_bias_shape": list(ref_bias.shape),
        "candidate_bias_shape": list(cand_bias.shape),
        "ref_bias_ms": ref_bias_ms,
        "candidate_bias_ms": cand_bias_ms,
        "bias_speedup": ref_bias_ms / cand_bias_ms,
        "ref_full_ms": ref_full_ms,
        "candidate_full_ms": cand_full_ms,
        "full_speedup": ref_full_ms / cand_full_ms,
        "bias_parity": compare(cand_bias, ref_bias),
        "output_parity": compare(cand_out, ref_out),
        "candidate_output_all_finite": bool(torch.isfinite(cand_out).all().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
