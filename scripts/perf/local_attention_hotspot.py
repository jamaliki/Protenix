#!/usr/bin/env python3
"""Benchmark Protenix atom local attention on representative H100 shapes."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from typing import Iterator

import torch

from protenix.model.modules.primitives import _local_attention


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


@contextmanager
def env_flags(values: dict[str, str]) -> Iterator[None]:
    old_values = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old in old_values.items():
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


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    n_trunks = (args.atoms + args.n_queries - 1) // args.n_queries
    leading_shape = (args.outer_batch, args.samples) if args.outer_batch else (args.samples,)

    # Match Attention._prep_qkv layout: linear output is [..., N_atom, H, D],
    # then transposed to [..., H, N_atom, D].
    q = torch.randn(
        *leading_shape,
        args.atoms,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    ).transpose(-3, -2)
    k = torch.randn_like(q.transpose(-3, -2)).transpose(-3, -2)
    v = torch.randn_like(q.transpose(-3, -2)).transpose(-3, -2)

    # Match permute_final_dims(..., [3, 0, 1, 2]) from
    # [..., n_trunks, n_queries, n_keys, n_heads].
    bias = torch.randn(
        *leading_shape,
        n_trunks,
        args.n_queries,
        args.n_keys,
        args.heads,
        device=device,
        dtype=dtype,
    ).movedim(-1, -4)
    gate = None
    if args.gate:
        # Match Attention._wrap_up's gate layout after view+transpose:
        # [*, N_atom, H, D] -> [*, H, N_atom, D].
        gate = torch.randn(
            *leading_shape,
            args.atoms,
            args.heads,
            args.head_dim,
            device=device,
            dtype=dtype,
        ).transpose(-3, -2)
    return q, k, v, bias, gate


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    out, gate_applied = _local_attention(
        q=q,
        k=k,
        v=v,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        trunked_attn_bias=bias,
        use_efficient_implementation=True,
        inplace_safe=False,
        chunk_size=None,
        gate=gate,
        return_gate_status=True,
    )
    if gate is not None and not gate_applied:
        # Reference for the fused-boundary candidate: local attention followed
        # by the same gate multiply that Attention._wrap_up would run.
        out = torch.sigmoid(gate) * out
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument(
        "--outer-batch",
        type=int,
        default=0,
        help="Optional leading batch dimension before N_sample; 1 matches Protenix batched inference.",
    )
    parser.add_argument("--atoms", type=int, default=2529)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--gate",
        type=int,
        choices=[0, 1],
        default=0,
        help="Include the Attention._wrap_up sigmoid gate in the timed boundary.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("local_attention_hotspot requires CUDA")
    torch.manual_seed(args.seed)
    args.gate = bool(args.gate)
    q, k, v, bias, gate = make_inputs(args)

    with env_flag("PROTENIX_TRITON_LOCAL_ATTN", "0"):
        ref, ref_ms = cuda_time(
            lambda: run_attention(q, k, v, bias, gate, args),
            warmup=args.warmup,
            iters=args.iters,
        )
    with env_flags(
        {
            "PROTENIX_TRITON_LOCAL_ATTN": "1",
            "PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE": "0",
        }
    ):
        triton, triton_ms = cuda_time(
            lambda: run_attention(q, k, v, bias, gate, args),
            warmup=args.warmup,
            iters=args.iters,
        )
    with env_flags(
        {
            "PROTENIX_TRITON_LOCAL_ATTN": "1",
            "PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE": "1",
        }
    ):
        cand, cand_ms = cuda_time(
            lambda: run_attention(q, k, v, bias, gate, args),
            warmup=args.warmup,
            iters=args.iters,
        )

    diff = (cand.float() - ref.float()).abs()
    triton_diff = (triton.float() - ref.float()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    finite_triton_diff = triton_diff[torch.isfinite(triton_diff)]
    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "q_shape": list(q.shape),
        "q_stride": list(q.stride()),
        "bias_shape": list(bias.shape),
        "bias_stride": list(bias.stride()),
        "gate_shape": None if gate is None else list(gate.shape),
        "gate_stride": None if gate is None else list(gate.stride()),
        "triton_precision": os.environ.get(
            "PROTENIX_TRITON_LOCAL_ATTN_INPUT_PRECISION", "tf32"
        ),
        "triton_num_warps": os.environ.get("PROTENIX_TRITON_LOCAL_ATTN_NUM_WARPS", "4"),
        "ref_ms": ref_ms,
        "triton_ms": triton_ms,
        "candidate_ms": cand_ms,
        "speedup": ref_ms / cand_ms,
        "candidate_vs_triton_speedup": triton_ms / cand_ms,
        "candidate_nan_count": int(torch.isnan(cand).sum().item()),
        "ref_nan_count": int(torch.isnan(ref).sum().item()),
        "diff_nan_count": int(torch.isnan(diff).sum().item()),
        "triton_max_abs_error": (
            float(finite_triton_diff.max().item())
            if finite_triton_diff.numel()
            else float("nan")
        ),
        "triton_mean_abs_error": (
            float(finite_triton_diff.mean().item())
            if finite_triton_diff.numel()
            else float("nan")
        ),
        "max_abs_error": (
            float(finite_diff.max().item()) if finite_diff.numel() else float("nan")
        ),
        "mean_abs_error": (
            float(finite_diff.mean().item()) if finite_diff.numel() else float("nan")
        ),
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
