#!/usr/bin/env python3
"""Test whether SDPA can use sample-broadcast pair bias efficiently.

The ragged low-sample diffusion path flattens ``(record, sample)`` before the
token diffusion transformer.  That preserves Protenix's fast rank-4 BF16
attention path, but the pair bias is actually sample-invariant:

    q/k/v: [B, S, H, N, D]
    bias:  [B,    H, N, N]

The production path currently repeats the smaller projected bias to
``[B * S, H, N, N]`` and then calls SDPA on flattened q/k/v.  This script asks a
decisive question before writing a custom kernel: can PyTorch/cuDNN consume the
natural rank-5 broadcast form quickly enough, or does it fall back to a slow
path?  A fast rank-5 result would justify a small model change; a slow result
justifies treating this as a custom attention-kernel boundary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _sync() -> None:
    torch.cuda.synchronize()


def _cuda_time(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    """Return ``(last_output, mean_ms, peak_allocated_mib)`` for one CUDA case."""
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = fn()
        for _ in range(warmup):
            out = fn()
        _sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        end.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    return out, start.elapsed_time(end) / iters, peak_mib


def _compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _record_case(
    name: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor | None,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    try:
        out, mean_ms, peak_mib = _cuda_time(fn, warmup=warmup, iters=iters)
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        return {"name": name, "ok": False, "error": repr(exc)}

    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "peak_allocated_mib": peak_mib,
        "output_shape": list(out.shape),
        "output_stride": list(out.stride()),
        "output_dtype": str(out.dtype),
    }
    if reference is not None:
        row["parity_vs_flat_prepeated"] = _compare(out, reference)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=48)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("token_attention_bias_broadcast_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    shape = (args.batch, args.samples, args.heads, args.tokens, args.head_dim)
    q5 = torch.randn(shape, device=device, dtype=dtype)
    k5 = torch.randn_like(q5)
    v5 = torch.randn_like(q5)
    bias = torch.randn(
        args.batch,
        args.heads,
        args.tokens,
        args.tokens,
        device=device,
        dtype=dtype,
    )

    q_flat = q5.flatten(0, 1)
    k_flat = k5.flatten(0, 1)
    v_flat = v5.flatten(0, 1)
    bias_flat = bias[:, None].expand(
        args.batch, args.samples, args.heads, args.tokens, args.tokens
    ).reshape(args.batch * args.samples, args.heads, args.tokens, args.tokens)
    bias_expanded5 = bias[:, None].expand(
        args.batch, args.samples, args.heads, args.tokens, args.tokens
    )

    cases: list[dict[str, Any]] = []

    reference_row = _record_case(
        "flat_prepeated_bias",
        lambda: F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=bias_flat
        ).reshape(*shape),
        reference=None,
        warmup=args.warmup,
        iters=args.iters,
    )
    cases.append(reference_row)
    reference = None
    if reference_row["ok"]:
        reference = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=bias_flat
        ).reshape(*shape)
        _sync()

    cases.append(
        _record_case(
            "flat_repeat_bias_inside_timed_region",
            lambda: F.scaled_dot_product_attention(
                q_flat,
                k_flat,
                v_flat,
                attn_mask=bias[:, None]
                .expand(
                    args.batch,
                    args.samples,
                    args.heads,
                    args.tokens,
                    args.tokens,
                )
                .reshape(
                    args.batch * args.samples,
                    args.heads,
                    args.tokens,
                    args.tokens,
                ),
            ).reshape(*shape),
            reference=reference,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    cases.append(
        _record_case(
            "rank5_expanded_stride0_bias",
            lambda: F.scaled_dot_product_attention(
                q5, k5, v5, attn_mask=bias_expanded5
            ),
            reference=reference,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    cases.append(
        _record_case(
            "rank5_broadcast_bias",
            lambda: F.scaled_dot_product_attention(q5, k5, v5, attn_mask=bias[:, None]),
            reference=reference,
            warmup=args.warmup,
            iters=args.iters,
        )
    )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "q5_shape": list(q5.shape),
        "bias_shape": list(bias.shape),
        "bias_flat_shape": list(bias_flat.shape),
        "bias_flat_is_contiguous": bool(bias_flat.is_contiguous()),
        "bias_expanded5_stride": list(bias_expanded5.stride()),
        "cases": cases,
        "timestamp": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
