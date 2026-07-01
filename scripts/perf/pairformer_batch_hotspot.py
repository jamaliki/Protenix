#!/usr/bin/env python3
"""Screen leading-batch throughput for independent pairformer sequences.

Production design inference often means many targets with ``N_sample=1..5``.
This compares running ``B`` same-length targets in a Python loop with running
one PairformerStack call with leading batch dimension ``B``.  The optional
padding check shows whether naive ragged padding is mathematically safe.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator

import torch

from protenix.model.modules.pairformer import PairformerStack


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
def cuda_autocast(dtype_name: str) -> Iterator[None]:
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def cuda_time(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iters: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
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


def make_stack(args: argparse.Namespace) -> PairformerStack:
    stack = PairformerStack(
        n_blocks=args.blocks,
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        blocks_per_ckpt=None,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    stack.eval()
    return stack


def make_inputs(
    *,
    batch: int,
    tokens: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.input_dtype)
    s = torch.randn(batch, tokens, args.c_s, device=device, dtype=dtype)
    z = torch.randn(batch, tokens, tokens, args.c_z, device=device, dtype=dtype)
    pair_mask = torch.ones(batch, tokens, tokens, device=device, dtype=dtype)
    return s, z, pair_mask


def run_batched(
    stack: PairformerStack,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Inference may overwrite ``z``; loop and batch paths clone the same volume.
    return stack(
        s.clone(),
        z.clone(),
        pair_mask=pair_mask,
        triangle_multiplicative=args.triangle_multiplicative,
        triangle_attention=args.triangle_attention,
        inplace_safe=args.inplace_safe,
        chunk_size=args.chunk_size,
    )


def run_loop(
    stack: PairformerStack,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_out = []
    z_out = []
    for index in range(s.shape[0]):
        s_i, z_i = stack(
            s[index].clone(),
            z[index].clone(),
            pair_mask=pair_mask[index],
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=args.inplace_safe,
            chunk_size=args.chunk_size,
        )
        s_out.append(s_i)
        z_out.append(z_i)
    return torch.stack(s_out, dim=0), torch.stack(z_out, dim=0)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs_error": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs_error": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def screen_batch(stack: PairformerStack, batch: int, tokens: int, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed + 1009 * batch + tokens)
    s, z, pair_mask = make_inputs(batch=batch, tokens=tokens, args=args)

    amp = cuda_autocast(args.compute_dtype)
    with amp:
        (loop_s, loop_z), loop_ms = cuda_time(
            lambda: run_loop(stack, s, z, pair_mask, args),
            warmup=args.warmup,
            iters=args.iters,
        )
        (batch_s, batch_z), batch_ms = cuda_time(
            lambda: run_batched(stack, s, z, pair_mask, args),
            warmup=args.warmup,
            iters=args.iters,
        )

    return {
        "batch": batch,
        "tokens": tokens,
        "pair_cells": batch * tokens * tokens,
        "loop_ms": loop_ms,
        "batched_ms": batch_ms,
        "sequence_throughput_loop_per_s": 1000.0 * batch / loop_ms,
        "sequence_throughput_batched_per_s": 1000.0 * batch / batch_ms,
        "batched_speedup_vs_loop": loop_ms / batch_ms,
        "per_sequence_ms_loop": loop_ms / batch,
        "per_sequence_ms_batched": batch_ms / batch,
        "s_parity": compare(batch_s, loop_s),
        "z_parity": compare(batch_z, loop_z),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def padding_check(stack: PairformerStack, args: argparse.Namespace) -> dict[str, object]:
    """Check whether padded valid-token outputs match a standalone short run."""
    short = args.padding_check_short_tokens
    full = args.padding_check_full_tokens
    if short <= 0 or full <= 0:
        return {"enabled": False}
    if short >= full:
        raise ValueError("--padding-check-short-tokens must be less than full tokens")

    torch.manual_seed(args.seed + 17)
    dtype = getattr(torch, args.input_dtype)
    device = torch.device("cuda")
    s_short = torch.randn(short, args.c_s, device=device, dtype=dtype)
    z_short = torch.randn(short, short, args.c_z, device=device, dtype=dtype)
    mask_short = torch.ones(short, short, device=device, dtype=dtype)

    s_pad = torch.zeros(full, args.c_s, device=device, dtype=dtype)
    z_pad = torch.zeros(full, full, args.c_z, device=device, dtype=dtype)
    mask_pad = torch.zeros(full, full, device=device, dtype=dtype)
    s_pad[:short] = s_short
    z_pad[:short, :short] = z_short
    mask_pad[:short, :short] = 1

    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        ref_s, ref_z = stack(
            s_short.clone(),
            z_short.clone(),
            pair_mask=mask_short,
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=args.inplace_safe,
            chunk_size=args.chunk_size,
        )
        pad_s, pad_z = stack(
            s_pad.clone(),
            z_pad.clone(),
            pair_mask=mask_pad,
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=args.inplace_safe,
            chunk_size=args.chunk_size,
        )

    return {
        "enabled": True,
        "short_tokens": short,
        "full_tokens": full,
        "s_valid_region": compare(pad_s[:short], ref_s),
        "z_valid_region": compare(pad_z[:short, :short], ref_z),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="245,600,1218")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--max-pair-cells", type=int, default=4_000_000)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--inplace-safe", type=str_bool, default=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    dtype_choices = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--padding-check-short-tokens", type=int, default=245)
    parser.add_argument("--padding-check-full-tokens", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_batch_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)
    stack = make_stack(args)

    results = []
    for tokens in parse_int_list(args.tokens):
        for batch in parse_int_list(args.batch_sizes):
            pair_cells = batch * tokens * tokens
            if pair_cells > args.max_pair_cells:
                results.append({
                    "batch": batch,
                    "tokens": tokens,
                    "pair_cells": pair_cells,
                    "skipped": True,
                    "reason": "exceeds --max-pair-cells",
                })
                continue
            torch.cuda.reset_peak_memory_stats()
            results.append(screen_batch(stack, batch, tokens, args))

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "results": results,
        "padding_check": padding_check(stack, args),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
