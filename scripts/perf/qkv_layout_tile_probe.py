#!/usr/bin/env python3
"""Screen CUEQ triangle-attention q/k/v layout tile choices.

The production CUEQ triangle-attention path keeps the fused q/k/v GEMM intact,
then uses a streaming Triton kernel to split the fused output and write CUEQ's
physical heads-first tensors.  Nsight Compute shows that kernel is HBM-bound at
large many-sequence batches.  This probe asks only the narrow tile question:
can a different one-dimensional streaming tile improve the layout pass enough
to justify changing production defaults?

This is benchmark-only code.  Production still lives in
``protenix.model.triangular.qkv_layout_triton`` and should only change after a
full PairformerBlock gate moves, not from an isolated copy-kernel win alone.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl


TensorFn = Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
Config = tuple[int, int]


def parse_config(spec: str) -> Config:
    parts = tuple(int(part) for part in spec.lower().replace("w", "x").split("x"))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("configs must be BLOCKxWARPS")
    return parts  # type: ignore[return-value]


def config_name(config: Config) -> str:
    block, warps = config
    return f"b{block}_w{warps}"


@triton.jit
def _qkv_to_heads_first_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements: tl.constexpr,
    j_count: tl.constexpr,
    no_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    mask = offsets < total_elements

    d = offsets % head_dim
    tmp = offsets // head_dim
    j = tmp % j_count
    tmp = tmp // j_count
    h = tmp % no_heads
    row = tmp // no_heads

    hidden = no_heads * head_dim
    input_base = (row * j_count + j) * (3 * hidden) + h * head_dim + d

    q = tl.load(qkv_ptr + input_base, mask=mask)
    k = tl.load(qkv_ptr + input_base + hidden, mask=mask)
    v = tl.load(qkv_ptr + input_base + 2 * hidden, mask=mask)

    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(k_ptr + offsets, k, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)


@triton.jit
def _qkv_to_heads_first_swapped_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements: tl.constexpr,
    i_count: tl.constexpr,
    j_count: tl.constexpr,
    no_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    mask = offsets < total_elements

    d = offsets % head_dim
    tmp = offsets // head_dim
    i = tmp % i_count
    tmp = tmp // i_count
    h = tmp % no_heads
    tmp = tmp // no_heads
    j = tmp % j_count
    batch = tmp // j_count

    hidden = no_heads * head_dim
    input_base = ((batch * i_count + i) * j_count + j) * (3 * hidden)
    input_base += h * head_dim + d

    q = tl.load(qkv_ptr + input_base, mask=mask)
    k = tl.load(qkv_ptr + input_base + hidden, mask=mask)
    v = tl.load(qkv_ptr + input_base + 2 * hidden, mask=mask)

    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(k_ptr + offsets, k, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)


def triton_layout(qkv: torch.Tensor, heads: int, config: Config, *, swapped: bool):
    block, warps = config
    head_dim = qkv.shape[-1] // (3 * heads)
    if swapped:
        out_shape = qkv.shape[:-3] + (qkv.shape[-2], heads, qkv.shape[-3], head_dim)
    else:
        out_shape = qkv.shape[:-2] + (heads, qkv.shape[-2], head_dim)
    q = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    if swapped:
        leading = 1
        for size in qkv.shape[:-3]:
            leading *= size
        total = leading * qkv.shape[-2] * heads * qkv.shape[-3] * head_dim
        grid = (triton.cdiv(total, block),)
        _qkv_to_heads_first_swapped_kernel[grid](
            qkv,
            q,
            k,
            v,
            total,
            qkv.shape[-3],
            qkv.shape[-2],
            heads,
            head_dim,
            block,
            num_warps=warps,
        )
        return q, k, v

    leading_i = 1
    for size in qkv.shape[:-2]:
        leading_i *= size
    total = leading_i * heads * qkv.shape[-2] * head_dim
    grid = (triton.cdiv(total, block),)
    _qkv_to_heads_first_kernel[grid](
        qkv,
        q,
        k,
        v,
        total,
        qkv.shape[-2],
        heads,
        head_dim,
        block,
        num_warps=warps,
    )
    return q, k, v


def torch_layout(qkv: torch.Tensor, heads: int, *, swapped: bool):
    head_dim = qkv.shape[-1] // (3 * heads)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(q.shape[:-1] + (heads, head_dim)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (heads, head_dim)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (heads, head_dim)).transpose(-2, -3)
    if swapped:
        q = q.transpose(-4, -2)
        k = k.transpose(-4, -2)
        v = v.transpose(-4, -2)
    return q.contiguous(), k.contiguous(), v.contiguous()


def time_cuda(fn: TensorFn, warmup: int, iters: int):
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


def diff_stats(reference, candidate) -> dict[str, float | int | bool]:
    max_abs = 0.0
    mean_sum = 0.0
    elem_count = 0
    nan_count = 0
    allclose = True
    for ref, cand in zip(reference, candidate):
        diff = (cand.float() - ref.float()).abs()
        finite = diff[torch.isfinite(diff)]
        if finite.numel():
            max_abs = max(max_abs, float(finite.max().item()))
            mean_sum += float(finite.sum().item())
            elem_count += finite.numel()
        nan_count += int(torch.isnan(diff).sum().item())
        allclose = allclose and bool(torch.equal(ref, cand))
    return {
        "exact": allclose,
        "max_abs": max_abs,
        "mean_abs": mean_sum / elem_count if elem_count else float("nan"),
        "nan_count": nan_count,
    }


def screen_layout(
    qkv: torch.Tensor,
    args: argparse.Namespace,
    *,
    swapped: bool,
) -> dict[str, Any]:
    reference = torch_layout(qkv, args.heads, swapped=swapped)
    current_cfg = (1024, 4)
    _current, current_ms = time_cuda(
        lambda: triton_layout(qkv, args.heads, current_cfg, swapped=swapped),
        args.warmup,
        args.iters,
    )
    rows: dict[str, Any] = {}
    best_name = ""
    best_ms = float("inf")
    best_out = None
    for config in args.configs:
        name = config_name(config)
        out, ms = time_cuda(
            lambda cfg=config: triton_layout(qkv, args.heads, cfg, swapped=swapped),
            args.warmup,
            args.iters,
        )
        rows[name] = {
            "ms": ms,
            "speedup_vs_current": current_ms / ms,
            "parity": diff_stats(reference, out),
        }
        if ms < best_ms:
            best_name = name
            best_ms = ms
            best_out = out
    assert best_out is not None
    return {
        "current": {
            "name": config_name(current_cfg),
            "ms": current_ms,
            "parity": diff_stats(reference, _current),
        },
        "candidates": rows,
        "best": rows[best_name] | {"name": best_name},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=251)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--configs", nargs="*", type=parse_config)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.configs is None:
        args.configs = [
            (128, 4),
            (256, 4),
            (512, 4),
            (1024, 4),
            (2048, 4),
            (4096, 4),
            (1024, 8),
            (2048, 8),
            (4096, 8),
        ]
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    hidden = args.heads * args.head_dim
    qkv = torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        3 * hidden,
        device="cuda",
        dtype=dtype,
    )

    with torch.inference_mode():
        normal = screen_layout(qkv, args, swapped=False)
        swapped = screen_layout(qkv, args, swapped=True)

    result = {
        "args": vars(args) | {"configs": [config_name(config) for config in args.configs]},
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "qkv_numel": qkv.numel(),
        },
        "normal": normal,
        "swapped": swapped,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
