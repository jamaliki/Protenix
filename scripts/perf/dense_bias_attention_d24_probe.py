#!/usr/bin/env python3
"""Screen a direct Triton kernel for pairformer token attention.

Pairformer token attention uses 16 heads with head dimension 24 and a dense
``[batch, head, query, key]`` pair-bias tensor.  In the current H100 runtime
this shape does not get a good cuDNN SDPA plan, so PyTorch falls back to a more
generic path.  A previous experiment reused the triangle-attention kernel by
padding D=24 to D=32; the attention math improved, but pad/copy traffic erased
the end-to-end win.

This probe tests the missing boundary directly: read Protenix's existing
``[B, H, N, 24]`` q/k/v layout and non-contiguous pair-bias layout, then do
QK + bias + softmax + PV in one Triton launch.  It is intentionally a
benchmark-only script.  If the direct kernel cannot beat the current path here,
it should not be wired into inference.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import time
from contextlib import nullcontext
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None

from protenix.model.modules.primitives import _attention
from protenix.model.modules.transformer import AttentionPairBias


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None and tl is not None:

    @triton.jit
    def _dense_bias_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        n_tokens: tl.constexpr,
        head_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        bias_stride_b: tl.constexpr,
        bias_stride_h: tl.constexpr,
        bias_stride_q: tl.constexpr,
        bias_stride_k: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_d: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        dot_input_precision: tl.constexpr,
    ):
        input_dtype = q_ptr.dtype.element_ty
        query_block = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        valid_m = offs_m < n_tokens
        valid_n = offs_n < n_tokens
        valid_d = offs_d < head_dim

        q_base = q_ptr + batch * q_stride_b + head * q_stride_h
        k_base = k_ptr + batch * k_stride_b + head * k_stride_h
        v_base = v_ptr + batch * v_stride_b + head * v_stride_h
        bias_base = bias_ptr + batch * bias_stride_b + head * bias_stride_h

        q = tl.load(
            q_base + offs_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=valid_m[:, None] & valid_d[None, :],
            other=0.0,
        )
        k = tl.load(
            k_base + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        )
        bias = tl.load(
            bias_base
            + offs_m[:, None] * bias_stride_q
            + offs_n[None, :] * bias_stride_k,
            mask=valid_m[:, None] & valid_n[None, :],
            other=-float("inf"),
        )

        scores = tl.dot(q, tl.trans(k), input_precision=dot_input_precision) + bias
        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        # Avoid NaNs in the final padded query block.  These rows are not stored.
        scores = tl.where(valid_m[:, None], scores, 0.0)
        scores = scores - tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        denom = tl.sum(probs, axis=1)
        probs = probs / denom[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        )
        # Cast probabilities down for the dot.  This mirrors flash-attention
        # style kernels: reductions stay in fp32, but the tensor-core matmul can
        # use the input dtype instead of spending the whole launch in scalar fp32.
        out = tl.dot(
            probs.to(input_dtype),
            v,
            input_precision=dot_input_precision,
        )

        out_base = out_ptr + batch * out_stride_b + head * out_stride_h
        tl.store(
            out_base
            + offs_m[:, None] * out_stride_n
            + offs_d[None, :] * out_stride_d,
            out,
            mask=valid_m[:, None] & valid_d[None, :],
        )


def triton_dense_bias_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    dot_input_precision: str,
) -> torch.Tensor | None:
    if triton is None or tl is None:
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda):
        return None
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        return None
    if bias.shape != q.shape[:3] + (q.shape[-2],):
        return None
    if q.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        return None

    batch, heads, n_tokens, head_dim = q.shape
    if head_dim > 32:
        return None
    block_d = next_power_of_2(head_dim)
    if block_n < n_tokens:
        return None

    out = torch.empty_like(q)
    grid = (triton.cdiv(n_tokens, block_m), heads, batch)
    _dense_bias_attention_kernel[grid](
        q,
        k,
        v,
        bias,
        out,
        n_tokens,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_m,
        block_n,
        block_d,
        dot_input_precision,
        num_warps=num_warps,
    )
    return out


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


def profile_cuda_ops(fn, warmup: int, row_limit: int) -> list[dict[str, Any]]:
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
        ) as profiler:
            fn()
        torch.cuda.synchronize()

    rows = []
    for event in profiler.key_averages():
        device_total = getattr(event, "device_time_total", None)
        if device_total is None:
            device_total = getattr(event, "cuda_time_total", 0.0)
        device_total = float(device_total or 0.0)
        if device_total <= 0:
            continue
        rows.append(
            {
                "key": event.key,
                "count": int(getattr(event, "count", 0) or 0),
                "cuda_time_total_us": device_total,
                "self_cuda_time_total_us": float(
                    getattr(event, "self_device_time_total", 0.0)
                    or getattr(event, "self_cuda_time_total", 0.0)
                    or 0.0
                ),
            }
        )
    rows.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
    return rows[:row_limit]


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def make_module(args: argparse.Namespace) -> AttentionPairBias:
    module = AttentionPairBias(
        has_s=False,
        create_offset_ln_z=True,
        n_heads=args.heads,
        c_a=args.c_s,
        c_z=args.c_z,
    ).cuda()
    module.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 991)
            for parameter in module.parameters():
                if parameter.numel() and not torch.count_nonzero(parameter).item():
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_token_mask(batch: int, tokens: int, min_tokens: int, device) -> torch.Tensor:
    if min_tokens <= 0 or min_tokens >= tokens:
        return torch.ones((batch, tokens), dtype=torch.bool, device=device)
    lengths = torch.linspace(tokens, min_tokens, batch, device=device).round().long()
    positions = torch.arange(tokens, device=device)
    return positions[None, :] < lengths[:, None]


def make_inputs(
    args: argparse.Namespace,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    s = torch.randn(args.batch, tokens, args.c_s, device=device, dtype=dtype)
    z = torch.randn(args.batch, tokens, tokens, args.c_z, device=device, dtype=dtype)
    token_mask = make_token_mask(args.batch, tokens, args.min_tokens, device)
    return s, z, token_mask


def autocast_context(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compare_tensors(ref: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (ref.float() - candidate.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def run_shape(args: argparse.Namespace, tokens: int) -> dict[str, Any]:
    torch.manual_seed(args.seed + tokens)
    module = make_module(args)
    s, z, token_mask = make_inputs(args, tokens)
    amp = autocast_context(args.dtype)

    with torch.inference_mode(), amp:
        s_norm = module.layernorm_a(s)
        bias = module.project_pair_bias(z=z, token_mask=token_mask)
        q, k, v = module.attention._prep_qkv(s_norm, s_norm, apply_scale=True)

    def ref_attention_core() -> torch.Tensor:
        with amp:
            return _attention(q=q, k=k, v=v, attn_bias=bias)

    def ref_full_attention() -> torch.Tensor:
        with amp:
            return module.attention(q_x=s_norm, kv_x=s_norm, attn_bias=bias)

    ref_core, ref_core_ms = cuda_time(ref_attention_core, args.warmup, args.iters)
    ref_full, ref_full_ms = cuda_time(ref_full_attention, args.warmup, args.iters)

    block_n = next_power_of_2(tokens)
    candidates = []
    for block_m in args.block_m:

        def triton_core(block_m=block_m) -> torch.Tensor:
            out = triton_dense_bias_attention(
                q,
                k,
                v,
                bias,
                block_m=block_m,
                block_n=block_n,
                num_warps=args.num_warps,
                dot_input_precision=args.dot_input_precision,
            )
            if out is None:
                raise RuntimeError("direct Triton attention path is unavailable")
            return out

        def triton_full(block_m=block_m) -> torch.Tensor:
            with amp:
                q_now, k_now, v_now = module.attention._prep_qkv(
                    s_norm, s_norm, apply_scale=True
                )
            out = triton_dense_bias_attention(
                q_now,
                k_now,
                v_now,
                bias,
                block_m=block_m,
                block_n=block_n,
                num_warps=args.num_warps,
                dot_input_precision=args.dot_input_precision,
            )
            if out is None:
                raise RuntimeError("direct Triton attention path is unavailable")
            out = out.transpose(-2, -3)
            with amp:
                return module.attention._wrap_up(out, s_norm)

        try:
            cand_core, cand_core_ms = cuda_time(
                triton_core, args.warmup, args.iters
            )
            cand_full, cand_full_ms = cuda_time(
                triton_full, args.warmup, args.iters
            )
            candidates.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "core_ms": cand_core_ms,
                    "full_ms": cand_full_ms,
                    "core_speedup": ref_core_ms / cand_core_ms,
                    "full_speedup": ref_full_ms / cand_full_ms,
                    "core_parity": compare_tensors(ref_core, cand_core),
                    "full_parity": compare_tensors(ref_full, cand_full),
                }
            )
        except Exception as exc:  # Keep one bad tile from hiding other tiles.
            candidates.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "error": repr(exc),
                }
            )

    profiles = {}
    if args.profile:
        profiles["ref_attention_core"] = profile_cuda_ops(
            ref_attention_core, args.warmup, args.profile_row_limit
        )
        if candidates and "error" not in candidates[0]:
            first_block_m = candidates[0]["block_m"]

            def profiled_triton_core() -> torch.Tensor:
                out = triton_dense_bias_attention(
                    q,
                    k,
                    v,
                    bias,
                    block_m=first_block_m,
                    block_n=block_n,
                    num_warps=args.num_warps,
                    dot_input_precision=args.dot_input_precision,
                )
                if out is None:
                    raise RuntimeError("direct Triton attention path is unavailable")
                return out

            profiles["triton_attention_core"] = profile_cuda_ops(
                profiled_triton_core, args.warmup, args.profile_row_limit
            )

    return {
        "tokens": tokens,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.c_s // args.heads,
        "dtype": args.dtype,
        "s_shape": list(s.shape),
        "z_shape": list(z.shape),
        "q_shape": list(q.shape),
        "q_stride": list(q.stride()),
        "bias_shape": list(bias.shape),
        "bias_stride": list(bias.stride()),
        "valid_tokens_min": int(token_mask.sum(dim=-1).min().item()),
        "valid_tokens_max": int(token_mask.sum(dim=-1).max().item()),
        "ref_attention_core_ms": ref_core_ms,
        "ref_full_attention_ms": ref_full_ms,
        "candidates": candidates,
        "profiles": profiles,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=parse_int_list, default=[124, 220])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Use <tokens for a padded mixed-length mask; 0 means all valid.",
    )
    parser.add_argument("--block-m", type=parse_int_list, default=[16, 32, 64])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument(
        "--dot-input-precision",
        choices=["tf32", "tf32x3", "ieee"],
        default="tf32",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--randomize-zero-weights",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Random pairformer AttentionPairBias zero-initializes gate/output "
            "weights. Randomize those tensors so full-boundary parity is tested."
        ),
    )
    parser.add_argument("--profile", type=int, choices=[0, 1], default=0)
    parser.add_argument("--profile-row-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("dense_bias_attention_d24_probe requires CUDA")
    if args.c_s % args.heads != 0:
        raise ValueError("--c-s must divide --heads")
    if args.c_s // args.heads != 24:
        raise ValueError("this probe is intentionally scoped to head_dim=24")

    torch.cuda.reset_peak_memory_stats()
    results = [run_shape(args, tokens) for tokens in args.tokens]
    print(
        json.dumps(
            {
                "args": vars(args),
                "device": torch.cuda.get_device_name(),
                "triton_available": triton is not None and tl is not None,
                "results": results,
                "timestamp": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
