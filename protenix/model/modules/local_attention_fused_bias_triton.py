# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import os
from math import prod
from typing import Optional

import torch
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None


_SUPPORTED_INPUT_DTYPES = {torch.float32, torch.bfloat16}

def triton_fused_local_attention_bias_attention_enabled() -> bool:
    return os.getenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_ATTN", "0").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_fused_local_attention_bias_attention_available() -> bool:
    return triton is not None and tl is not None


def triton_fused_local_attention_bias_attention_input_precision() -> str:
    precision = os.getenv(
        "PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_ATTN_INPUT_PRECISION", "tf32"
    ).lower()
    if precision in {"tf32x3", "tf32", "ieee"}:
        return precision
    return "tf32"


def triton_fused_local_attention_bias_attention_num_warps() -> int:
    try:
        num_warps = int(os.getenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_ATTN_NUM_WARPS", "4"))
    except ValueError:
        return 4
    return num_warps if num_warps in {1, 2, 4, 8} else 4


if triton_fused_local_attention_bias_attention_available():

    @triton.jit
    def _local_attention_with_bias_projection_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        z_ptr,
        ln_weight_ptr,
        linear_weight_ptr,
        out_ptr,
        n_atoms: tl.constexpr,
        n_trunks: tl.constexpr,
        pad_left: tl.constexpr,
        q_stride_s: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_s: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_s: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        z_stride_s: tl.constexpr,
        z_stride_t: tl.constexpr,
        z_stride_q: tl.constexpr,
        z_stride_k: tl.constexpr,
        z_stride_c: tl.constexpr,
        lw_stride_h: tl.constexpr,
        lw_stride_c: tl.constexpr,
        o_stride_s: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_n: tl.constexpr,
        o_stride_d: tl.constexpr,
        n_heads: tl.constexpr,
        c_z: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        dot_input_precision: tl.constexpr,
    ):
        pid = tl.program_id(0)
        trunk = pid % n_trunks
        head = (pid // n_trunks) % n_heads
        sample = pid // (n_trunks * n_heads)

        offs_m = tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        q_idx = trunk * block_m + offs_m
        k_idx = trunk * block_m - pad_left + offs_n
        q_mask = q_idx < n_atoms
        k_mask = (k_idx >= 0) & (k_idx < n_atoms)
        pair_mask = q_mask[:, None] & k_mask[None, :]

        q_base = q_ptr + sample * q_stride_s + head * q_stride_h
        k_base = k_ptr + sample * k_stride_s + head * k_stride_h
        v_base = v_ptr + sample * v_stride_s + head * v_stride_h
        z_base = z_ptr + sample * z_stride_s + trunk * z_stride_t

        q = tl.load(
            q_base + q_idx[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=q_mask[:, None],
            other=0.0,
        )
        k = tl.load(
            k_base + k_idx[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
            mask=k_mask[:, None],
            other=0.0,
        )

        sum_z = tl.zeros((block_m, block_n), tl.float32)
        sum_z2 = tl.zeros((block_m, block_n), tl.float32)
        weighted = tl.zeros((block_m, block_n), tl.float32)
        weight_sum = tl.full((), 0.0, tl.float32)
        for c in range(0, c_z):
            z_c = tl.load(
                z_base
                + offs_m[:, None] * z_stride_q
                + offs_n[None, :] * z_stride_k
                + c * z_stride_c,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            ln_w = tl.load(ln_weight_ptr + c).to(tl.float32)
            linear_w = tl.load(
                linear_weight_ptr + head * lw_stride_h + c * lw_stride_c
            ).to(tl.float32)
            w = ln_w * linear_w
            sum_z += z_c
            sum_z2 += z_c * z_c
            weighted += z_c * w
            weight_sum += w

        inv_c = 1.0 / c_z
        mean = sum_z * inv_c
        var = sum_z2 * inv_c - mean * mean
        bias = (weighted - mean * weight_sum) * tl.rsqrt(var + eps)

        score = tl.dot(q, tl.trans(k), input_precision=dot_input_precision) + bias
        score = tl.where(pair_mask, score, -float("inf"))
        score = tl.where(q_mask[:, None], score, 0.0)
        score_max = tl.max(score, 1)
        score = score - score_max[:, None]
        prob = tl.exp(score)
        prob = prob / tl.sum(prob, 1)[:, None]

        v = tl.load(
            v_base + k_idx[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=k_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        out = tl.dot(prob, v, input_precision=dot_input_precision)
        out_base = out_ptr + sample * o_stride_s + head * o_stride_h
        tl.store(
            out_base + q_idx[:, None] * o_stride_n + offs_d[None, :] * o_stride_d,
            out,
            mask=q_mask[:, None],
        )


def triton_local_attention_with_bias_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    layernorm_weight: Optional[torch.Tensor],
    linear_weight: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    eps: float,
) -> Optional[torch.Tensor]:
    if not triton_fused_local_attention_bias_attention_enabled():
        return None
    if not triton_fused_local_attention_bias_attention_available():
        return None
    if torch.is_grad_enabled():
        return None
    if not q.is_cuda:
        return None
    if layernorm_weight is None:
        return None
    if n_queries != 32 or n_keys != 128:
        return None
    if q.shape != k.shape or q.shape != v.shape:
        return None
    if q.ndim < 3 or z.ndim < 5:
        return None
    if q.shape[-1] != 32:
        return None
    if (
        q.dtype not in _SUPPORTED_INPUT_DTYPES
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        return None
    if z.dtype not in _SUPPORTED_INPUT_DTYPES:
        return None

    batch_shape = q.shape[:-3]
    n_heads, n_atoms, head_dim = q.shape[-3:]
    n_trunks = (n_atoms + n_queries - 1) // n_queries
    c_z = z.shape[-1]
    if n_heads != 4 or c_z not in {16, 32}:
        return None
    if z.shape[:-4] != batch_shape:
        return None
    if z.shape[-4:-1] != (n_trunks, n_queries, n_keys):
        return None
    if layernorm_weight.shape != (c_z,) or linear_weight.shape != (n_heads, c_z):
        return None
    if not layernorm_weight.is_cuda or not linear_weight.is_cuda:
        return None
    if not layernorm_weight.is_contiguous() or not linear_weight.is_contiguous():
        return None

    n_sample = prod(batch_shape) if batch_shape else 1
    flat_q_shape = (n_sample, n_heads, n_atoms, head_dim)
    flat_z_shape = (n_sample, n_trunks, n_queries, n_keys, c_z)
    q_flat = q.reshape(flat_q_shape)
    k_flat = k.reshape(flat_q_shape)
    v_flat = v.reshape(flat_q_shape)
    z_flat = z.reshape(flat_z_shape)
    if (
        q_flat.data_ptr() != q.data_ptr()
        or k_flat.data_ptr() != k.data_ptr()
        or v_flat.data_ptr() != v.data_ptr()
        or z_flat.data_ptr() != z.data_ptr()
    ):
        return None

    out = torch.empty_strided(
        flat_q_shape,
        q_flat.stride(),
        device=q.device,
        dtype=torch.float32,
    )
    grid = (n_sample * n_heads * n_trunks,)
    _local_attention_with_bias_projection_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        z_flat,
        layernorm_weight,
        linear_weight,
        out,
        n_atoms,
        n_trunks,
        (n_keys - n_queries) // 2,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        q_flat.stride(3),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        k_flat.stride(3),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        v_flat.stride(3),
        z_flat.stride(0),
        z_flat.stride(1),
        z_flat.stride(2),
        z_flat.stride(3),
        z_flat.stride(4),
        linear_weight.stride(0),
        linear_weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_heads,
        c_z,
        eps,
        block_m=n_queries,
        block_n=n_keys,
        block_d=head_dim,
        dot_input_precision=triton_fused_local_attention_bias_attention_input_precision(),
        num_warps=triton_fused_local_attention_bias_attention_num_warps(),
    )
    if not batch_shape:
        return out[0]
    return out.reshape(*batch_shape, n_heads, n_atoms, head_dim)
