# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None


def triton_full_attention_enabled() -> bool:
    return os.getenv("PROTENIX_TRITON_FULL_ATTN", "0").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_full_attention_available() -> bool:
    return triton is not None and tl is not None


def triton_full_attention_input_precision() -> str:
    precision = os.getenv("PROTENIX_TRITON_FULL_ATTN_INPUT_PRECISION", "tf32").lower()
    if precision in {"tf32x3", "tf32", "ieee"}:
        return precision
    return "tf32"


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


if triton_full_attention_available():

    @triton.jit
    def _full_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
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
        b_stride_b: tl.constexpr,
        b_stride_h: tl.constexpr,
        b_stride_q: tl.constexpr,
        b_stride_k: tl.constexpr,
        o_stride_b: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_n: tl.constexpr,
        o_stride_d: tl.constexpr,
        bias_batches: tl.constexpr,
        bias_heads: tl.constexpr,
        n_heads: tl.constexpr,
        n_tokens: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        dot_input_precision: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // n_heads
        head = pid_bh - batch * n_heads

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        m_mask = offs_m < n_tokens
        d_mask = offs_d < head_dim

        q_base = q_ptr + batch * q_stride_b + head * q_stride_h
        k_base = k_ptr + batch * k_stride_b + head * k_stride_h
        v_base = v_ptr + batch * v_stride_b + head * v_stride_h
        o_base = out_ptr + batch * o_stride_b + head * o_stride_h

        bias_batch = 0 if bias_batches == 1 else batch
        bias_head = 0 if bias_heads == 1 else head
        b_base = bias_ptr + bias_batch * b_stride_b + bias_head * b_stride_h

        q = tl.load(
            q_base + offs_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        row_max = tl.full((block_m,), -float("inf"), tl.float32)
        row_sum = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, block_d), tl.float32)

        for start_n in range(0, n_tokens, block_n):
            cols = start_n + offs_n
            n_mask = cols < n_tokens
            k = tl.load(
                k_base + cols[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), input_precision=dot_input_precision)
            bias = tl.load(
                b_base + offs_m[:, None] * b_stride_q + cols[None, :] * b_stride_k,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            scores = scores + bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            scores = tl.where(m_mask[:, None], scores, 0.0)

            next_max = tl.maximum(row_max, tl.max(scores, 1))
            old_scale = tl.exp(row_max - next_max)
            probs = tl.exp(scores - next_max[:, None])
            probs = tl.where(m_mask[:, None] & n_mask[None, :], probs, 0.0)

            v = tl.load(
                v_base + cols[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = v.to(tl.float32)
            acc = acc * old_scale[:, None] + tl.dot(
                probs, v, input_precision=dot_input_precision
            )
            row_sum = row_sum * old_scale + tl.sum(probs, 1)
            row_max = next_max

        out = acc / row_sum[:, None]
        tl.store(
            o_base + offs_m[:, None] * o_stride_n + offs_d[None, :] * o_stride_d,
            out,
            mask=m_mask[:, None] & d_mask[None, :],
        )


def triton_full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """H100-oriented full token attention for Protenix trunk inference.

    This intentionally handles only the fixed small-token full-attention family
    used by the diffusion trunk: q/k/v are `[samples, heads, tokens, head_dim]`,
    tokens are at most 256, and pair bias is broadcastable as
    `[1|samples, 1|heads, tokens, tokens]`.
    """
    if not triton_full_attention_enabled():
        return None
    if not triton_full_attention_available():
        return None
    if torch.is_grad_enabled():
        return None
    if attn_bias is None:
        return None
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return None
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not attn_bias.is_cuda:
        return None
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if attn_bias.dtype != q.dtype:
        return None

    n_batches, n_heads, n_tokens, head_dim = q.shape
    if n_tokens > 256 or head_dim > 64:
        return None
    if attn_bias.ndim != 4 or attn_bias.shape[-2:] != (n_tokens, n_tokens):
        return None
    if attn_bias.shape[0] not in {1, n_batches}:
        return None
    if attn_bias.shape[1] not in {1, n_heads}:
        return None

    block_m = 32
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(n_tokens, block_m), n_batches * n_heads)
    _full_attention_kernel[grid](
        q,
        k,
        v,
        attn_bias,
        out,
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
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        attn_bias.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        attn_bias.shape[0],
        attn_bias.shape[1],
        n_heads,
        n_tokens,
        head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        dot_input_precision=triton_full_attention_input_precision(),
        num_warps=4,
    )
    return out
