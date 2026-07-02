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

"""Experimental fused Triton forward for triangle attention.

The promoted CUEQ path is fast, but the post-QKV profile shows a costly
producer/consumer boundary: fused qkv is rewritten into CUEQ layout, CUEQ wraps
cuDNN attention with several nvjet transforms, then a Triton epilogue gates and
flattens the heads.  This prototype tests the larger boundary directly:

``qkv projection -> triangular attention with key bias/mask -> sigmoid gate``

in one inference-only kernel that writes the flattened ``[..., Q, H * D]``
layout consumed by the output projection.  It is intentionally opt-in while we
learn whether this larger fused kernel can beat the vendor CUEQ/cuDNN path.
"""

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


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_SUPPORTED_DTYPES = {torch.bfloat16}
_BLOCK_M = 32
_BLOCK_N = 64


def triton_triangle_attention_fused_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TRIANGLE_ATTENTION_FUSED", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_attention_fused_available() -> bool:
    return triton is not None and tl is not None


def _can_use_fused_triangle_attention(
    qkv: torch.Tensor,
    triangle_bias: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor,
    no_heads: int,
) -> bool:
    if not triton_triangle_attention_fused_enabled():
        return False
    if not triton_triangle_attention_fused_available():
        return False
    if torch.is_grad_enabled():
        return False
    if qkv.dtype not in _SUPPORTED_DTYPES:
        return False
    if not (qkv.is_cuda and triangle_bias.is_cuda and mask.is_cuda and gate.is_cuda):
        return False
    if not (qkv.is_contiguous() and triangle_bias.is_contiguous() and gate.is_contiguous()):
        return False
    if not mask.is_contiguous():
        return False
    if qkv.dim() != 4 or triangle_bias.dim() != 4 or mask.dim() != 3:
        return False
    if gate.shape != qkv.shape[:-1] + (qkv.shape[-1] // 3,):
        return False
    batch, rows, cols, qkv_channels = qkv.shape
    if rows != cols or triangle_bias.shape != (batch, rows, cols, no_heads):
        return False
    if mask.shape != (batch, rows, cols):
        return False
    if no_heads <= 0 or qkv_channels % (3 * no_heads) != 0:
        return False
    # The first prototype specializes the real pairformer shape.  Generalizing
    # head dimensions before the boundary wins would only make the kernel harder
    # to reason about and tune.
    return qkv_channels // (3 * no_heads) == 32


if triton_triangle_attention_fused_available():

    @triton.jit
    def _triangle_attention_qkv_gate_kernel(
        qkv_ptr,
        bias_ptr,
        mask_ptr,
        gate_ptr,
        out_ptr,
        n_tokens: tl.constexpr,
        no_heads: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        bih = tl.program_id(1)

        h = bih % no_heads
        tmp = bih // no_heads
        i = tmp % n_tokens
        b = tmp // n_tokens

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, head_dim)
        qkv_channels = 3 * no_heads * head_dim
        head_channels = no_heads * head_dim

        q_offsets = (
            ((b * n_tokens + i) * n_tokens + offs_m[:, None]) * qkv_channels
            + h * head_dim
            + offs_d[None, :]
        )
        q_mask = offs_m[:, None] < n_tokens
        q = tl.load(qkv_ptr + q_offsets, mask=q_mask, other=0.0)

        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, head_dim), tl.float32)

        for k_start in range(0, n_tokens, block_n):
            k_idxs = k_start + offs_n
            kv_mask = k_idxs[:, None] < n_tokens
            k_offsets = (
                ((b * n_tokens + i) * n_tokens + k_idxs[:, None]) * qkv_channels
                + head_channels
                + h * head_dim
                + offs_d[None, :]
            )
            v_offsets = k_offsets + head_channels
            k = tl.load(qkv_ptr + k_offsets, mask=kv_mask, other=0.0)
            v = tl.load(qkv_ptr + v_offsets, mask=kv_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k)) * scale
            bias_offsets = (
                ((b * n_tokens + offs_m[:, None]) * n_tokens + k_idxs[None, :])
                * no_heads
                + h
            )
            bias = tl.load(
                bias_ptr + bias_offsets,
                mask=(offs_m[:, None] < n_tokens) & (k_idxs[None, :] < n_tokens),
                other=0.0,
            ).to(tl.float32)
            key_mask = tl.load(
                mask_ptr + (b * n_tokens + i) * n_tokens + k_idxs,
                mask=k_idxs < n_tokens,
                other=0.0,
            )
            valid = (
                (offs_m[:, None] < n_tokens)
                & (k_idxs[None, :] < n_tokens)
                & (key_mask[None, :] > 0.5)
            )
            scores = tl.where(valid, scores + bias, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_offsets = (
            ((b * n_tokens + i) * n_tokens + offs_m[:, None]) * head_channels
            + h * head_dim
            + offs_d[None, :]
        )
        gate = tl.load(gate_ptr + out_offsets, mask=q_mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-gate))) * acc
        tl.store(out_ptr + out_offsets, out, mask=q_mask)


def triton_triangle_attention_qkv_gate(
    qkv: torch.Tensor,
    triangle_bias: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor,
    no_heads: int,
    scale: float,
) -> Optional[torch.Tensor]:
    """Return gated/flattened triangle-attention output, or ``None`` to fall back."""
    if not _can_use_fused_triangle_attention(qkv, triangle_bias, mask, gate, no_heads):
        return None

    head_dim = qkv.shape[-1] // (3 * no_heads)
    out = torch.empty_like(gate)
    grid = (triton.cdiv(qkv.shape[-2], _BLOCK_M), qkv.shape[0] * qkv.shape[1] * no_heads)
    _triangle_attention_qkv_gate_kernel[grid](
        qkv,
        triangle_bias,
        mask,
        gate,
        out,
        qkv.shape[1],
        no_heads,
        head_dim,
        scale,
        _BLOCK_M,
        _BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return out
