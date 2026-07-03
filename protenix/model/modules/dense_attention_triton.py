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

"""Guarded Triton kernel for dense-bias pairformer token attention.

The practical batched-inference workload spends a material amount of time in
pairformer token attention.  That attention has a very specific shape:
``[batch, 16 heads, N tokens, head_dim 24]`` with a dense learned pair bias.
Generic SDPA is correct, but the H100 profile shows this boundary is still
expensive for the 40-220 token buckets we care about.

This module specializes only that narrow inference case.  It computes the full
``QK + bias -> softmax -> PV`` for one head/batch/query tile in one launch,
reading Protenix's existing q/k/v and bias strides directly.  Unsupported
shapes return ``None`` so callers use the original PyTorch path.  The env flag
is default-off until an end-to-end gate proves the candidate should be promoted.
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
_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16}


def triton_dense_attention_enabled() -> bool:
    return (
        os.getenv("PROTENIX_TRITON_D24_DENSE_ATTN", "0").strip().lower()
        not in _FALSE_ENV_VALUES
    )


def triton_dense_attention_available() -> bool:
    return triton is not None and tl is not None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_m() -> int:
    try:
        value = int(os.getenv("PROTENIX_TRITON_D24_DENSE_ATTN_BLOCK_M", "16"))
    except ValueError:
        return 16
    return value if value in {16, 32, 64} else 16


def _num_warps(n_tokens: int) -> int:
    explicit = os.getenv("PROTENIX_TRITON_D24_DENSE_ATTN_NUM_WARPS")
    if explicit is not None:
        try:
            value = int(explicit)
        except ValueError:
            return 4
        return value if value in {1, 2, 4, 8} else 4
    # Canary screens showed block_m=16 with four warps was best for the short
    # N=124 bucket and tied the best long-bucket core timing closely enough to
    # use one default.  Keep this as a launch choice so future gates can tune it
    # without code changes.
    return 4


def _dot_input_precision() -> str:
    precision = os.getenv("PROTENIX_TRITON_D24_DENSE_ATTN_INPUT_PRECISION", "tf32")
    return precision if precision in {"tf32", "tf32x3", "ieee"} else "tf32"


if triton_dense_attention_available():

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
        # Padded query rows appear only in the final tile.  They are not stored,
        # but keeping them finite avoids NaNs inside the row reduction.
        scores = tl.where(valid_m[:, None], scores, 0.0)
        scores = scores - tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=1)[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        )
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
    attn_bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return Triton attention output for the profiled D=24 case, else ``None``."""

    if (
        not triton_dense_attention_enabled()
        or not triton_dense_attention_available()
        or torch.is_grad_enabled()
        or attn_bias is None
    ):
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and attn_bias.is_cuda):
        return None
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        return None
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if attn_bias.shape != q.shape[:3] + (q.shape[-2],):
        return None

    batch, heads, n_tokens, head_dim = q.shape
    if head_dim != 24 or heads != 16:
        return None
    if n_tokens > 256:
        return None

    block_m = _block_m()
    block_n = _next_power_of_2(n_tokens)
    block_d = _next_power_of_2(head_dim)
    out = torch.empty_like(q)
    grid = (triton.cdiv(n_tokens, block_m), heads, batch)
    _dense_bias_attention_kernel[grid](
        q,
        k,
        v,
        attn_bias,
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
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        attn_bias.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_m,
        block_n,
        block_d,
        _dot_input_precision(),
        num_warps=_num_warps(n_tokens),
    )
    return out
