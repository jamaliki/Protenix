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
from math import prod
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None


def triton_local_attention_enabled() -> bool:
    return os.getenv("PROTENIX_TRITON_LOCAL_ATTN", "0").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_local_attention_available() -> bool:
    return triton is not None and tl is not None

_SUPPORTED_INPUT_DTYPES = {torch.float32, torch.bfloat16}


if triton_local_attention_available():

    @triton.jit
    def _local_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
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
        b_stride_s: tl.constexpr,
        b_stride_h: tl.constexpr,
        b_stride_t: tl.constexpr,
        b_stride_q: tl.constexpr,
        b_stride_k: tl.constexpr,
        o_stride_s: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_n: tl.constexpr,
        o_stride_d: tl.constexpr,
        n_heads: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
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

        q_base = q_ptr + sample * q_stride_s + head * q_stride_h
        k_base = k_ptr + sample * k_stride_s + head * k_stride_h
        v_base = v_ptr + sample * v_stride_s + head * v_stride_h
        b_base = (
            bias_ptr
            + sample * b_stride_s
            + head * b_stride_h
            + trunk * b_stride_t
        )

        q_mask = q_idx < n_atoms
        k_mask = (k_idx >= 0) & (k_idx < n_atoms)
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
        bias = tl.load(
            b_base + offs_m[:, None] * b_stride_q + offs_n[None, :] * b_stride_k,
            mask=q_mask[:, None] & k_mask[None, :],
            other=-float("inf"),
        )
        score = tl.dot(q, tl.trans(k), input_precision="tf32x3") + bias
        score = tl.where(q_mask[:, None] & k_mask[None, :], score, -float("inf"))
        # The final trunk may contain padded query rows. Keep those rows finite
        # through the reduction even though they are masked out on store.
        score = tl.where(q_mask[:, None], score, 0.0)
        score_max = tl.max(score, 1)
        score = score - score_max[:, None]
        prob = tl.exp(score)
        prob = prob / tl.sum(prob, 1)[:, None]

        v = tl.load(
            v_base + k_idx[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=k_mask[:, None],
            other=0.0,
        )
        v = v.to(tl.float32)
        out = tl.dot(prob, v, input_precision="tf32x3")
        out_base = out_ptr + sample * o_stride_s + head * o_stride_h
        tl.store(
            out_base + q_idx[:, None] * o_stride_n + offs_d[None, :] * o_stride_d,
            out,
            mask=q_mask[:, None],
        )


def triton_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    trunked_attn_bias: torch.Tensor,
    n_queries: int,
    n_keys: int,
) -> Optional[torch.Tensor]:
    """Run the H100-specialized atom local attention forward path.

    This intentionally handles only the diffusion atom-transformer shape:
    q/k/v are [..., n_heads, N_atom, head_dim] and bias is
    [..., n_heads, n_trunks, 32, 128]. Leading batch dimensions are flattened
    into the kernel's sample axis when that can be done as a view. Other shapes
    return None so the caller can use the existing PyTorch path.
    """
    if not triton_local_attention_enabled():
        return None
    if not triton_local_attention_available():
        return None
    if torch.is_grad_enabled():
        return None
    if not q.is_cuda:
        return None
    if q.ndim < 3:
        return None
    if trunked_attn_bias.ndim != q.ndim + 1:
        return None
    if n_queries != 32 or n_keys != 128:
        return None
    if q.shape != k.shape or q.shape != v.shape:
        return None
    if q.shape[-1] != 32:
        return None
    if q.dtype not in _SUPPORTED_INPUT_DTYPES:
        return None
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if trunked_attn_bias.dtype not in _SUPPORTED_INPUT_DTYPES:
        return None

    batch_shape = q.shape[:-3]
    n_heads, n_atoms, head_dim = q.shape[-3:]
    n_trunks = (n_atoms + n_queries - 1) // n_queries
    bias_batch_shape = trunked_attn_bias.shape[:-4]
    if trunked_attn_bias.shape[-4:] != (n_heads, n_trunks, n_queries, n_keys):
        return None
    if len(bias_batch_shape) != len(batch_shape) or any(
        bias_dim not in {1, q_dim}
        for bias_dim, q_dim in zip(bias_batch_shape, batch_shape)
    ):
        return None

    n_sample = prod(batch_shape) if batch_shape else 1
    flat_q_shape = (n_sample, n_heads, n_atoms, head_dim)
    flat_bias_shape = (n_sample, n_heads, n_trunks, n_queries, n_keys)
    q_flat = q.reshape(flat_q_shape)
    k_flat = k.reshape(flat_q_shape)
    v_flat = v.reshape(flat_q_shape)
    bias_view = trunked_attn_bias.expand(
        *batch_shape,
        n_heads,
        n_trunks,
        n_queries,
        n_keys,
    )
    bias_flat = bias_view.reshape(flat_bias_shape)
    if (
        q_flat.data_ptr() != q.data_ptr()
        or k_flat.data_ptr() != k.data_ptr()
        or v_flat.data_ptr() != v.data_ptr()
        or bias_flat.data_ptr() != trunked_attn_bias.data_ptr()
    ):
        return None

    # The reference local-attention path forces atom attention to FP32 before
    # SDPA and returns FP32 even when the surrounding model is BF16.
    out = torch.empty_strided(
        flat_q_shape,
        q_flat.stride(),
        device=q.device,
        dtype=torch.float32,
    )
    grid = (n_sample * n_heads * n_trunks,)
    _local_attention_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        bias_flat,
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
        bias_flat.stride(0),
        bias_flat.stride(1),
        bias_flat.stride(2),
        bias_flat.stride(3),
        bias_flat.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_heads,
        block_m=n_queries,
        block_n=n_keys,
        block_d=head_dim,
        num_warps=4,
    )
    if not batch_shape:
        return out[0]
    return out.reshape(*batch_shape, n_heads, n_atoms, head_dim)
