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

"""Fused pair-bias producer for atom local attention.

The unfused path normalizes pair features, projects them to attention heads,
then permutes to the local-attention bias layout.  At large ``N_sample`` those
intermediates are bandwidth-heavy even though the math is small.  This kernel
writes the final bias layout directly and removes the normalized intermediate.
As with the attention kernel, it is inference-only and falls back for any
unprofiled shape.
"""

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


_SUPPORTED_DTYPES = {torch.float32, torch.bfloat16}
_BLOCK_SIZE = 256
_FALSE_ENV_VALUES = {"0", "false", "off", "no"}


def triton_fused_local_attention_bias_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS")
    if value is None:
        # Match the local-attention default. The fused producer only helps when
        # the consumer can use the final bias layout. Opting out of local
        # attention should disable both sides unless this producer is explicitly
        # overridden.
        value = os.getenv("PROTENIX_TRITON_LOCAL_ATTN", "1")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def triton_fused_local_attention_bias_available() -> bool:
    return triton is not None and tl is not None


if triton_fused_local_attention_bias_available():

    @triton.jit
    def _project_local_attention_bias_kernel(
        z_ptr,
        ln_weight_ptr,
        linear_weight_ptr,
        out_ptr,
        total_pairs: tl.constexpr,
        n_trunks: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        c_z: tl.constexpr,
        n_heads: tl.constexpr,
        z_stride_s: tl.constexpr,
        z_stride_t: tl.constexpr,
        z_stride_q: tl.constexpr,
        z_stride_k: tl.constexpr,
        z_stride_c: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_q: tl.constexpr,
        out_stride_k: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
        mask = offsets < total_pairs

        key = offsets % n_keys
        tmp = offsets // n_keys
        query = tmp % n_queries
        tmp = tmp // n_queries
        trunk = tmp % n_trunks
        sample = tmp // n_trunks

        z_base = (
            z_ptr
            + sample * z_stride_s
            + trunk * z_stride_t
            + query * z_stride_q
            + key * z_stride_k
        )

        sum_z = tl.zeros((block_size,), tl.float32)
        sum_z2 = tl.zeros((block_size,), tl.float32)
        weighted0 = tl.zeros((block_size,), tl.float32)
        weighted1 = tl.zeros((block_size,), tl.float32)
        weighted2 = tl.zeros((block_size,), tl.float32)
        weighted3 = tl.zeros((block_size,), tl.float32)
        weight_sum0 = tl.full((), 0.0, tl.float32)
        weight_sum1 = tl.full((), 0.0, tl.float32)
        weight_sum2 = tl.full((), 0.0, tl.float32)
        weight_sum3 = tl.full((), 0.0, tl.float32)
        for c in range(0, c_z):
            z_c = tl.load(
                z_base + c * z_stride_c,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            ln_w = tl.load(ln_weight_ptr + c).to(tl.float32)
            linear_w0 = tl.load(linear_weight_ptr + c).to(tl.float32)
            linear_w1 = tl.load(linear_weight_ptr + c_z + c).to(tl.float32)
            linear_w2 = tl.load(linear_weight_ptr + 2 * c_z + c).to(tl.float32)
            linear_w3 = tl.load(linear_weight_ptr + 3 * c_z + c).to(tl.float32)
            w0 = ln_w * linear_w0
            w1 = ln_w * linear_w1
            w2 = ln_w * linear_w2
            w3 = ln_w * linear_w3
            sum_z += z_c
            sum_z2 += z_c * z_c
            weighted0 += z_c * w0
            weighted1 += z_c * w1
            weighted2 += z_c * w2
            weighted3 += z_c * w3
            weight_sum0 += w0
            weight_sum1 += w1
            weight_sum2 += w2
            weight_sum3 += w3

        inv_c = 1.0 / c_z
        mean = sum_z * inv_c
        var = sum_z2 * inv_c - mean * mean
        rstd = tl.rsqrt(var + eps)

        for head in range(0, 4):
            if head == 0:
                projected = (weighted0 - mean * weight_sum0) * rstd
            elif head == 1:
                projected = (weighted1 - mean * weight_sum1) * rstd
            elif head == 2:
                projected = (weighted2 - mean * weight_sum2) * rstd
            else:
                projected = (weighted3 - mean * weight_sum3) * rstd
            out_offsets = (
                out_ptr
                + sample * out_stride_s
                + head * out_stride_h
                + trunk * out_stride_t
                + query * out_stride_q
                + key * out_stride_k
            )
            tl.store(out_offsets, projected, mask=mask)


def triton_project_local_attention_bias(
    z: torch.Tensor,
    layernorm_weight: Optional[torch.Tensor],
    linear_weight: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    eps: float,
) -> Optional[torch.Tensor]:
    """Fuse atom local-attention pair-bias LayerNorm, projection, and permute.

    The unfused local path computes:

    ``linear_nobias_z(layernorm_z(z)).movedim(-1, -4)``

    for ``z`` shaped ``[..., n_trunks, 32, 128, c_z]``.  At high ``N_sample``
    this materializes both the normalized pair tensor and a multi-GiB trunked
    bias tensor.  This kernel removes the normalized intermediate and writes
    the final ``[..., n_heads, n_trunks, 32, 128]`` layout directly.
    """
    if not triton_fused_local_attention_bias_enabled():
        return None
    if not triton_fused_local_attention_bias_available():
        return None
    if torch.is_grad_enabled():
        return None
    if not z.is_cuda:
        return None
    if z.dtype not in _SUPPORTED_DTYPES:
        return None
    if layernorm_weight is None:
        return None
    if not layernorm_weight.is_cuda or not linear_weight.is_cuda:
        return None
    if z.ndim < 4:
        return None
    if z.shape[-3:-1] != (n_queries, n_keys):
        return None
    if n_queries != 32 or n_keys != 128:
        return None

    c_z = z.shape[-1]
    n_heads = linear_weight.shape[0]
    if n_heads != 4:
        return None
    if c_z not in {16, 32}:
        return None
    if layernorm_weight.shape != (c_z,):
        return None
    if linear_weight.shape != (n_heads, c_z):
        return None
    if layernorm_weight.dtype != linear_weight.dtype:
        return None
    if linear_weight.dtype not in _SUPPORTED_DTYPES:
        return None

    batch_shape = z.shape[:-4]
    n_trunks = z.shape[-4]
    n_sample = prod(batch_shape) if batch_shape else 1
    flat_z_shape = (n_sample, n_trunks, n_queries, n_keys, c_z)
    z_flat = z.reshape(flat_z_shape)
    if z_flat.data_ptr() != z.data_ptr():
        return None
    if not layernorm_weight.is_contiguous() or not linear_weight.is_contiguous():
        return None

    out = torch.empty(
        (n_sample, n_heads, n_trunks, n_queries, n_keys),
        dtype=z.dtype,
        device=z.device,
    )
    total_pairs = n_sample * n_trunks * n_queries * n_keys
    grid = (triton.cdiv(total_pairs, _BLOCK_SIZE),)
    _project_local_attention_bias_kernel[grid](
        z_flat,
        layernorm_weight,
        linear_weight,
        out,
        total_pairs,
        n_trunks,
        n_queries,
        n_keys,
        c_z,
        n_heads,
        z_flat.stride(0),
        z_flat.stride(1),
        z_flat.stride(2),
        z_flat.stride(3),
        z_flat.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        eps,
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    if not batch_shape:
        return out[0]
    return out.reshape(*batch_shape, n_heads, n_trunks, n_queries, n_keys)
