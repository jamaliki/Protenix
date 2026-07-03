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

"""Triton layout producer for cuequivariance triangle attention.

The triangle-attention q/k/v projection naturally leaves the fused GEMM output
in ``[..., I, J, 3 * H * D]`` layout.  CUEQ consumes ``[..., I, H, J, D]``.
The baseline obtains that layout with view/transpose metadata and then the CUEQ
wrapper materializes separate contiguous q/k/v copies before calling its CUDA
op.  At high sequence batch those copies move several GiB per triangle
attention, so they show up clearly in kernel-level profiles.

This helper keeps the vendor GEMM intact and changes only the producer/consumer
boundary: split the fused qkv projection and write the CUEQ physical layout
directly in one streaming pass.  If any guard fails, the caller falls back to
the original view/transpose path.
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


_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16, torch.float32}
_BLOCK_SIZE = 1024
_FALSE_ENV_VALUES = {"0", "false", "off", "no"}


def triton_cueq_qkv_layout_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_CUEQ_QKV_LAYOUT", "1")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_cueq_qkv_layout_available() -> bool:
    return triton is not None and tl is not None


def _can_use_qkv_layout(qkv: torch.Tensor, no_heads: int) -> bool:
    if not triton_cueq_qkv_layout_enabled():
        return False
    if not triton_cueq_qkv_layout_available():
        return False
    if torch.is_grad_enabled():
        return False
    if not qkv.is_cuda or not qkv.is_contiguous():
        return False
    if qkv.dtype not in _SUPPORTED_DTYPES:
        return False
    if qkv.dim() < 4 or qkv.numel() == 0:
        return False
    if no_heads <= 0 or qkv.shape[-1] % (3 * no_heads) != 0:
        return False
    return True


if triton_cueq_qkv_layout_available():

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
        # B96/N245 already addresses more than 2**31 elements in the fused qkv
        # tensor.  Keep offsets in int64 so pointer arithmetic cannot silently
        # wrap and corrupt q/k/v at the target many-sequence batch sizes.
        offsets = (
            tl.program_id(0) * block_size + tl.arange(0, block_size)
        ).to(tl.int64)
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
        # Ending-node triangle attention is mathematically the starting-node
        # computation on x.transpose(I, J).  If projections are computed on the
        # original contiguous x, this producer writes CUEQ's required
        # [..., J, H, I, D] physical layout directly instead of first creating
        # non-contiguous transposed projection inputs.
        offsets = (
            tl.program_id(0) * block_size + tl.arange(0, block_size)
        ).to(tl.int64)
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


def triton_qkv_to_cueq_heads_first(
    qkv: torch.Tensor,
    no_heads: int,
    *,
    swap_pair_axes: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return contiguous q/k/v in CUEQ's physical heads-first layout.

    The normal layout is ``[..., I, H, J, D]``.  ``swap_pair_axes=True`` is a
    benchmark/prototype mode for ending-node attention: it writes
    ``[..., J, H, I, D]`` directly from a projection computed on the original
    contiguous ``[..., I, J, 3*H*D]`` tensor.
    """
    if not _can_use_qkv_layout(qkv, no_heads):
        return None

    head_dim = qkv.shape[-1] // (3 * no_heads)
    if swap_pair_axes:
        output_shape = qkv.shape[:-3] + (
            qkv.shape[-2],
            no_heads,
            qkv.shape[-3],
            head_dim,
        )
    else:
        output_shape = qkv.shape[:-2] + (no_heads, qkv.shape[-2], head_dim)
    q = torch.empty(output_shape, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    if swap_pair_axes:
        leading = prod(qkv.shape[:-3])
        total_elements = leading * qkv.shape[-2] * no_heads * qkv.shape[-3] * head_dim
        grid = (triton.cdiv(total_elements, _BLOCK_SIZE),)
        _qkv_to_heads_first_swapped_kernel[grid](
            qkv,
            q,
            k,
            v,
            total_elements,
            qkv.shape[-3],
            qkv.shape[-2],
            no_heads,
            head_dim,
            _BLOCK_SIZE,
            num_warps=4,
        )
        return q, k, v

    leading_i = prod(qkv.shape[:-2])
    total_elements = leading_i * no_heads * qkv.shape[-2] * head_dim
    grid = (triton.cdiv(total_elements, _BLOCK_SIZE),)
    _qkv_to_heads_first_kernel[grid](
        qkv,
        q,
        k,
        v,
        total_elements,
        qkv.shape[-2],
        no_heads,
        head_dim,
        _BLOCK_SIZE,
        num_warps=4,
    )
    return q, k, v
