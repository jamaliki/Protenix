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

"""Triton helpers for the pairformer ending triangle-attention boundary.

Ending triangle attention wants to attend over the opposite pair axis.  The
baseline materializes ``z.transpose(-2, -3).contiguous()``, layer-normalizes
that tensor, runs CUEQ attention, adds the residual in transposed layout, then
materializes the opposite transpose to return to normal pair layout.

These helpers screen a narrower fusion boundary: compute LayerNorm while
writing the transposed layout that CUEQ needs, then add the transposed update
back into the original ``z`` in one streaming kernel.  This does not replace
CUEQ; it removes avoidable full-pair layout traffic around it.
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
_BLOCK_ROWS = 256


def triton_fused_ending_attention_transpose_enabled() -> bool:
    return os.getenv(
        "PROTENIX_TRITON_FUSED_ENDING_ATTENTION_TRANSPOSE", "0"
    ).lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_fused_ending_attention_transpose_available() -> bool:
    return triton is not None and tl is not None


def _can_use_transpose_kernels(z: torch.Tensor) -> bool:
    if not triton_fused_ending_attention_transpose_enabled():
        return False
    if not triton_fused_ending_attention_transpose_available():
        return False
    if torch.is_grad_enabled():
        return False
    if not z.is_cuda or not z.is_contiguous():
        return False
    if z.dtype not in _SUPPORTED_DTYPES:
        return False
    if z.dim() < 4 or z.shape[-3] != z.shape[-2] or z.shape[-1] != 128:
        return False
    return z.numel() > 0


if triton_fused_ending_attention_transpose_available():

    @triton.jit
    def _transpose_layer_norm_128_kernel(
        z_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_pairs: tl.constexpr,
        n_tokens: tl.constexpr,
        eps: tl.constexpr,
        has_bias: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        rows = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        row_mask = rows < total_pairs

        j = rows % n_tokens
        tmp = rows // n_tokens
        i = tmp % n_tokens
        batch = tmp // n_tokens

        input_base = rows * 128
        output_base = ((batch * n_tokens + j) * n_tokens + i) * 128

        sum_x = tl.zeros((block_rows,), tl.float32)
        sum_x2 = tl.zeros((block_rows,), tl.float32)
        for c in range(0, 128):
            x = tl.load(z_ptr + input_base + c, mask=row_mask, other=0.0).to(
                tl.float32
            )
            sum_x += x
            sum_x2 += x * x

        mean = sum_x * (1.0 / 128.0)
        var = sum_x2 * (1.0 / 128.0) - mean * mean
        rstd = tl.rsqrt(var + eps)

        for c in range(0, 128):
            x = tl.load(z_ptr + input_base + c, mask=row_mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(weight_ptr + c).to(tl.float32)
            y = (x - mean) * rstd * w
            if has_bias:
                y += tl.load(bias_ptr + c).to(tl.float32)
            tl.store(out_ptr + output_base + c, y, mask=row_mask)

    @triton.jit
    def _transpose_add_128_kernel(
        z_ptr,
        update_t_ptr,
        total_pairs: tl.constexpr,
        n_tokens: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        rows = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        row_mask = rows < total_pairs

        j = rows % n_tokens
        tmp = rows // n_tokens
        i = tmp % n_tokens
        batch = tmp // n_tokens

        z_base = rows * 128
        update_base = ((batch * n_tokens + j) * n_tokens + i) * 128
        for c in range(0, 128):
            z = tl.load(z_ptr + z_base + c, mask=row_mask, other=0.0).to(tl.float32)
            update = tl.load(
                update_t_ptr + update_base + c, mask=row_mask, other=0.0
            ).to(tl.float32)
            tl.store(z_ptr + z_base + c, z + update, mask=row_mask)


def triton_transpose_layer_norm_128(
    z: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    *,
    eps: float,
) -> Optional[torch.Tensor]:
    """Return ``LayerNorm(z).transpose(-2, -3).contiguous()`` for C=128.

    The output is contiguous in the transposed physical layout expected by the
    existing CUEQ starting-attention kernel.
    """
    if not _can_use_transpose_kernels(z):
        return None
    if weight is None or not weight.is_cuda or weight.numel() != 128:
        return None
    if weight.dtype != z.dtype:
        return None
    if bias is not None and (
        not bias.is_cuda or bias.numel() != 128 or bias.dtype != z.dtype
    ):
        return None

    out = torch.empty_like(z)
    leading = prod(z.shape[:-3])
    total_pairs = leading * z.shape[-3] * z.shape[-2]
    grid = (triton.cdiv(total_pairs, _BLOCK_ROWS),)
    _transpose_layer_norm_128_kernel[grid](
        z,
        weight,
        bias if bias is not None else weight,
        out,
        total_pairs,
        z.shape[-2],
        eps,
        bias is not None,
        _BLOCK_ROWS,
    )
    return out


def triton_transpose_add_128_(z: torch.Tensor, update_t: torch.Tensor) -> bool:
    """In-place ``z += update_t.transpose(-2, -3)`` for C=128."""
    if not _can_use_transpose_kernels(z):
        return False
    if update_t.shape != z.shape or update_t.dtype != z.dtype:
        return False
    if not update_t.is_cuda or not update_t.is_contiguous():
        return False

    leading = prod(z.shape[:-3])
    total_pairs = leading * z.shape[-3] * z.shape[-2]
    grid = (triton.cdiv(total_pairs, _BLOCK_ROWS),)
    _transpose_add_128_kernel[grid](
        z,
        update_t,
        total_pairs,
        z.shape[-2],
        _BLOCK_ROWS,
    )
    return True
