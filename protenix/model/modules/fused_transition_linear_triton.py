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


def triton_fused_silu_linear_enabled() -> bool:
    return os.getenv("PROTENIX_TRITON_FUSED_SILU_LINEAR", "0").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_fused_silu_linear_available() -> bool:
    return triton is not None and tl is not None


if triton_fused_silu_linear_available():

    @triton.jit
    def _silu_mul_linear_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        out_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        dot_input_precision: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        row_mask = offs_m < n_rows
        col_mask = offs_n < out_dim

        acc = tl.zeros([block_m, block_n], dtype=tl.float32)
        for k_start in range(0, hidden_dim, block_k):
            k = k_start + offs_k
            k_mask = k < hidden_dim
            x = tl.load(
                x_ptr + offs_m[:, None] * hidden_dim + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            y = tl.load(
                y_ptr + offs_m[:, None] * hidden_dim + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            activated = (x * (1.0 / (1.0 + tl.exp(-x))) * y).to(tl.bfloat16)
            weight = tl.load(
                weight_ptr + offs_n[None, :] * hidden_dim + k[:, None],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(activated, weight, input_precision=dot_input_precision)

        tl.store(
            out_ptr + offs_m[:, None] * out_dim + offs_n[None, :],
            acc,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def triton_silu_mul_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compute ``F.linear(F.silu(x) * y, weight)`` for atom-transition BF16.

    The current promoted path stores the hidden activation before the output
    projection.  This guarded path avoids that store/read pair for the stable
    atom transition shape ``hidden_dim=256, out_dim=128``.  Other shapes fall
    back to vendor GEMMs plus the existing fused elementwise kernel.
    """
    if not triton_fused_silu_linear_enabled():
        return None
    if not triton_fused_silu_linear_available():
        return None
    if torch.is_grad_enabled():
        return None
    if not x.is_cuda or not y.is_cuda or not weight.is_cuda:
        return None
    if x.dtype != torch.bfloat16 or y.dtype != x.dtype:
        return None
    if weight.dtype not in {torch.float32, torch.bfloat16}:
        return None
    if x.shape != y.shape or x.ndim != 3:
        return None
    if not x.is_contiguous() or not y.is_contiguous() or not weight.is_contiguous():
        return None
    hidden_dim = x.shape[-1]
    out_dim = weight.shape[0]
    if weight.shape != (out_dim, hidden_dim):
        return None
    if hidden_dim != 256 or out_dim != 128:
        return None

    out = torch.empty(
        (*x.shape[:-1], out_dim),
        device=x.device,
        dtype=x.dtype,
    )
    n_rows = x.numel() // hidden_dim
    grid = (triton.cdiv(n_rows, 16), triton.cdiv(out_dim, 64))
    _silu_mul_linear_kernel[grid](
        x,
        y,
        weight,
        out,
        n_rows,
        hidden_dim,
        out_dim,
        block_m=16,
        block_n=64,
        block_k=64,
        dot_input_precision="tf32",
        num_warps=4,
        num_stages=3,
    )
    return out
