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

"""Triton transition input projection for large inference batches.

Pairformer transition computes two GEMMs from the same normalized pair rows:

    a = y @ Wa.T
    b = y @ Wb.T
    hidden = silu(a) * b

The default PyTorch path writes both pre-activation matrices to HBM before the
cheap elementwise product.  For large `[B, N, N, C]` pair tensors those writes
are a real bottleneck.  This helper keeps both accumulators inside one Triton
tile and stores only the consumed `hidden` tensor.  It deliberately does *not*
replace the following output projection, where cuBLAS is still the right tool.
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
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 64


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in _FALSE_ENV_VALUES


def triton_transition_dual_gemm_enabled() -> bool:
    return _env_flag_enabled("PROTENIX_TRITON_TRANSITION_DUAL_GEMM", default=False)


def triton_transition_dual_gemm_available() -> bool:
    return triton is not None and tl is not None


def _min_rows() -> int:
    raw = os.getenv("PROTENIX_TRITON_TRANSITION_DUAL_GEMM_MIN_ROWS")
    if raw is None:
        return 262_144
    try:
        return max(0, int(raw))
    except ValueError:
        return 262_144


def _max_output_elements() -> int:
    raw = os.getenv("PROTENIX_TRITON_TRANSITION_DUAL_GEMM_MAX_OUTPUT_ELEMENTS")
    if raw is None:
        return 4_000_000_000
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


if triton_transition_dual_gemm_available():

    @triton.jit
    def _dual_gemm_silu_product_kernel(
        y_ptr,
        wa_ptr,
        wb_ptr,
        out_ptr,
        m_rows,
        stride_y_m: tl.constexpr,
        stride_wa_n: tl.constexpr,
        stride_wb_n: tl.constexpr,
        stride_out_m: tl.constexpr,
        n_hidden: tl.constexpr,
        k_input: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
        offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
        offs_k = tl.arange(0, block_k).to(tl.int64)

        acc_a = tl.zeros((block_m, block_n), dtype=tl.float32)
        acc_b = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k0 in range(0, k_input, block_k):
            k = k0 + offs_k
            y = tl.load(
                y_ptr + offs_m[:, None] * stride_y_m + k[None, :],
                mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_input),
                other=0.0,
            )
            wa = tl.load(
                wa_ptr + offs_n[:, None] * stride_wa_n + k[None, :],
                mask=(offs_n[:, None] < n_hidden) & (k[None, :] < k_input),
                other=0.0,
            )
            wb = tl.load(
                wb_ptr + offs_n[:, None] * stride_wb_n + k[None, :],
                mask=(offs_n[:, None] < n_hidden) & (k[None, :] < k_input),
                other=0.0,
            )
            acc_a += tl.dot(y, tl.trans(wa))
            acc_b += tl.dot(y, tl.trans(wb))

        out = (acc_a / (1.0 + tl.exp(-acc_a))) * acc_b
        tl.store(
            out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_hidden),
        )


def triton_dual_gemm_silu_product(
    y: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not triton_transition_dual_gemm_enabled():
        return None
    if not triton_transition_dual_gemm_available():
        return None
    if torch.is_grad_enabled():
        return None
    if y.dtype not in _SUPPORTED_DTYPES:
        return None
    if not y.is_cuda or not weight_a.is_cuda or not weight_b.is_cuda:
        return None
    if (
        not y.is_contiguous()
        or not weight_a.is_contiguous()
        or not weight_b.is_contiguous()
    ):
        return None
    if weight_a.dtype != y.dtype or weight_b.dtype != y.dtype:
        return None
    if y.dim() != 2 or weight_a.dim() != 2 or weight_b.dim() != 2:
        return None
    if weight_a.shape != weight_b.shape:
        return None
    m_rows, k_input = y.shape
    n_hidden, weight_k = weight_a.shape
    if weight_k != k_input or m_rows < _min_rows():
        return None
    if m_rows * n_hidden > _max_output_elements():
        return None

    out = torch.empty((m_rows, n_hidden), dtype=y.dtype, device=y.device)
    grid = (triton.cdiv(m_rows, _BLOCK_M), triton.cdiv(n_hidden, _BLOCK_N))
    _dual_gemm_silu_product_kernel[grid](
        y,
        weight_a,
        weight_b,
        out,
        m_rows,
        y.stride(0),
        weight_a.stride(0),
        weight_b.stride(0),
        out.stride(0),
        n_hidden,
        k_input,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out
