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
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_BLOCK_SIZE = 1024


def triton_fused_elementwise_enabled() -> bool:
    return os.getenv("PROTENIX_TRITON_FUSED_ELEMENTWISE", "0").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def triton_fused_elementwise_available() -> bool:
    return triton is not None and tl is not None


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    if not triton_fused_elementwise_enabled():
        return False
    if not triton_fused_elementwise_available():
        return False
    if torch.is_grad_enabled():
        return False
    if not tensors:
        return False
    shape = tensors[0].shape
    dtype = tensors[0].dtype
    if dtype not in _SUPPORTED_DTYPES:
        return False
    for tensor in tensors:
        if not tensor.is_cuda:
            return False
        if tensor.shape != shape:
            return False
        if tensor.dtype != dtype:
            return False
        if not tensor.is_contiguous():
            return False
    return tensors[0].numel() > 0


def _can_use_triton_strided_y(x: torch.Tensor, y: torch.Tensor) -> bool:
    if not triton_fused_elementwise_enabled():
        return False
    if not triton_fused_elementwise_available():
        return False
    if torch.is_grad_enabled():
        return False
    if x.shape != y.shape:
        return False
    if x.dtype != y.dtype or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_cuda or not y.is_cuda:
        return False
    if x.ndim != 4:
        return False
    if not x.is_contiguous():
        return False
    return y.numel() > 0


if triton_fused_elementwise_available():

    @triton.jit
    def _sigmoid_mul_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _sigmoid_mul_4d_strided_y_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        dim0: tl.constexpr,
        dim1: tl.constexpr,
        dim2: tl.constexpr,
        dim3: tl.constexpr,
        y_stride0: tl.constexpr,
        y_stride1: tl.constexpr,
        y_stride2: tl.constexpr,
        y_stride3: tl.constexpr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        d3 = offsets % dim3
        t = offsets // dim3
        d2 = t % dim2
        t = t // dim2
        d1 = t % dim1
        d0 = t // dim1
        y_offsets = (
            d0 * y_stride0
            + d1 * y_stride1
            + d2 * y_stride2
            + d3 * y_stride3
        )
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _sigmoid_mul_add_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-x))) * y + z
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _silu_mul_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(out_ptr + offsets, out, mask=mask)


def _grid(n_elements: int) -> tuple[int]:
    return (triton.cdiv(n_elements, _BLOCK_SIZE),)


def triton_sigmoid_mul(x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
    if _can_use_triton(x, y):
        out = torch.empty_like(y)
        _sigmoid_mul_kernel[_grid(y.numel())](
            x, y, out, y.numel(), block_size=_BLOCK_SIZE, num_warps=4
        )
        return out
    if not _can_use_triton_strided_y(x, y):
        return None
    out = torch.empty_like(y, memory_format=torch.contiguous_format)
    _sigmoid_mul_4d_strided_y_kernel[_grid(y.numel())](
        x,
        y,
        out,
        y.shape[0],
        y.shape[1],
        y.shape[2],
        y.shape[3],
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        y.numel(),
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return out


def triton_sigmoid_mul_add(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not _can_use_triton(x, y, z):
        return None
    out = torch.empty_like(y)
    _sigmoid_mul_add_kernel[_grid(y.numel())](
        x, y, z, out, y.numel(), block_size=_BLOCK_SIZE, num_warps=4
    )
    return out


def triton_silu_mul(x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
    if not _can_use_triton(x, y):
        return None
    out = torch.empty_like(y)
    _silu_mul_kernel[_grid(y.numel())](
        x, y, out, y.numel(), block_size=_BLOCK_SIZE, num_warps=4
    )
    return out


def fused_sigmoid_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = triton_sigmoid_mul(x, y)
    if out is not None:
        return out
    return torch.sigmoid(x) * y


def fused_sigmoid_mul_add(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    out = triton_sigmoid_mul_add(x, y, z)
    if out is not None:
        return out
    return torch.sigmoid(x) * y + z


def fused_silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = triton_silu_mul(x, y)
    if out is not None:
        return out
    return F.silu(x) * y
