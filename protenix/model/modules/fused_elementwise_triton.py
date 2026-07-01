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


def _can_use_first_dim_broadcast_triton(
    x: torch.Tensor,
    y: torch.Tensor,
) -> bool:
    if not triton_fused_elementwise_enabled():
        return False
    if not triton_fused_elementwise_available():
        return False
    if torch.is_grad_enabled():
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_cuda or not y.is_cuda:
        return False
    if x.dtype != y.dtype:
        return False
    if not x.is_contiguous() or not y.is_contiguous():
        return False
    if x.dim() != 3 or y.dim() != 3:
        return False
    if x.shape[0] != 1 or y.shape[0] <= 1:
        return False
    return x.shape[1:] == y.shape[1:] and x.numel() > 0


def _broadcast_z_mode(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> Optional[bool]:
    if not _can_use_first_dim_broadcast_triton(x, y):
        return None
    if not z.is_cuda or z.dtype != y.dtype or not z.is_contiguous():
        return None
    if z.shape == x.shape:
        return True
    if z.shape == y.shape:
        return False
    return None


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
    def _sigmoid_mul_broadcast_x_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        n_x_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x_offsets = offsets % n_x_elements
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _sigmoid_mul_add_broadcast_x_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        n_x_elements: tl.constexpr,
        z_broadcast: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x_offsets = offsets % n_x_elements
        if z_broadcast:
            z_offsets = x_offsets
        else:
            z_offsets = offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + z_offsets, mask=mask, other=0.0).to(tl.float32)
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
    if _can_use_first_dim_broadcast_triton(x, y):
        out = torch.empty_like(y)
        _sigmoid_mul_broadcast_x_kernel[_grid(y.numel())](
            x,
            y,
            out,
            y.numel(),
            x.numel(),
            block_size=_BLOCK_SIZE,
            num_warps=4,
        )
        return out
    return None


def triton_sigmoid_mul_add(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> Optional[torch.Tensor]:
    if _can_use_triton(x, y, z):
        out = torch.empty_like(y)
        _sigmoid_mul_add_kernel[_grid(y.numel())](
            x, y, z, out, y.numel(), block_size=_BLOCK_SIZE, num_warps=4
        )
        return out
    z_broadcast = _broadcast_z_mode(x, y, z)
    if z_broadcast is not None:
        out = torch.empty_like(y)
        _sigmoid_mul_add_broadcast_x_kernel[_grid(y.numel())](
            x,
            y,
            z,
            out,
            y.numel(),
            x.numel(),
            z_broadcast=z_broadcast,
            block_size=_BLOCK_SIZE,
            num_warps=4,
        )
        return out
    return None


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
