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

"""Small Triton kernels for memory-bound inference gates.

These helpers intentionally fuse only simple elementwise expressions such as
``sigmoid(x) * y`` and ``sigmoid(x) * y + z``.  Those expressions are cheap in
FLOPs but expensive in memory traffic when written as separate PyTorch ops:
each intermediate tensor is written to and reread from HBM.  A one-pass Triton
kernel is useful here because it removes intermediate memory traffic without
trying to replace cuBLAS/cuDNN kernels that are already highly optimized.

Every public helper falls back to the original PyTorch expression when the
shape, dtype, device, or autograd mode is outside the profiled inference path.
"""

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
_SUPPORTED_MATMUL_DTYPES = {torch.float16, torch.bfloat16}
_BLOCK_SIZE = 1024
_TRANSITION_INPUT_BLOCK_M = 128
_TRANSITION_INPUT_BLOCK_N = 128
_TRANSITION_INPUT_BLOCK_K = 64


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
    # The high-throughput diffusion path often has a gate tensor with sample
    # dimension 1 and an activation tensor with sample dimension N_sample.  Let
    # the kernel broadcast that first dimension instead of materializing the
    # gate N_sample times.
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

    @triton.jit
    def _silu_mul_split_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        split_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        cols = offsets % split_size
        rows = offsets // split_size
        row_base = rows * (2 * split_size)
        a = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(
            x_ptr + row_base + split_size + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = a * (1.0 / (1.0 + tl.exp(-a))) * b
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _transition_input_projection_silu_kernel(
        x_ptr,
        w1_t_ptr,
        w2_t_ptr,
        out_ptr,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * block_m + tl.arange(0, block_m)
        offsets_n = pid_n * block_n + tl.arange(0, block_n)
        offsets_k = tl.arange(0, block_k)

        acc1 = tl.zeros((block_m, block_n), tl.float32)
        acc2 = tl.zeros((block_m, block_n), tl.float32)
        for k0 in range(0, k_size, block_k):
            k = k0 + offsets_k
            x_tile = tl.load(
                x_ptr + offsets_m[:, None] * k_size + k[None, :],
                mask=(offsets_m[:, None] < m_size) & (k[None, :] < k_size),
                other=0.0,
            )
            # The weights are cached as [K, N].  That makes each program load
            # contiguous N columns instead of walking the model's native
            # LinearNoBias [N, K] storage with a large stride.
            w1_tile = tl.load(
                w1_t_ptr + k[:, None] * n_size + offsets_n[None, :],
                mask=(k[:, None] < k_size) & (offsets_n[None, :] < n_size),
                other=0.0,
            )
            w2_tile = tl.load(
                w2_t_ptr + k[:, None] * n_size + offsets_n[None, :],
                mask=(k[:, None] < k_size) & (offsets_n[None, :] < n_size),
                other=0.0,
            )
            acc1 += tl.dot(x_tile, w1_tile, out_dtype=tl.float32)
            acc2 += tl.dot(x_tile, w2_tile, out_dtype=tl.float32)

        out = acc1 * (1.0 / (1.0 + tl.exp(-acc1))) * acc2
        tl.store(
            out_ptr + offsets_m[:, None] * n_size + offsets_n[None, :],
            out,
            mask=(offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size),
        )


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


def triton_silu_mul_split(
    x: torch.Tensor,
    split_size: int,
) -> Optional[torch.Tensor]:
    if not triton_fused_elementwise_enabled():
        return None
    if not triton_fused_elementwise_available():
        return None
    if torch.is_grad_enabled():
        return None
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES or not x.is_contiguous():
        return None
    if x.shape[-1] != 2 * split_size or x.numel() == 0:
        return None
    out = torch.empty((*x.shape[:-1], split_size), dtype=x.dtype, device=x.device)
    _silu_mul_split_kernel[_grid(out.numel())](
        x,
        out,
        out.numel(),
        split_size,
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return out


def triton_transition_input_projection_silu(
    x: torch.Tensor,
    weight_a1_t: torch.Tensor,
    weight_a2_t: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not triton_fused_elementwise_enabled():
        return None
    if not triton_fused_elementwise_available():
        return None
    if torch.is_grad_enabled():
        return None
    if x.dtype not in _SUPPORTED_MATMUL_DTYPES:
        return None
    if not x.is_cuda or not weight_a1_t.is_cuda or not weight_a2_t.is_cuda:
        return None
    if weight_a1_t.dtype != x.dtype or weight_a2_t.dtype != x.dtype:
        return None
    if (
        not x.is_contiguous()
        or not weight_a1_t.is_contiguous()
        or not weight_a2_t.is_contiguous()
    ):
        return None
    if x.dim() < 2 or weight_a1_t.dim() != 2 or weight_a2_t.dim() != 2:
        return None

    k_size = x.shape[-1]
    if weight_a1_t.shape[0] != k_size or weight_a2_t.shape[0] != k_size:
        return None
    n_size = weight_a1_t.shape[1]
    if weight_a2_t.shape[1] != n_size or x.numel() == 0:
        return None
    # These multiples are the profiled H100 path.  The masks would make other
    # shapes correct, but the larger point of this kernel is throughput; fall
    # back to cuBLAS outside the shape family we have measured.
    if k_size % _TRANSITION_INPUT_BLOCK_K != 0:
        return None
    if n_size % _TRANSITION_INPUT_BLOCK_N != 0:
        return None

    m_size = x.numel() // k_size
    out = torch.empty((*x.shape[:-1], n_size), dtype=x.dtype, device=x.device)
    grid = (
        triton.cdiv(m_size, _TRANSITION_INPUT_BLOCK_M),
        triton.cdiv(n_size, _TRANSITION_INPUT_BLOCK_N),
    )
    _transition_input_projection_silu_kernel[grid](
        x,
        weight_a1_t,
        weight_a2_t,
        out,
        m_size,
        n_size,
        k_size,
        block_m=_TRANSITION_INPUT_BLOCK_M,
        block_n=_TRANSITION_INPUT_BLOCK_N,
        block_k=_TRANSITION_INPUT_BLOCK_K,
        num_warps=8,
        num_stages=3,
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


def fused_silu_mul_split(x: torch.Tensor, split_size: int) -> torch.Tensor:
    out = triton_silu_mul_split(x, split_size)
    if out is not None:
        return out
    a, b = x.split(split_size, dim=-1)
    return F.silu(a) * b
