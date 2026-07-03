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
_BLOCK_SIZE = 1024
_MAX_TRITON_VECTOR_ELEMENTS = 1_000_000_000
_FALSE_ENV_VALUES = {"0", "false", "off", "no"}


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in _FALSE_ENV_VALUES


def triton_fused_elementwise_enabled() -> bool:
    return _env_flag_enabled("PROTENIX_TRITON_FUSED_ELEMENTWISE", default=False)


def _min_elementwise_elements() -> int:
    raw = os.getenv("PROTENIX_TRITON_FUSED_ELEMENTWISE_MIN_ELEMENTS")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _large_enough_for_elementwise_triton(n_elements: int) -> bool:
    """Keep opt-in Triton fusion away from tiny tensors.

    The global elementwise flag is intentionally default-off because a previous
    representative gate showed that many small Triton launches can overwhelm
    the memory-traffic savings.  The current profile still has large BF16
    add/mul/silu kernels near the top, so experiments can set a minimum element
    count and fuse only the HBM-bound cases worth a Triton launch.
    """
    return n_elements > 0 and n_elements >= _min_elementwise_elements()


def triton_triangle_attention_epilogue_enabled() -> bool:
    """Whether to fuse cuequivariance triangle-attention's output epilogue.

    This epilogue is the fixed memory-bound pattern after CUEQ returns heads in
    ``[..., H, Q, C]`` layout: sigmoid gate, multiply, transpose/flatten, then
    output projection.  The Triton path is inference-only and has strict shape
    guards, so keep it default-on for throughput while retaining two opt-outs:

    - ``PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE=0`` disables only this path.
    - ``PROTENIX_TRITON_FUSED_ELEMENTWISE=0`` disables all elementwise fusions
      unless the triangle-attention-specific variable is set explicitly.
    """
    specific = os.getenv("PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE")
    if specific is not None:
        return specific.lower() not in _FALSE_ENV_VALUES
    global_flag = os.getenv("PROTENIX_TRITON_FUSED_ELEMENTWISE")
    if global_flag is not None:
        return global_flag.lower() not in _FALSE_ENV_VALUES
    return True


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
    return _large_enough_for_elementwise_triton(tensors[0].numel())


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
    return x.shape[1:] == y.shape[1:] and _large_enough_for_elementwise_triton(
        y.numel()
    )


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
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
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
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
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
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
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
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
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
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _silu_mul_inplace_y_kernel(
        x_ptr,
        y_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(y_ptr + offsets, out, mask=mask)

    @triton.jit
    def _silu_mul_split_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        split_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
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
    def _sigmoid_mul_heads_first_kernel(
        gate_ptr,
        heads_first_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        q_count: tl.constexpr,
        head_dim: tl.constexpr,
        heads_channels: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
        valid = offsets < n_elements

        hc = offsets % heads_channels
        q = (offsets // heads_channels) % q_count
        batch = offsets // (q_count * heads_channels)
        h = hc // head_dim
        c = hc % head_dim

        # ``out`` and ``gate`` are contiguous in [..., Q, H * C] order.  The
        # cuequivariance attention output is contiguous in [..., H, Q, C] order.
        # This index swap performs the transpose+flatten while applying the
        # gate, avoiding a separate sigmoid tensor, gated tensor, and reshape
        # copy of the large pair representation.
        source_offsets = (
            batch * heads_channels * q_count
            + h * q_count * head_dim
            + q * head_dim
            + c
        )
        gate = tl.load(gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        value = tl.load(heads_first_ptr + source_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        out = (1.0 / (1.0 + tl.exp(-gate))) * value
        tl.store(out_ptr + offsets, out, mask=valid)

    @triton.jit
    def _sigmoid_mul_heads_first_swapped_gate_kernel(
        gate_ptr,
        heads_first_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        i_count: tl.constexpr,
        j_count: tl.constexpr,
        head_dim: tl.constexpr,
        heads_channels: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * block_size + tl.arange(0, block_size)).to(tl.int64)
        valid = offsets < n_elements

        hc = offsets % heads_channels
        i = (offsets // heads_channels) % i_count
        tmp = offsets // (heads_channels * i_count)
        j = tmp % j_count
        batch = tmp // j_count
        h = hc // head_dim
        c = hc % head_dim

        # ``out`` is contiguous in [..., J, I, H * C] order for the ending-node
        # output projection.  ``gate`` was projected from the original
        # contiguous [..., I, J, H * C] pair tensor, while CUEQ returned
        # heads-first [..., J, H, I, C].  Read both with swapped pair axes so the
        # prototype can test a broad ending-attention producer without
        # materializing gate.transpose(I, J).
        gate_offsets = ((batch * i_count + i) * j_count + j) * heads_channels + hc
        source_offsets = (
            ((batch * j_count + j) * (heads_channels // head_dim) + h)
            * i_count
            + i
        ) * head_dim + c

        gate = tl.load(gate_ptr + gate_offsets, mask=valid, other=0.0).to(tl.float32)
        value = tl.load(heads_first_ptr + source_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        out = (1.0 / (1.0 + tl.exp(-gate))) * value
        tl.store(out_ptr + offsets, out, mask=valid)


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


def triton_silu_mul_inplace_y(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Overwrite ``y`` with ``silu(x) * y`` when the inference fast path is safe.

    Pair-transition's large-batch fallback already holds both input projections
    ``x`` and ``y``.  Allocating a third activation just to fuse SiLU and multiply
    can erase the memory benefit, so this helper keeps the original in-place
    memory model while reducing two memory-bound ATen launches to one Triton
    launch.
    """
    if not _can_use_triton(x, y):
        return None
    total = y.numel()
    if total <= _MAX_TRITON_VECTOR_ELEMENTS:
        _silu_mul_inplace_y_kernel[_grid(total)](
            x, y, total, block_size=_BLOCK_SIZE, num_warps=4
        )
    else:
        # Very large pair-transition batches exceed the safe range for one
        # Triton vector launch.  Use narrowed flat views so each kernel sees a
        # small pointer base and local offsets, while still replacing PyTorch's
        # separate SiLU and multiply chunk launches with one fused launch/chunk.
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        for start in range(0, total, _MAX_TRITON_VECTOR_ELEMENTS):
            n_elements = min(_MAX_TRITON_VECTOR_ELEMENTS, total - start)
            _silu_mul_inplace_y_kernel[_grid(n_elements)](
                x_flat.narrow(0, start, n_elements),
                y_flat.narrow(0, start, n_elements),
                n_elements,
                block_size=_BLOCK_SIZE,
                num_warps=4,
            )
    return y


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
    if x.shape[-1] != 2 * split_size:
        return None
    output_elements = x.numel() // 2
    if not _large_enough_for_elementwise_triton(output_elements):
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


def triton_sigmoid_mul_heads_first(
    gate: torch.Tensor,
    heads_first: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Gate and flatten cuequivariance attention output in one pass.

    ``gate`` is the linear gate projection in ``[..., Q, H * C]`` layout.
    ``heads_first`` is the cuequivariance attention output in ``[..., H, Q, C]``
    layout.  The ordinary PyTorch path computes
    ``heads_first.transpose(-2, -3) * sigmoid(gate.view(...))`` and then
    reshapes to ``[..., Q, H * C]`` for the output projection.  At high
    sequence batch this reshape is a real HBM copy, not metadata.  This kernel
    writes the contiguous projected layout directly.
    """
    if not triton_triangle_attention_epilogue_enabled():
        return None
    if not triton_fused_elementwise_available():
        return None
    if torch.is_grad_enabled():
        return None
    if gate.dtype not in _SUPPORTED_DTYPES:
        return None
    if not gate.is_cuda or not heads_first.is_cuda:
        return None
    if gate.dtype != heads_first.dtype:
        return None
    if not gate.is_contiguous() or not heads_first.is_contiguous():
        return None
    if gate.dim() < 2 or heads_first.dim() != gate.dim() + 1:
        return None
    q_count = gate.shape[-2]
    heads_channels = gate.shape[-1]
    head_count = heads_first.shape[-3]
    if heads_first.shape[:-3] != gate.shape[:-2]:
        return None
    if heads_first.shape[-2] != q_count:
        return None
    if heads_channels % head_count != 0:
        return None
    head_dim = heads_channels // head_count
    if heads_first.shape[-1] != head_dim:
        return None
    if gate.numel() == 0:
        return None

    out = torch.empty_like(gate)
    _sigmoid_mul_heads_first_kernel[_grid(gate.numel())](
        gate,
        heads_first,
        out,
        gate.numel(),
        q_count=q_count,
        head_dim=head_dim,
        heads_channels=heads_channels,
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return out


def triton_sigmoid_mul_heads_first_swapped_gate(
    gate: torch.Tensor,
    heads_first: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Gate CUEQ ending attention without materializing ``gate.transpose``.

    This is a benchmark/prototype helper for ending-node triangle attention.
    ``gate`` is projected from the original contiguous ``[..., I, J, H * C]``
    pair tensor, while ``heads_first`` is CUEQ output for the logical
    transposed problem in ``[..., J, H, I, C]`` layout.  The output is
    contiguous ``[..., J, I, H * C]`` for the existing output GEMM.
    """
    if not triton_triangle_attention_epilogue_enabled():
        return None
    if not triton_fused_elementwise_available():
        return None
    if torch.is_grad_enabled():
        return None
    if gate.dtype not in _SUPPORTED_DTYPES:
        return None
    if not gate.is_cuda or not heads_first.is_cuda:
        return None
    if gate.dtype != heads_first.dtype:
        return None
    if not gate.is_contiguous() or not heads_first.is_contiguous():
        return None
    if gate.dim() < 3 or heads_first.dim() != gate.dim() + 1:
        return None
    if heads_first.shape[:-4] != gate.shape[:-3]:
        return None
    i_count = gate.shape[-3]
    j_count = gate.shape[-2]
    heads_channels = gate.shape[-1]
    head_count = heads_first.shape[-3]
    if heads_first.shape[-4] != j_count or heads_first.shape[-2] != i_count:
        return None
    if heads_channels % head_count != 0:
        return None
    head_dim = heads_channels // head_count
    if heads_first.shape[-1] != head_dim:
        return None
    if gate.numel() == 0:
        return None

    out = torch.empty(
        (*gate.shape[:-3], j_count, i_count, heads_channels),
        dtype=gate.dtype,
        device=gate.device,
    )
    _sigmoid_mul_heads_first_swapped_gate_kernel[_grid(out.numel())](
        gate,
        heads_first,
        out,
        out.numel(),
        i_count=i_count,
        j_count=j_count,
        head_dim=head_dim,
        heads_channels=heads_channels,
        block_size=_BLOCK_SIZE,
        num_warps=4,
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


def fused_sigmoid_mul_heads_first(
    gate: torch.Tensor,
    heads_first: torch.Tensor,
) -> torch.Tensor:
    out = triton_sigmoid_mul_heads_first(gate, heads_first)
    if out is not None:
        return out
    head_count = heads_first.shape[-3]
    return torch.sigmoid(gate).view(gate.shape[:-1] + (head_count, -1)) * (
        heads_first.transpose(-2, -3)
    )
