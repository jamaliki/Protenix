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

"""Opt-in Triton LayerNorm for inference-only GPU profiling.

The built-in CUDA LayerNorm extension is fastest when it builds, but current
Tokyo profiling runs use PyTorch's generic ``native_layer_norm`` fallback
because CUDA 13 cannot compile that extension against the installed PyTorch
headers.  The profiler then shows tens of thousands of memory-bound LayerNorm
launches near the top of the trace.

This file is deliberately narrow: it implements only forward inference for a
contiguous tensor normalized over its last dimension.  Training, CPU, missing
Triton, non-contiguous inputs, and large/unprofiled feature dimensions all fall
back to PyTorch.  That keeps the model semantics safe while letting us test
whether a shape-specialized kernel removes enough launch and memory traffic to
matter end to end.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional for CPU installs.
    triton = None
    tl = None


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_DEFAULT_MAX_COLS = 2048


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in _FALSE_ENV_VALUES


def triton_layer_norm_enabled() -> bool:
    """Return whether the experimental Triton LayerNorm path is requested."""

    return _env_flag_enabled("PROTENIX_TRITON_LAYER_NORM", default=False)


def triton_layer_norm_available() -> bool:
    return triton is not None and tl is not None


def _max_cols() -> int:
    raw = os.getenv("PROTENIX_TRITON_LAYER_NORM_MAX_COLS")
    if raw is None:
        return _DEFAULT_MAX_COLS
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_MAX_COLS


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton_layer_norm_available():

    @triton.jit
    def _layer_norm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        has_weight: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols
        offsets = row * n_cols + cols

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / n_cols
        y = centered * tl.rsqrt(variance + eps)

        if has_weight:
            weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            y = y * weight
        if has_bias:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y + bias

        tl.store(output_ptr + offsets, y, mask=mask)


def triton_layer_norm(
    input: torch.Tensor,
    normalized_shape: torch.Size,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Optional[torch.Tensor]:
    """Return Triton LayerNorm output, or ``None`` when a fallback is safer.

    The feature dimension is a compile-time constant for Triton.  In Protenix
    inference most LayerNorms normalize channels such as 128, 384, or 768, so a
    single program can reduce one row in registers.  We require contiguous input
    instead of silently calling ``contiguous()`` because hidden copies would
    erase the exact traffic win this kernel is meant to measure.
    """

    if not triton_layer_norm_enabled():
        return None
    if not triton_layer_norm_available():
        return None
    if torch.is_grad_enabled():
        return None
    if len(normalized_shape) != 1:
        return None
    if input.dtype not in _SUPPORTED_DTYPES:
        return None
    if not input.is_cuda or not input.is_contiguous():
        return None
    if input.numel() == 0:
        return None

    n_cols = int(normalized_shape[0])
    if input.shape[-1] != n_cols:
        return None
    if n_cols <= 0 or n_cols > _max_cols():
        return None

    if weight is not None:
        if (
            not weight.is_cuda
            or weight.shape != normalized_shape
            or not weight.is_contiguous()
        ):
            return None
    if bias is not None:
        if not bias.is_cuda or bias.shape != normalized_shape or not bias.is_contiguous():
            return None

    block_size = _next_power_of_2(n_cols)
    n_rows = input.numel() // n_cols
    output = torch.empty_like(input)
    num_warps = 4 if block_size <= 1024 else 8
    _layer_norm_kernel[(n_rows,)](
        input,
        weight,
        bias,
        output,
        n_cols,
        eps,
        weight is not None,
        bias is not None,
        block_size,
        num_warps=num_warps,
    )
    return output
