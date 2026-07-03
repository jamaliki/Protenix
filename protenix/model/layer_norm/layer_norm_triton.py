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

"""Triton LayerNorm fallback for inference.

The built-in CUDA LayerNorm extension is fastest when it builds, but current
Tokyo profiling runs use PyTorch's generic ``native_layer_norm`` fallback
because CUDA 13 cannot compile that extension against the installed PyTorch
headers.  The profiler then shows tens of thousands of memory-bound LayerNorm
launches near the top of the trace.

This file is deliberately narrow: it implements only forward inference for a
contiguous tensor normalized over its last dimension.  Training, CPU, missing
Triton, non-contiguous inputs, and large/unprofiled feature dimensions all fall
back to PyTorch.  Auto mode uses Triton broadly for BF16/FP16 and only for the
large-row FP32 diffusion-token shapes that moved the representative H100 gate.
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
_AUTO_ENV_VALUES = {"", "auto", "default"}
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_DEFAULT_DTYPES = {torch.float16, torch.bfloat16}
_DEFAULT_FP32_COLS = {384, 768}
_DEFAULT_FP32_MIN_ROWS = 8192
_DEFAULT_MAX_COLS = 2048


def _env_policy_value() -> str | None:
    value = os.getenv("PROTENIX_TRITON_LAYER_NORM")
    return None if value is None else value.strip().lower()


def _auto_fp32_cols() -> set[int]:
    raw = os.getenv("PROTENIX_TRITON_LAYER_NORM_FP32_COLS")
    if raw is None:
        return _DEFAULT_FP32_COLS
    cols = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            cols.add(int(item))
        except ValueError:
            return _DEFAULT_FP32_COLS
    return cols


def _auto_fp32_min_rows() -> int:
    raw = os.getenv("PROTENIX_TRITON_LAYER_NORM_FP32_MIN_ROWS")
    if raw is None:
        return _DEFAULT_FP32_MIN_ROWS
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_FP32_MIN_ROWS


def triton_layer_norm_enabled(dtype: torch.dtype) -> bool:
    """Return the dtype-only LayerNorm policy.

    ``PROTENIX_TRITON_LAYER_NORM`` is intentionally a three-state switch:

    - unset/``auto``: use Triton for FP16/BF16 inference.  FP32 auto-selection
      also needs shape information, so this dtype-only helper returns ``False``
      for FP32 and ``triton_layer_norm_enabled_for_shape`` makes the final
      runtime decision;
    - false values such as ``0``/``off``: disable the candidate completely;
    - any other value, including ``1``: force it for every supported dtype.
    """

    value = _env_policy_value()
    if value is None or value.lower() in _AUTO_ENV_VALUES:
        return dtype in _DEFAULT_DTYPES
    return value.lower() not in _FALSE_ENV_VALUES


def triton_layer_norm_enabled_for_shape(
    dtype: torch.dtype,
    n_rows: int,
    n_cols: int,
) -> bool:
    """Return the runtime policy after seeing the flattened LayerNorm shape.

    The first FP32 screens were mixed: small or narrow FP32 shapes could lose to
    PyTorch.  The end-to-end gate only justified enabling Triton for the large
    diffusion-token shapes left after the BF16 fallback, e.g. ``[80, N, 384]``
    and ``[80, N, 768]``.  The row/width guard keeps that win without turning
    every FP32 LayerNorm in the model into a new default experiment.
    """

    value = _env_policy_value()
    if value is not None and value not in _AUTO_ENV_VALUES:
        return value not in _FALSE_ENV_VALUES
    if dtype in _DEFAULT_DTYPES:
        return True
    if dtype is not torch.float32:
        return False
    return n_cols in _auto_fp32_cols() and n_rows >= _auto_fp32_min_rows()


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

    if not triton_layer_norm_available():
        return None
    if torch.is_grad_enabled():
        return None
    if len(normalized_shape) != 1:
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
    if input.dtype not in _SUPPORTED_DTYPES:
        return None

    n_rows = input.numel() // n_cols
    if not triton_layer_norm_enabled_for_shape(input.dtype, n_rows, n_cols):
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
    output = torch.empty_like(input)
    num_warps = 4 if block_size <= 1024 else 8
    # Triton still expects pointer-typed arguments even when the corresponding
    # constexpr branch does not load them.  Reuse ``input`` as a harmless dummy
    # pointer for absent affine parameters.
    weight_ptr = input if weight is None else weight
    bias_ptr = input if bias is None else bias
    _layer_norm_kernel[(n_rows,)](
        input,
        weight_ptr,
        bias_ptr,
        output,
        n_cols,
        eps,
        weight is not None,
        bias is not None,
        block_size,
        num_warps=num_warps,
    )
    return output
