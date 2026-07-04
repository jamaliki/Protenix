# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Fused LayerNorm + q/k/v producer for base-width CUEQ triangle attention.

Triangle attention reuses the same normalized pair tensor three times: q/k/v,
the small triangle-bias projection, and the output gate.  A usable fusion must
therefore still write the normalized activation.  This helper replaces:

    LayerNorm(x) -> qkv GEMM -> qkv CUEQ-layout copy

with one BF16 Triton producer that writes both ``x_norm`` and CUEQ's
heads-first q/k/v layout.  It deliberately keeps the CUEQ/cuDNN attention
mainloop and the output epilogue unchanged.

The wider Protenix-v2 shape loses to cuBLAS in this simple Triton mainloop, so
the production guard is base-width only until a vendor-class CuTe/CUTLASS
producer exists.
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
_SUPPORTED_DTYPES = {torch.bfloat16}
_BLOCK_M = 32
_BLOCK_N = 128
_DEFAULT_MIN_ROWS = 100_000


def triton_ln_qkv_cueq_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_LN_QKV_CUEQ", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def _min_rows() -> int:
    raw = os.getenv("PROTENIX_TRITON_LN_QKV_CUEQ_MIN_ROWS")
    if raw is None:
        return _DEFAULT_MIN_ROWS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MIN_ROWS


def triton_ln_qkv_cueq_available() -> bool:
    return triton is not None and tl is not None


if triton_ln_qkv_cueq_available():

    @triton.jit
    def _ln_qkv_to_cueq_kernel(
        x_ptr,
        gamma_ptr,
        beta_ptr,
        weight_ptr,
        norm_out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        total_rows: tl.constexpr,
        i_count: tl.constexpr,
        j_count: tl.constexpr,
        c_z: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        swapped: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(tl.int64)
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
        component = tl.program_id(2)

        k_offsets = tl.arange(0, block_k)
        row_mask = rows < total_rows
        k_mask = k_offsets < c_z

        x = tl.load(
            x_ptr + rows[:, None] * c_z + k_offsets[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(x, axis=1) / c_z
        centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / c_z
        inv_std = tl.rsqrt(var + eps)

        gamma = tl.load(gamma_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        norm = centered * inv_std[:, None]
        norm = tl.where(k_mask[None, :], norm * gamma[None, :] + beta[None, :], 0.0)

        # Only one of the three q/k/v component programs writes x_norm.  The
        # other two reuse the same row statistics for their projection tile but
        # do not race on the normalized output.
        tl.store(
            norm_out_ptr + rows[:, None] * c_z + k_offsets[None, :],
            norm,
            mask=(component == 0) & row_mask[:, None] & k_mask[None, :],
        )

        hidden = head_count * head_dim
        weight_rows = component * hidden + cols
        col_mask = cols < hidden
        weight = tl.load(
            weight_ptr + weight_rows[:, None] * c_z + k_offsets[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        projected = tl.dot(norm.to(tl.bfloat16), tl.trans(weight))

        row = rows[:, None]
        j = row % j_count
        tmp = row // j_count
        i = tmp % i_count
        batch = tmp // i_count

        hidden_col = cols[None, :]
        head = hidden_col // head_dim
        dim = hidden_col - head * head_dim
        if swapped:
            out_offsets = (
                (((batch * j_count + j) * head_count + head) * i_count + i)
                * head_dim
                + dim
            )
        else:
            out_offsets = (
                (((batch * i_count + i) * head_count + head) * j_count + j)
                * head_dim
                + dim
            )

        mask = row_mask[:, None] & col_mask[None, :]
        tl.store(q_ptr + out_offsets, projected, mask=mask & (component == 0))
        tl.store(k_ptr + out_offsets, projected, mask=mask & (component == 1))
        tl.store(v_ptr + out_offsets, projected, mask=mask & (component == 2))


def _can_use_ln_qkv(
    x: torch.Tensor,
    layer_norm,
    mha,
) -> bool:
    if not triton_ln_qkv_cueq_enabled() or not triton_ln_qkv_cueq_available():
        return False
    if torch.is_grad_enabled():
        return False
    if x.dtype not in _SUPPORTED_DTYPES or not x.is_cuda or not x.is_contiguous():
        return False
    if x.dim() != 4 or x.numel() == 0:
        return False
    batch, i_count, j_count, c_z = x.shape
    if batch * i_count * j_count < _min_rows():
        return False
    # Current H100 screens: base width wins; Protenix-v2 width loses to cuBLAS.
    if c_z != 128 or mha.no_heads != 4 or mha.c_hidden != 32:
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    if layer_norm.weight.shape != (c_z,) or layer_norm.bias.shape != (c_z,):
        return False
    return True


def triton_ln_qkv_to_cueq_heads_first(
    x: torch.Tensor,
    layer_norm,
    mha,
    *,
    swap_pair_axes: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return ``x_norm, q, k, v`` for CUEQ triangle attention, or ``None``."""

    if not _can_use_ln_qkv(x, layer_norm, mha):
        return None

    batch, i_count, j_count, c_z = x.shape
    head_count = mha.no_heads
    head_dim = mha.c_hidden
    if swap_pair_axes:
        out_shape = (batch, j_count, head_count, i_count, head_dim)
    else:
        out_shape = (batch, i_count, head_count, j_count, head_dim)

    x_norm = torch.empty_like(x)
    q = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    weight = mha._qkv_weight_for_dtype(x.dtype, x.device)

    total_rows = batch * i_count * j_count
    hidden = head_count * head_dim
    block_k = 1 << (c_z - 1).bit_length()
    grid = (
        triton.cdiv(total_rows, _BLOCK_M),
        triton.cdiv(hidden, _BLOCK_N),
        3,
    )
    _ln_qkv_to_cueq_kernel[grid](
        x,
        layer_norm.weight,
        layer_norm.bias,
        weight,
        x_norm,
        q,
        k,
        v,
        total_rows,
        i_count,
        j_count,
        c_z,
        head_count,
        head_dim,
        layer_norm.eps,
        swap_pair_axes,
        _BLOCK_M,
        _BLOCK_N,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    return x_norm, q, k, v
