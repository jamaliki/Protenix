# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Triton epilogue for cuequivariance triangle attention.

The fast path for triangle attention is already CUEQ/cuDNN.  The attractive
boundary is immediately after that kernel: CUEQ returns heads in
``[..., H, Q, D]`` order, while the model needs a sigmoid gate, flattening to
``[..., Q, H * D]``, the output projection, and the residual add.

This file deliberately fuses only that boundary.  It does not try to replace
the attention mainloop; it removes HBM traffic around an already-good kernel.
The helper is inference-only, BF16-only for now, and returns ``None`` whenever
the caller is outside the profiled path so the ordinary PyTorch/CUEQ path stays
the correctness fallback.
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


def triton_triangle_attention_output_projection_enabled() -> bool:
    """Return whether the larger triangle-attention epilogue is enabled.

    Keep this default-off until a full-model gate promotes it.  The older
    ``PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE`` flag controls only the
    elementwise gate/flatten fusion; this flag additionally fuses the output
    GEMM epilogue and residual write.
    """

    value = os.getenv("PROTENIX_TRITON_TRIANGLE_ATTENTION_OUTPUT_PROJECTION", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_attention_output_projection_available() -> bool:
    return triton is not None and tl is not None


if triton is not None and tl is not None:

    @triton.jit
    def _gate_linear_residual_heads_first_kernel(
        gate_ptr,
        heads_first_ptr,
        weight_ptr,
        residual_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        q_count: tl.constexpr,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)

        # Flatten all prefix dimensions.  For triangle attention this prefix is
        # usually ``batch * source_token``.  The last query dimension remains
        # explicit because CUEQ stores heads as [prefix, H, Q, D].
        q = rows % q_count
        prefix = rows // q_count

        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k0 in range(0, in_features, block_k):
            ks = k0 + tl.arange(0, block_k)
            head = ks // head_dim
            channel = ks % head_dim

            gate_offsets = rows[:, None] * in_features + ks[None, :]
            heads_offsets = (
                ((prefix[:, None] * n_heads + head[None, :]) * q_count + q[:, None])
                * head_dim
                + channel[None, :]
            )
            weight_offsets = cols[None, :] * in_features + ks[:, None]

            valid_a = (rows[:, None] < n_rows) & (ks[None, :] < in_features)
            gate = tl.load(gate_ptr + gate_offsets, mask=valid_a, other=0.0)
            heads = tl.load(heads_first_ptr + heads_offsets, mask=valid_a, other=0.0)

            # Match OpenfoldLinear's BF16 inference behavior: sigmoid is
            # evaluated in FP32, then the matmul operands are rounded to BF16 so
            # H100 tensor cores do the projection.
            activation = (tl.sigmoid(gate.to(tl.float32)) * heads).to(tl.bfloat16)

            valid_w = (cols[None, :] < out_features) & (ks[:, None] < in_features)
            weight = tl.load(weight_ptr + weight_offsets, mask=valid_w, other=0.0)
            weight = weight.to(tl.bfloat16)
            acc += tl.dot(activation, weight)

        out_offsets = rows[:, None] * out_features + cols[None, :]
        valid_out = (rows[:, None] < n_rows) & (cols[None, :] < out_features)
        residual = tl.load(residual_ptr + out_offsets, mask=valid_out, other=0.0)
        tl.store(out_ptr + out_offsets, acc + residual, mask=valid_out)


def triton_gate_linear_residual_heads_first(
    gate: torch.Tensor,
    heads_first: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    inplace: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    require_env: bool = True,
) -> Optional[torch.Tensor]:
    """Fuse CUEQ heads-first gating, output projection, and residual add.

    Args:
        gate: Linear gate projection in ``[..., Q, H * D]`` layout.
        heads_first: CUEQ attention output in ``[..., H, Q, D]`` layout.
        weight: ``linear_o.weight`` with shape ``[C_out, H * D]``.
        residual: Residual tensor in ``[..., Q, C_out]`` layout.
        inplace: If true, write the result into ``residual``.  This is safe for
            inference because the residual is read only once at the final store.

    Returns:
        The fused output, or ``None`` when guards reject the path.
    """

    if require_env and not triton_triangle_attention_output_projection_enabled():
        return None
    if not triton_triangle_attention_output_projection_available():
        return None
    if torch.is_grad_enabled():
        return None
    if gate.dtype != torch.bfloat16 or heads_first.dtype != gate.dtype:
        return None
    if not gate.is_cuda or not heads_first.is_cuda:
        return None
    if not weight.is_cuda or not residual.is_cuda:
        return None
    if not gate.is_contiguous() or not heads_first.is_contiguous():
        return None
    if not weight.is_contiguous() or not residual.is_contiguous():
        return None
    if gate.dim() < 2 or heads_first.dim() != gate.dim() + 1:
        return None
    if weight.dim() != 2:
        return None

    q_count = gate.shape[-2]
    in_features = gate.shape[-1]
    n_heads = heads_first.shape[-3]
    if heads_first.shape[:-3] != gate.shape[:-2]:
        return None
    if heads_first.shape[-2] != q_count:
        return None
    if n_heads <= 0 or in_features % n_heads != 0:
        return None
    head_dim = in_features // n_heads
    if heads_first.shape[-1] != head_dim:
        return None
    if weight.shape[1] != in_features:
        return None
    out_features = weight.shape[0]
    if residual.shape[:-1] != gate.shape[:-1] or residual.shape[-1] != out_features:
        return None
    if gate.numel() == 0:
        return None

    n_rows = gate.numel() // in_features
    out = residual if inplace else torch.empty_like(residual)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(out_features, block_n))
    _gate_linear_residual_heads_first_kernel[grid](
        gate,
        heads_first,
        weight,
        residual,
        out,
        n_rows,
        q_count,
        n_heads,
        head_dim,
        in_features,
        out_features,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
    )
    return out
