"""Opt-in fused input producer for CUEQ triangle multiplication.

This keeps the current CUEQ/PyTorch triangle-multiplication structure, but
replaces the first producer pair:

    layer_norm_transpose(z) + fused_sigmoid_gated_dual_gemm(...)

with a Triton producer that applies LayerNorm inside the dual-GEMM loop and
writes CUEQ's `[2D, B, N, N]` projection layout directly.  The output gate still
needs the normalized input, so the producer also materializes `[B, N, N, D]`
once.  The point is not to avoid all normalization traffic; it is to avoid
writing normalized input only to immediately reread it for the input GEMM.

The path is deliberately narrow and default-off until block/full-model gates
prove it should be promoted for Protenix-v2 inference.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from cuequivariance_ops_torch.gated_gemm_torch import (
        fused_sigmoid_gated_dual_gemm_dual_x,
    )
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None
    layer_norm_transpose = None
    fused_sigmoid_gated_dual_gemm_dual_x = None


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
# The kernel explicitly casts the normalized tile to BF16 before the tensor-core
# dot products, matching Sam's H100 inference target.  Keep the production path
# BF16-only rather than silently changing FP16 numerics.
_SUPPORTED_DTYPES = {torch.bfloat16}
_CHANNELS = 256
_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 64


def triton_triangle_ln_dual_gemm_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TRIANGLE_LN_DUAL_GEMM", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_ln_dual_gemm_available() -> bool:
    return (
        triton is not None
        and tl is not None
        and layer_norm_transpose is not None
        and fused_sigmoid_gated_dual_gemm_dual_x is not None
    )


if triton_triangle_ln_dual_gemm_available():

    @triton.jit
    def _row_stats_kernel(
        x,
        mean,
        rstd,
        rows: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        # Triton JIT kernels cannot reliably close over ordinary Python globals.
        # The production guard fixes `channels == 256`, so keep the static block
        # width literal here and still use the constexpr `channels` for masks.
        offs = tl.arange(0, 256)
        mask = offs < channels
        values = tl.load(x + row * channels + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        mu = tl.sum(values, axis=0) / channels
        centered = tl.where(mask, values - mu, 0.0)
        var = tl.sum(centered * centered, axis=0) / channels
        tl.store(mean + row, mu)
        tl.store(rstd + row, tl.rsqrt(var + eps))

    @triton.jit
    def _ln_dual_gemm_transpose_kernel(
        x,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        proj_weight,
        mask_ptr,
        ab_out,
        norm_out,
        rows,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
        offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
        offs_k = tl.arange(0, block_k).to(tl.int64)

        acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
        acc_proj = tl.zeros((block_m, block_n), dtype=tl.float32)
        row_mean = tl.load(mean + offs_m, mask=offs_m < rows, other=0.0)[:, None]
        row_rstd = tl.load(rstd + offs_m, mask=offs_m < rows, other=1.0)[:, None]
        for k0 in range(0, channels, block_k):
            k = k0 + offs_k
            k_mask = k < channels
            values = tl.load(
                x + offs_m[:, None] * channels + k[None, :],
                mask=(offs_m[:, None] < rows) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
            bias = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
            values = (values - row_mean) * row_rstd * weight[None, :] + bias[None, :]
            values = values.to(tl.bfloat16)
            # Only the first output-channel tile writes normalized input.  All
            # tiles need the same values for the dot products, so writing from
            # every tile would duplicate a full `[B,N,N,D]` HBM store.
            tl.store(
                norm_out + offs_m[:, None] * channels + k[None, :],
                values,
                mask=(pid_n == 0) & (offs_m[:, None] < rows) & k_mask[None, :],
            )
            gate_w = tl.load(
                gate_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < out_channels) & k_mask[None, :],
                other=0.0,
            )
            proj_w = tl.load(
                proj_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < out_channels) & k_mask[None, :],
                other=0.0,
            )
            acc_gate += tl.dot(values, tl.trans(gate_w))
            acc_proj += tl.dot(values, tl.trans(proj_w))

        pair_mask = tl.load(mask_ptr + offs_m, mask=offs_m < rows, other=0.0).to(
            tl.float32
        )
        result = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj * pair_mask[:, None]
        tl.store(
            ab_out + offs_n[None, :] * rows + offs_m[:, None],
            result,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_channels),
        )


def _input_producer(
    z: torch.Tensor,
    mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = z.numel() // z.shape[-1]
    out_channels = 2 * z.shape[-1]
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty(rows, device=z.device, dtype=torch.float32)
    ab = torch.empty((out_channels, rows), device=z.device, dtype=z.dtype)
    x_norm = torch.empty_like(z)
    _row_stats_kernel[(rows,)](z, mean, rstd, rows, _CHANNELS, eps)
    grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(out_channels, _BLOCK_N))
    _ln_dual_gemm_transpose_kernel[grid](
        z,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        proj_weight,
        mask,
        ab,
        x_norm,
        rows,
        _CHANNELS,
        out_channels,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return ab.reshape(out_channels, *z.shape[:3]), x_norm


def triangle_update_with_ln_dual_gemm(
    z: torch.Tensor,
    *,
    direction: str,
    mask: Optional[torch.Tensor],
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
) -> Optional[torch.Tensor]:
    if not triton_triangle_ln_dual_gemm_enabled():
        return None
    if not triton_triangle_ln_dual_gemm_available():
        return None
    if torch.is_grad_enabled() or z.dim() != 4 or z.dtype not in _SUPPORTED_DTYPES:
        return None
    if not z.is_cuda or not z.is_contiguous() or z.shape[-1] != _CHANNELS:
        return None
    if p_in_weight.shape != (2 * _CHANNELS, _CHANNELS):
        return None
    if g_in_weight.shape != p_in_weight.shape:
        return None
    if p_out_weight.shape != (_CHANNELS, _CHANNELS):
        return None
    if g_out_weight.shape != p_out_weight.shape:
        return None
    if not (
        p_in_weight.dtype == z.dtype
        and g_in_weight.dtype == z.dtype
        and p_out_weight.dtype == z.dtype
        and g_out_weight.dtype == z.dtype
    ):
        return None
    if not (
        p_in_weight.is_cuda
        and g_in_weight.is_cuda
        and p_out_weight.is_cuda
        and g_out_weight.is_cuda
    ):
        return None
    if not (
        p_in_weight.is_contiguous()
        and g_in_weight.is_contiguous()
        and p_out_weight.is_contiguous()
        and g_out_weight.is_contiguous()
    ):
        return None
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    if mask.shape != z.shape[:-1] or not mask.is_cuda:
        return None
    if not mask.is_contiguous():
        mask = mask.contiguous()

    ab, x_norm = _input_producer(
        z,
        mask,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        eps,
    )
    a, b = torch.chunk(ab, 2, dim=0)
    if direction == "outgoing":
        update = torch.einsum("dbik,dbjk->dbij", a, b)
    elif direction == "incoming":
        update = torch.einsum("dbki,dbkj->dbij", a, b)
    else:
        raise ValueError("direction must be either 'outgoing' or 'incoming'")

    update = layer_norm_transpose(
        update,
        norm_out_weight,
        norm_out_bias,
        eps=eps,
        layout="dbij->bijd",
    )
    out = fused_sigmoid_gated_dual_gemm_dual_x(
        x_norm,
        update,
        g_out_weight,
        p_out_weight,
    )
    return out + z
