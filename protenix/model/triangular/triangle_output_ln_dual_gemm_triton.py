"""Opt-in output LayerNorm/GEMM fusion for triangle multiplication.

This path is deliberately narrower than a full triangle-multiplication rewrite.
CUEQ remains responsible for the input normalization/projection and PyTorch
keeps the dense triangular contraction.  The fused part is the consumer side:

    output LayerNorm/layout conversion -> output gate/value GEMMs -> residual

CUEQ's `layer_norm_transpose` is bandwidth-oriented and fast, but it still
materializes a `[B,N,N,D]` tensor that the output value GEMM immediately reads.
For Protenix-v2 (`D=256`) that is a large HBM round trip inside the hottest
pairformer trunk.  The Triton kernel below computes the LayerNorm row stats,
applies normalization inside the value GEMM, computes the gate GEMM from the
already-normalized input, and adds the residual in the final store.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None
    layer_norm_transpose = None
    fused_sigmoid_gated_dual_gemm = None


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_SUPPORTED_DTYPES = {torch.bfloat16}
_CHANNELS = 256
_BLOCK_M = 32
_BLOCK_N = 128
_BLOCK_K = 64


def triton_triangle_output_ln_dual_gemm_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TRIANGLE_OUTPUT_LN_DUAL_GEMM", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_output_ln_dual_gemm_available() -> bool:
    return (
        triton is not None
        and tl is not None
        and layer_norm_transpose is not None
        and fused_sigmoid_gated_dual_gemm is not None
    )


if triton_triangle_output_ln_dual_gemm_available():

    @triton.jit
    def _dbij_row_stats_kernel(
        update,
        mean,
        rstd,
        rows: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, 256)
        mask = offs < channels
        # `update` is contiguous `[D,B,N,N]`, so one logical BIJ row is strided
        # by `rows` along D.  This is the layout conversion we are trying to
        # delete by consuming DBIJ directly.
        values = tl.load(update + offs * rows + row, mask=mask, other=0.0).to(
            tl.float32
        )
        mu = tl.sum(values, axis=0) / channels
        centered = tl.where(mask, values - mu, 0.0)
        var = tl.sum(centered * centered, axis=0) / channels
        tl.store(mean + row, mu)
        tl.store(rstd + row, tl.rsqrt(var + eps))

    @triton.jit
    def _output_ln_dual_gemm_residual_kernel(
        x_gate,
        update,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        value_weight,
        residual,
        out,
        rows,
        channels: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
        offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
        offs_k = tl.arange(0, block_k).to(tl.int64)

        gate = tl.zeros((block_m, block_n), dtype=tl.float32)
        value = tl.zeros((block_m, block_n), dtype=tl.float32)
        row_mean = tl.load(mean + offs_m, mask=offs_m < rows, other=0.0)[:, None]
        row_rstd = tl.load(rstd + offs_m, mask=offs_m < rows, other=1.0)[:, None]
        for k0 in range(0, channels, block_k):
            k = k0 + offs_k
            k_mask = k < channels
            xg = tl.load(
                x_gate + offs_m[:, None] * channels + k[None, :],
                mask=(offs_m[:, None] < rows) & k_mask[None, :],
                other=0.0,
            )
            upd = tl.load(
                update + k[None, :] * rows + offs_m[:, None],
                mask=(offs_m[:, None] < rows) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            w = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
            b = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
            upd = ((upd - row_mean) * row_rstd * w[None, :] + b[None, :]).to(
                tl.bfloat16
            )
            wg = tl.load(
                gate_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < channels) & k_mask[None, :],
                other=0.0,
            )
            wv = tl.load(
                value_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < channels) & k_mask[None, :],
                other=0.0,
            )
            gate += tl.dot(xg, tl.trans(wg))
            value += tl.dot(upd, tl.trans(wv))

        result = (1.0 / (1.0 + tl.exp(-gate))) * value
        res = tl.load(
            residual + offs_m[:, None] * channels + offs_n[None, :],
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < channels),
            other=0.0,
        )
        tl.store(
            out + offs_m[:, None] * channels + offs_n[None, :],
            result + res,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < channels),
        )


def _output_ln_dual_gemm_residual(
    x_gate: torch.Tensor,
    update: torch.Tensor,
    residual: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    value_weight: torch.Tensor,
    eps: float,
) -> Optional[torch.Tensor]:
    if not triton_triangle_output_ln_dual_gemm_available():
        return None
    if x_gate.dtype not in _SUPPORTED_DTYPES or x_gate.shape[-1] != _CHANNELS:
        return None
    if not (x_gate.is_cuda and update.is_cuda and residual.is_cuda):
        return None
    if x_gate.shape != residual.shape or update.shape[0] != _CHANNELS:
        return None
    if not (x_gate.is_contiguous() and update.is_contiguous() and residual.is_contiguous()):
        return None
    if gate_weight.shape != (_CHANNELS, _CHANNELS) or value_weight.shape != gate_weight.shape:
        return None
    if gate_weight.dtype != x_gate.dtype or value_weight.dtype != x_gate.dtype:
        return None

    rows = x_gate.numel() // _CHANNELS
    mean = torch.empty(rows, device=x_gate.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    out = torch.empty_like(x_gate.reshape(rows, _CHANNELS))
    _dbij_row_stats_kernel[(rows,)](update, mean, rstd, rows, _CHANNELS, eps)
    grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(_CHANNELS, _BLOCK_N))
    _output_ln_dual_gemm_residual_kernel[grid](
        x_gate,
        update,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        value_weight,
        residual,
        out,
        rows,
        _CHANNELS,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out.reshape_as(x_gate)


def triangle_update_with_output_ln_dual_gemm(
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
    if not triton_triangle_output_ln_dual_gemm_enabled():
        return None
    if torch.is_grad_enabled() or z.dim() != 4 or z.dtype not in _SUPPORTED_DTYPES:
        return None
    if not z.is_cuda or not z.is_contiguous() or z.shape[-1] != _CHANNELS:
        return None
    if p_in_weight.shape != (2 * _CHANNELS, _CHANNELS) or g_in_weight.shape != p_in_weight.shape:
        return None

    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    x_norm = layer_norm_transpose(z, norm_in_weight, norm_in_bias, eps=eps, layout="bijd->bijd")
    ab = fused_sigmoid_gated_dual_gemm(
        x_norm,
        g_in_weight,
        p_in_weight,
        mask,
        transpose_out=True,
    )
    a, b = torch.chunk(ab, 2, dim=0)
    if direction == "outgoing":
        update = torch.einsum("dbik,dbjk->dbij", a, b)
    elif direction == "incoming":
        update = torch.einsum("dbki,dbkj->dbij", a, b)
    else:
        raise ValueError("direction must be either 'outgoing' or 'incoming'")

    return _output_ln_dual_gemm_residual(
        x_norm,
        update,
        z,
        norm_weight=norm_out_weight,
        norm_bias=norm_out_bias,
        gate_weight=g_out_weight,
        value_weight=p_out_weight,
        eps=eps,
    )
