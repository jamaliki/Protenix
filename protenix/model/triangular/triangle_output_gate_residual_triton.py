"""Opt-in residual fusion for CUEQ triangle multiplication.

CUEQ's triangle update is already the right mainloop for the pairformer.  The
remaining waste in inference is narrower: CUEQ returns an update tensor and
Protenix immediately launches a full-tensor residual add.  This module keeps
CUEQ's normalization, input projection, and triangular product, but replaces the
final output gate with a Triton kernel that adds the residual in its final
store.  It is guarded behind an environment flag until full model gates prove
that the isolated win survives the whole pairformer.
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
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 64


def triton_triangle_output_gate_residual_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_output_gate_residual_available() -> bool:
    return (
        triton is not None
        and tl is not None
        and layer_norm_transpose is not None
        and fused_sigmoid_gated_dual_gemm is not None
    )


if triton_triangle_output_gate_residual_available():

    @triton.jit
    def _gate_value_residual_kernel(
        x_gate_ptr,
        x_value_ptr,
        w_gate_ptr,
        w_value_ptr,
        residual_ptr,
        out_ptr,
        m_rows,
        n_out: tl.constexpr,
        k_in: tl.constexpr,
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
        for k0 in range(0, k_in, block_k):
            k = k0 + offs_k
            xg = tl.load(
                x_gate_ptr + offs_m[:, None] * k_in + k[None, :],
                mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_in),
                other=0.0,
            )
            xv = tl.load(
                x_value_ptr + offs_m[:, None] * k_in + k[None, :],
                mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_in),
                other=0.0,
            )
            wg = tl.load(
                w_gate_ptr + offs_n[:, None] * k_in + k[None, :],
                mask=(offs_n[:, None] < n_out) & (k[None, :] < k_in),
                other=0.0,
            )
            wv = tl.load(
                w_value_ptr + offs_n[:, None] * k_in + k[None, :],
                mask=(offs_n[:, None] < n_out) & (k[None, :] < k_in),
                other=0.0,
            )
            gate += tl.dot(xg, tl.trans(wg))
            value += tl.dot(xv, tl.trans(wv))

        out = (1.0 / (1.0 + tl.exp(-gate))) * value
        residual = tl.load(
            residual_ptr + offs_m[:, None] * n_out + offs_n[None, :],
            mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_out),
            other=0.0,
        )
        out += residual
        tl.store(
            out_ptr + offs_m[:, None] * n_out + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_out),
        )


def _gate_value_residual(
    x_gate: torch.Tensor,
    x_value: torch.Tensor,
    w_gate: torch.Tensor,
    w_value: torch.Tensor,
    residual: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not triton_triangle_output_gate_residual_available():
        return None
    if x_gate.dtype not in _SUPPORTED_DTYPES:
        return None
    if not (x_gate.is_cuda and x_value.is_cuda and residual.is_cuda):
        return None
    if x_gate.shape != x_value.shape or x_gate.shape != residual.shape:
        return None
    if w_gate.shape != w_value.shape:
        return None
    if w_gate.dtype != x_gate.dtype or w_value.dtype != x_gate.dtype:
        return None
    if not (
        x_gate.is_contiguous()
        and x_value.is_contiguous()
        and residual.is_contiguous()
    ):
        return None

    original_shape = x_gate.shape
    k_in = original_shape[-1]
    n_out, weight_k = w_gate.shape
    if weight_k != k_in or n_out != residual.shape[-1]:
        return None

    x_gate_2d = x_gate.reshape(-1, k_in)
    x_value_2d = x_value.reshape(-1, k_in)
    residual_2d = residual.reshape(-1, n_out)
    out = torch.empty_like(residual_2d)
    grid = (triton.cdiv(x_gate_2d.shape[0], _BLOCK_M), triton.cdiv(n_out, _BLOCK_N))
    _gate_value_residual_kernel[grid](
        x_gate_2d,
        x_value_2d,
        w_gate,
        w_value,
        residual_2d,
        out,
        x_gate_2d.shape[0],
        n_out,
        k_in,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out.reshape(*original_shape[:-1], n_out)


def triangle_update_with_residual(
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
    if not triton_triangle_output_gate_residual_enabled():
        return None
    if torch.is_grad_enabled() or z.dim() != 4 or z.dtype not in _SUPPORTED_DTYPES:
        return None
    if not z.is_cuda or not z.is_contiguous():
        return None

    if mask is None:
        mask = z.new_ones(z.shape[:-1])

    x_norm = layer_norm_transpose(
        z,
        norm_in_weight,
        norm_in_bias,
        eps=eps,
        layout="bijd->bijd",
    )
    x_in = x_norm
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

    update = layer_norm_transpose(
        update,
        norm_out_weight,
        norm_out_bias,
        eps=eps,
        layout="dbij->bijd",
    )
    return _gate_value_residual(x_in, update, g_out_weight, p_out_weight, z)
