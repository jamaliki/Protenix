#!/usr/bin/env python3
"""Upper-bound screen for a producer-owned exact-group triangle update.

Recent compact-update screens showed that exact-length contraction can be fast,
but every PyTorch wrapper around the existing compact ``ab`` ABI pays copy debt:
``torch.stack``/``cat`` in one version, hidden ``copy_`` materialization in the
strided-view version.  This benchmark asks the next, more native question:

    if the input LayerNorm + dual-GEMM producer already wrote exact-length
    record groups in the contractor layout, is the remaining contraction plus
    output/residual boundary worth turning into a real kernel?

The grouped inputs are pre-materialized before timing.  Therefore this is not a
deployable speedup claim; it is an acceptance test for investing in a producer-
owned CUDA/CuTe boundary.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from scripts.perf.triangle_update_compact_segmented_probe import (
    C_Z,
    DTYPE,
    LONG_LENGTHS,
    SHORT_LENGTHS,
    ContractConfig,
    baseline_update,
    cached_weights,
    compact_finish_from_update,
    compare,
    cuda_time,
    make_inputs,
    make_metadata,
    make_modules,
    parse_ints,
    triton_producer_intermediates,
    valid_from_dense,
)


@dataclass(frozen=True)
class OwnedGroup:
    length: int
    start: int
    count: int
    lhs: torch.Tensor
    rhs: torch.Tensor


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def make_owned_groups(ab: torch.Tensor, lengths: list[int]) -> list[OwnedGroup]:
    """Materialize exact-length contractor inputs outside the timed path."""

    groups: list[OwnedGroup] = []
    offset = 0
    spans: list[tuple[int, int]] = []
    for length in lengths:
        spans.append((offset, length))
        offset += length * length

    for length in sorted(set(lengths)):
        rows = length * length
        offsets = [start for start, item in spans if item == length]
        lhs_records = [
            ab[:C_Z, start : start + rows].reshape(C_Z, length, length)
            for start in offsets
        ]
        rhs_records = [
            ab[C_Z:, start : start + rows].reshape(C_Z, length, length)
            for start in offsets
        ]
        groups.append(
            OwnedGroup(
                length=length,
                start=offsets[0],
                count=len(offsets),
                lhs=torch.stack(lhs_records, dim=0).reshape(-1, length, length).contiguous(),
                rhs=torch.stack(rhs_records, dim=0).reshape(-1, length, length).contiguous(),
            )
        )
    return groups


def contract_groups(groups: list[OwnedGroup], direction: str) -> list[torch.Tensor]:
    if direction == "outgoing":
        return [torch.bmm(group.lhs, group.rhs.transpose(1, 2)) for group in groups]
    return [torch.bmm(group.lhs.transpose(1, 2), group.rhs) for group in groups]


@triton.jit
def _assemble_grouped_update_kernel(
    contracted,
    update,
    group_start: tl.constexpr,
    group_rows: tl.constexpr,
    length: tl.constexpr,
    channels: tl.constexpr,
    block_rows: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    # ``torch.bmm`` returns each exact-length group as
    # [record * channel, i, j].  The output projection wants row-major compact
    # rows [record, i, j, channel].  A naive ``permute(...).reshape(...)`` makes
    # PyTorch clone every group; this tiled copy does only the required layout
    # move and keeps the tax visible as one small kernel per exact length.
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    c = tl.program_id(1) * block_c + tl.arange(0, block_c)
    pair_rows = length * length
    record = row // pair_rows
    pair = row - record * pair_rows
    src = contracted + (record[:, None] * channels + c[None, :]) * pair_rows + pair[:, None]
    dst = update + (group_start + row[:, None]) * channels + c[None, :]
    mask = (row[:, None] < group_rows) & (c[None, :] < channels)
    tl.store(dst, tl.load(src, mask=mask, other=0.0), mask=mask)


def assemble_update(groups: list[OwnedGroup], contracted: list[torch.Tensor], rows: int) -> torch.Tensor:
    update = contracted[0].new_empty((rows, C_Z))
    for group, result in zip(groups, contracted, strict=True):
        length = group.length
        group_rows = group.count * length * length
        grid = (triton.cdiv(group_rows, 16), triton.cdiv(C_Z, 64))
        _assemble_grouped_update_kernel[grid](
            result, update, group.start, group_rows, length, C_Z, 16, 64, num_warps=4
        )
    return update


@triton.jit
def _update_row_stats_tiled_kernel(
    update,
    mean,
    rstd,
    rows: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    row = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    c = tl.arange(0, block_c).to(tl.int64)
    row_mask = row < rows
    c_mask = c < channels
    values = tl.load(
        update + row[:, None] * channels + c[None, :],
        mask=row_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mu = tl.sum(values, axis=1) / channels
    centered = tl.where(c_mask[None, :], values - mu[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / channels
    tl.store(mean + row, mu, mask=row_mask)
    tl.store(rstd + row, tl.rsqrt(var + eps), mask=row_mask)


@triton.jit
def _contracted_row_stats_tiled_kernel(
    contracted,
    mean,
    rstd,
    group_start: tl.constexpr,
    group_rows: tl.constexpr,
    length: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    # ``contracted`` is still in the exact-group GEMM layout
    # [record * channel, i, j].  This deliberately skips the usual assembly
    # into row-major [valid_row, channel] storage, so the benchmark can answer
    # whether the row-major update tensor is a useful cache-friendly ABI or just
    # memory traffic we should remove in the native kernel.
    row = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    c = tl.arange(0, block_c).to(tl.int64)
    pair_rows = length * length
    record = row // pair_rows
    pair = row - record * pair_rows
    mask = (row[:, None] < group_rows) & (c[None, :] < channels)
    values = tl.load(
        contracted + (record[:, None] * channels + c[None, :]) * pair_rows + pair[:, None],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mu = tl.sum(values, axis=1) / channels
    centered = tl.where(c[None, :] < channels, values - mu[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / channels
    compact_row = group_start + row
    tl.store(mean + compact_row, mu, mask=row < group_rows)
    tl.store(rstd + compact_row, tl.rsqrt(var + eps), mask=row < group_rows)


@triton.jit
def _output_dual_gemm_scatter_kernel(
    update,
    x_norm,
    z,
    dense_offsets,
    mean,
    rstd,
    norm_weight,
    norm_bias,
    gate_weight,
    value_weight,
    out,
    rows: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    # This is the output-side analogue of the direct input producer.  It keeps
    # the strong exact-group contraction unchanged, but removes two compact
    # materializations after it: normalized update rows and compact deltas.  Each
    # program computes a `[rows, output_channels]` tile and scatters the residual
    # directly into the padded dense tensor.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    out_c = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
    k_offsets = tl.arange(0, block_k).to(tl.int64)
    row_mask = row < rows
    out_mask = out_c < channels
    row_mean = tl.load(mean + row, mask=row_mask, other=0.0)[:, None]
    row_rstd = tl.load(rstd + row, mask=row_mask, other=1.0)[:, None]
    acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
    acc_value = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, channels, block_k):
        k = k0 + k_offsets
        k_mask = k < channels
        gate_x = tl.load(
            x_norm + row[:, None] * channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        upd = tl.load(
            update + row[:, None] * channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        norm_w = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        norm_b = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        upd = (upd - row_mean) * row_rstd * norm_w[None, :] + norm_b[None, :]
        gate_w = tl.load(
            gate_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        value_w = tl.load(
            value_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(gate_x, tl.trans(gate_w))
        acc_value += tl.dot(upd.to(tl.bfloat16), tl.trans(value_w))

    dense_row = tl.load(dense_offsets + row, mask=row_mask, other=0).to(tl.int64)
    ptr = dense_row[:, None] * channels + out_c[None, :]
    residual = tl.load(z + ptr, mask=row_mask[:, None] & out_mask[None, :], other=0.0)
    delta = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_value
    tl.store(out + ptr, delta + residual, mask=row_mask[:, None] & out_mask[None, :])


@triton.jit
def _output_dual_gemm_scatter_recompute_gate_kernel(
    update,
    z,
    dense_offsets,
    input_mean,
    input_rstd,
    output_mean,
    output_rstd,
    input_norm_weight,
    input_norm_bias,
    output_norm_weight,
    output_norm_bias,
    gate_weight,
    value_weight,
    out,
    rows: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    # Benchmark-only larger fusion boundary: do not read producer-written
    # compact x_norm.  Instead, rebuild the output-gate input from the original
    # dense z rows and the already-computed input LayerNorm statistics.  If this
    # wins, it means the compact x_norm write/read pair is real memory debt; if
    # it loses, the extra z reload plus normalization arithmetic costs more than
    # that saved traffic.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    out_c = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
    k_offsets = tl.arange(0, block_k).to(tl.int64)
    row_mask = row < rows
    out_mask = out_c < channels
    dense_row = tl.load(dense_offsets + row, mask=row_mask, other=0).to(tl.int64)
    in_mean = tl.load(input_mean + row, mask=row_mask, other=0.0)[:, None]
    in_rstd = tl.load(input_rstd + row, mask=row_mask, other=1.0)[:, None]
    out_mean = tl.load(output_mean + row, mask=row_mask, other=0.0)[:, None]
    out_rstd = tl.load(output_rstd + row, mask=row_mask, other=1.0)[:, None]
    acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
    acc_value = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, channels, block_k):
        k = k0 + k_offsets
        k_mask = k < channels
        raw_gate = tl.load(
            z + dense_row[:, None] * channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        in_w = tl.load(input_norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        in_b = tl.load(input_norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        gate_x = (raw_gate - in_mean) * in_rstd * in_w[None, :] + in_b[None, :]
        gate_x = gate_x.to(tl.bfloat16)

        upd = tl.load(
            update + row[:, None] * channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out_w = tl.load(output_norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        out_b = tl.load(output_norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        upd = (upd - out_mean) * out_rstd * out_w[None, :] + out_b[None, :]

        gate_w = tl.load(
            gate_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        value_w = tl.load(
            value_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(gate_x, tl.trans(gate_w))
        acc_value += tl.dot(upd.to(tl.bfloat16), tl.trans(value_w))

    ptr = dense_row[:, None] * channels + out_c[None, :]
    residual = tl.load(z + ptr, mask=row_mask[:, None] & out_mask[None, :], other=0.0)
    delta = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_value
    tl.store(out + ptr, delta + residual, mask=row_mask[:, None] & out_mask[None, :])


@triton.jit
def _contracted_output_dual_gemm_scatter_recompute_gate_kernel(
    contracted,
    z,
    dense_offsets,
    input_mean,
    input_rstd,
    output_mean,
    output_rstd,
    input_norm_weight,
    input_norm_bias,
    output_norm_weight,
    output_norm_bias,
    gate_weight,
    value_weight,
    out,
    group_start: tl.constexpr,
    group_rows: tl.constexpr,
    length: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    # Same output boundary as ``recompute_gate``, but it consumes the
    # contraction result in [record * channel, i, j] layout.  If this wins, a
    # future CuTe kernel should keep the contraction output in its native layout
    # and fuse the output projection instead of assembling a row-major update.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    local_row = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    out_c = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
    k_offsets = tl.arange(0, block_k).to(tl.int64)
    global_row = group_start + local_row
    pair_rows = length * length
    record = local_row // pair_rows
    pair = local_row - record * pair_rows
    row_mask = local_row < group_rows
    out_mask = out_c < channels
    dense_row = tl.load(dense_offsets + global_row, mask=row_mask, other=0).to(tl.int64)
    in_mean = tl.load(input_mean + global_row, mask=row_mask, other=0.0)[:, None]
    in_rstd = tl.load(input_rstd + global_row, mask=row_mask, other=1.0)[:, None]
    out_mean = tl.load(output_mean + global_row, mask=row_mask, other=0.0)[:, None]
    out_rstd = tl.load(output_rstd + global_row, mask=row_mask, other=1.0)[:, None]
    acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
    acc_value = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, channels, block_k):
        k = k0 + k_offsets
        k_mask = k < channels
        raw_gate = tl.load(
            z + dense_row[:, None] * channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        in_w = tl.load(input_norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        in_b = tl.load(input_norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        gate_x = ((raw_gate - in_mean) * in_rstd * in_w[None, :] + in_b[None, :]).to(
            tl.bfloat16
        )

        upd = tl.load(
            contracted + (record[:, None] * channels + k[None, :]) * pair_rows + pair[:, None],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out_w = tl.load(output_norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        out_b = tl.load(output_norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        upd = (upd - out_mean) * out_rstd * out_w[None, :] + out_b[None, :]

        gate_w = tl.load(
            gate_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        value_w = tl.load(
            value_weight + out_c[:, None] * channels + k[None, :],
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(gate_x, tl.trans(gate_w))
        acc_value += tl.dot(upd.to(tl.bfloat16), tl.trans(value_w))

    ptr = dense_row[:, None] * channels + out_c[None, :]
    residual = tl.load(z + ptr, mask=row_mask[:, None] & out_mask[None, :], other=0.0)
    delta = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_value
    tl.store(out + ptr, delta + residual, mask=row_mask[:, None] & out_mask[None, :])


def compact_finish_from_update_fused_output(
    module: torch.nn.Module,
    z: torch.Tensor,
    x_norm: torch.Tensor,
    update: torch.Tensor,
    dense_offsets: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fuse output LayerNorm, output gate/value GEMMs, and dense scatter.

    The current benchmark finish uses three post-contraction stages:
    CUEQ LayerNorm writes normalized compact update rows, CUEQ's dual-GEMM
    writes compact deltas, then a Triton kernel scatters the residual into the
    padded dense tensor.  This screen keeps the contraction identical and tests
    whether owning that whole output boundary is worthwhile.
    """

    rows = dense_offsets.numel()
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    stats_grid = (triton.cdiv(rows, 8),)
    _update_row_stats_tiled_kernel[stats_grid](
        update,
        mean,
        rstd,
        rows,
        C_Z,
        1e-5,
        8,
        C_Z,
        num_warps=8,
    )
    out = torch.empty_like(z)
    grid = (triton.cdiv(rows, 64), triton.cdiv(C_Z, 128))
    _output_dual_gemm_scatter_kernel[grid](
        update,
        x_norm,
        z,
        dense_offsets,
        mean,
        rstd,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        weights["g_out"],
        weights["p_out"],
        out,
        rows,
        C_Z,
        64,
        128,
        64,
        num_warps=4,
        num_stages=3,
    )
    return out


def compact_finish_from_update_recompute_gate(
    module: torch.nn.Module,
    z: torch.Tensor,
    update: torch.Tensor,
    dense_offsets: torch.Tensor,
    weights: dict[str, torch.Tensor],
    input_mean: torch.Tensor,
    input_rstd: torch.Tensor,
) -> torch.Tensor:
    """Fuse output finish while rebuilding the gate input instead of reading x_norm."""

    rows = dense_offsets.numel()
    output_mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    output_rstd = torch.empty_like(output_mean)
    stats_grid = (triton.cdiv(rows, 8),)
    _update_row_stats_tiled_kernel[stats_grid](
        update,
        output_mean,
        output_rstd,
        rows,
        C_Z,
        1e-5,
        8,
        C_Z,
        num_warps=8,
    )
    out = torch.empty_like(z)
    grid = (triton.cdiv(rows, 64), triton.cdiv(C_Z, 128))
    _output_dual_gemm_scatter_recompute_gate_kernel[grid](
        update,
        z,
        dense_offsets,
        input_mean,
        input_rstd,
        output_mean,
        output_rstd,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        weights["g_out"],
        weights["p_out"],
        out,
        rows,
        C_Z,
        64,
        128,
        64,
        num_warps=4,
        num_stages=3,
    )
    return out


def producer_owned_finish(
    module: torch.nn.Module,
    z: torch.Tensor,
    x_norm: torch.Tensor,
    groups: list[OwnedGroup],
    dense_offsets: torch.Tensor,
    direction: str,
    weights: dict[str, torch.Tensor],
    finish: str = "current",
) -> torch.Tensor:
    contracted = contract_groups(groups, direction)
    update = assemble_update(groups, contracted, dense_offsets.numel())
    if finish == "fused_output":
        return compact_finish_from_update_fused_output(
            module, z, x_norm, update, dense_offsets, weights
        )
    if finish != "current":
        raise ValueError(f"unknown finish implementation: {finish!r}")
    return compact_finish_from_update(module, z, x_norm, update, dense_offsets, weights)


def producer_owned_finish_recompute_gate(
    module: torch.nn.Module,
    z: torch.Tensor,
    groups: list[OwnedGroup],
    dense_offsets: torch.Tensor,
    direction: str,
    weights: dict[str, torch.Tensor],
    input_mean: torch.Tensor,
    input_rstd: torch.Tensor,
) -> torch.Tensor:
    contracted = contract_groups(groups, direction)
    update = assemble_update(groups, contracted, dense_offsets.numel())
    return compact_finish_from_update_recompute_gate(
        module, z, update, dense_offsets, weights, input_mean, input_rstd
    )


def producer_owned_finish_contracted_recompute_gate(
    module: torch.nn.Module,
    z: torch.Tensor,
    groups: list[OwnedGroup],
    dense_offsets: torch.Tensor,
    direction: str,
    weights: dict[str, torch.Tensor],
    input_mean: torch.Tensor,
    input_rstd: torch.Tensor,
) -> torch.Tensor:
    """Finish directly from exact-group contraction outputs.

    This is a benchmark-only boundary test.  The current best direct path does:
    ``contracted -> assemble row-major update -> stats -> output projection``.
    This variant does ``contracted -> stats/output projection`` and therefore
    answers whether a native triangle-update kernel should keep the contraction
    result in its natural exact-group layout through the output epilogue.
    """

    contracted = contract_groups(groups, direction)
    rows = dense_offsets.numel()
    output_mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    output_rstd = torch.empty_like(output_mean)
    for group, result in zip(groups, contracted, strict=True):
        group_rows = group.count * group.length * group.length
        stats_grid = (triton.cdiv(group_rows, 8),)
        _contracted_row_stats_tiled_kernel[stats_grid](
            result,
            output_mean,
            output_rstd,
            group.start,
            group_rows,
            group.length,
            C_Z,
            1e-5,
            8,
            C_Z,
            num_warps=8,
        )

    out = torch.empty_like(z)
    for group, result in zip(groups, contracted, strict=True):
        group_rows = group.count * group.length * group.length
        grid = (triton.cdiv(group_rows, 64), triton.cdiv(C_Z, 128))
        _contracted_output_dual_gemm_scatter_recompute_gate_kernel[grid](
            result,
            z,
            dense_offsets,
            input_mean,
            input_rstd,
            output_mean,
            output_rstd,
            module.layer_norm_in.weight,
            module.layer_norm_in.bias,
            module.layer_norm_out.weight,
            module.layer_norm_out.bias,
            weights["g_out"],
            weights["p_out"],
            out,
            group.start,
            group_rows,
            group.length,
            C_Z,
            64,
            128,
            64,
            num_warps=4,
            num_stages=3,
        )
    return out


def producer_owned_output_only(
    module: torch.nn.Module,
    z: torch.Tensor,
    x_norm: torch.Tensor,
    groups: list[OwnedGroup],
    contracted: list[torch.Tensor],
    dense_offsets: torch.Tensor,
    weights: dict[str, torch.Tensor],
    finish: str = "current",
) -> torch.Tensor:
    update = assemble_update(groups, contracted, dense_offsets.numel())
    if finish == "fused_output":
        return compact_finish_from_update_fused_output(
            module, z, x_norm, update, dense_offsets, weights
        )
    if finish != "current":
        raise ValueError(f"unknown finish implementation: {finish!r}")
    return compact_finish_from_update(module, z, x_norm, update, dense_offsets, weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="both")
    parser.add_argument(
        "--config",
        type=ContractConfig.parse,
        default=ContractConfig.parse("32x32x64x1x4x3"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--finish", choices=["current", "fused_output"], default="current")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    lengths = lengths_from_args(args)
    modules = make_modules(args)
    z, mask = make_inputs(lengths, args.seed)
    dense_offsets, _ = make_metadata(lengths, args.config)
    directions = ["outgoing", "incoming"] if args.direction == "both" else [args.direction]

    results: dict[str, Any] = {}
    for direction in directions:
        module = modules[direction]
        weights = cached_weights(module)
        base_out, base_ms = cuda_time(
            lambda module=module: baseline_update(module, z, mask),
            args.warmup,
            args.iters,
        )
        producer_out, producer_ms = cuda_time(
            lambda: triton_producer_intermediates(module, z, dense_offsets, weights),
            args.warmup,
            args.iters,
        )
        x_norm, ab = producer_out
        groups = make_owned_groups(ab, lengths)
        contracted = contract_groups(groups, direction)

        finish_out, finish_ms = cuda_time(
            lambda: producer_owned_finish(
                module, z, x_norm, groups, dense_offsets, direction, weights, args.finish
            ),
            args.warmup,
            args.iters,
        )
        output_out, output_ms = cuda_time(
            lambda: producer_owned_output_only(
                module, z, x_norm, groups, contracted, dense_offsets, weights, args.finish
            ),
            args.warmup,
            args.iters,
        )
        ref_valid = valid_from_dense(base_out, dense_offsets)
        results[direction] = {
            "baseline_cueq": {"ms": base_ms},
            "current_triton_producer_only": {"ms": producer_ms},
            "producer_owned_finish": {
                "ms": finish_ms,
                "speedup_vs_baseline": base_ms / finish_ms,
                "parity_valid": compare(ref_valid, valid_from_dense(finish_out, dense_offsets)),
            },
            "producer_owned_output_only": {
                "ms": output_ms,
                "speedup_vs_baseline": base_ms / output_ms,
                "parity_valid": compare(ref_valid, valid_from_dense(output_out, dense_offsets)),
            },
            "group_count": len(groups),
        }

    n_max = max(lengths)
    payload = {
        "args": {**vars(args), "config": args.config.label(), "lengths": lengths},
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": len(lengths),
            "n_max": n_max,
            "c_z": C_Z,
            "valid_pair_efficiency": sum(length * length for length in lengths)
            / (len(lengths) * n_max * n_max),
            "valid_cubic_efficiency": sum(length**3 for length in lengths)
            / (len(lengths) * n_max**3),
        },
        "results": results,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
