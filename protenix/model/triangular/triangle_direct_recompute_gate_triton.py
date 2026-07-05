"""Default-off direct v2 triangle update with recomputed output gate.

This is the production-facing version of the best benchmark boundary from
``scripts/perf/triangle_update_producer_owned_direct_probe.py``.  It is guarded
behind ``PROTENIX_TRITON_TRIANGLE_DIRECT_RECOMPUTE_GATE`` because it deliberately
changes the internal schedule for Protenix-v2 mixed-token inference:

* compact valid pair rows are grouped by exact token length;
* the input LayerNorm + dual gated projection writes exact-group contraction
  tensors directly;
* the output gate recomputes its input from dense ``z`` instead of reading a
  producer-written compact ``x_norm`` tensor.

The last point is the important GPU lesson: storing ``x_norm`` is a large HBM
write/read product.  On H100 v2 screens, recomputing it inside the output kernel
was cheaper than moving it through memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_CHANNELS = 256
_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 64
_ROW_STAT_BLOCK_ROWS = 8
_METADATA_CACHE_LIMIT = 32
_metadata_cache: dict[tuple[object, ...], tuple[tuple[int, ...], torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class _Group:
    length: int
    start: int
    count: int
    lhs: torch.Tensor
    rhs: torch.Tensor


def triton_triangle_direct_recompute_gate_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TRIANGLE_DIRECT_RECOMPUTE_GATE", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def triton_triangle_direct_recompute_gate_available() -> bool:
    return triton is not None and tl is not None


if triton_triangle_direct_recompute_gate_available():

    @triton.jit
    def _row_stats_tiled_kernel(
        z,
        dense_offsets,
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
        dense_row = tl.load(dense_offsets + row, mask=row_mask, other=0).to(tl.int64)
        values = tl.load(
            z + dense_row[:, None] * channels + c[None, :],
            mask=row_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mu = tl.sum(values, axis=1) / channels
        centered = tl.where(c_mask[None, :], values - mu[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / channels
        tl.store(mean + row, mu, mask=row_mask)
        tl.store(rstd + row, tl.rsqrt(var + eps), mask=row_mask)

    @triton.jit
    def _producer_kernel(
        z,
        dense_offsets,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        proj_weight,
        lhs,
        rhs,
        group_start: tl.constexpr,
        group_rows: tl.constexpr,
        length: tl.constexpr,
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
        global_rows = group_start + offs_m
        dense_rows = tl.load(
            dense_offsets + global_rows, mask=offs_m < group_rows, other=0
        ).to(tl.int64)
        row_mean = tl.load(mean + global_rows, mask=offs_m < group_rows, other=0.0)[
            :, None
        ]
        row_rstd = tl.load(rstd + global_rows, mask=offs_m < group_rows, other=1.0)[
            :, None
        ]
        acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
        acc_proj = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k0 in range(0, channels, block_k):
            k = k0 + offs_k
            k_mask = k < channels
            values = tl.load(
                z + dense_rows[:, None] * channels + k[None, :],
                mask=(offs_m[:, None] < group_rows) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
            bias = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
            values = ((values - row_mean) * row_rstd * weight[None, :] + bias[None, :]).to(
                tl.bfloat16
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

        result = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj
        pair_rows = length * length
        record = offs_m // pair_rows
        pair = offs_m - record * pair_rows
        lhs_ptr = lhs + (record[:, None] * channels + offs_n[None, :]) * pair_rows + pair[
            :, None
        ]
        rhs_d = offs_n - channels
        rhs_ptr = rhs + (record[:, None] * channels + rhs_d[None, :]) * pair_rows + pair[
            :, None
        ]
        valid = (offs_m[:, None] < group_rows) & (offs_n[None, :] < out_channels)
        tl.store(lhs_ptr, result, mask=valid & (offs_n[None, :] < channels))
        tl.store(rhs_ptr, result, mask=valid & (offs_n[None, :] >= channels))

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
    def _assemble_kernel(
        contracted,
        update,
        group_start: tl.constexpr,
        group_rows: tl.constexpr,
        length: tl.constexpr,
        channels: tl.constexpr,
        block_rows: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        c = tl.program_id(1) * block_c + tl.arange(0, block_c)
        pair_rows = length * length
        record = row // pair_rows
        pair = row - record * pair_rows
        src = contracted + (record[:, None] * channels + c[None, :]) * pair_rows + pair[
            :, None
        ]
        dst = update + (group_start + row[:, None]) * channels + c[None, :]
        mask = (row[:, None] < group_rows) & (c[None, :] < channels)
        tl.store(dst, tl.load(src, mask=mask, other=0.0), mask=mask)

    @triton.jit
    def _output_kernel(
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
            gate_x = ((raw_gate - in_mean) * in_rstd * in_w[None, :] + in_b[None, :]).to(
                tl.bfloat16
            )

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
    def _zero_offsets_kernel(
        z,
        offsets,
        rows,
        channels: tl.constexpr,
        block_rows: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        c = tl.program_id(1) * block_c + tl.arange(0, block_c)
        dense_row = tl.load(offsets + row, mask=row < rows, other=0).to(tl.int64)
        mask = (row[:, None] < rows) & (c[None, :] < channels)
        tl.store(z + dense_row[:, None] * channels + c[None, :], 0.0, mask=mask)


def _mask_cache_key(mask: torch.Tensor) -> tuple[object, ...]:
    return (
        mask.untyped_storage().data_ptr(),
        mask.storage_offset(),
        tuple(mask.shape),
        tuple(mask.stride()),
        str(mask.device),
        mask.dtype,
        getattr(mask, "_version", 0),
    )


def _metadata_from_mask(mask: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    key = _mask_cache_key(mask)
    cached = _metadata_cache.get(key)
    if cached is not None:
        return cached

    lengths = tuple(
        int(item)
        for item in torch.diagonal(mask, dim1=-2, dim2=-1)
        .sum(dim=-1)
        .to(torch.int64)
        .cpu()
        .tolist()
    )
    if len(set(lengths)) <= 1 or any(length <= 0 for length in lengths):
        empty = torch.empty(0, device=mask.device, dtype=torch.int64)
        result = (lengths, empty, empty)
        _metadata_cache[key] = result
        return result

    n_max = mask.shape[-1]
    dense_offsets: list[int] = []
    invalid_offsets: list[int] = []
    for length in sorted(set(lengths)):
        for batch_index, item in enumerate(lengths):
            if item != length:
                continue
            for i in range(length):
                base = batch_index * n_max * n_max + i * n_max
                dense_offsets.extend(base + j for j in range(length))
    for batch_index, length in enumerate(lengths):
        batch_base = batch_index * n_max * n_max
        for i in range(n_max):
            invalid_start = length if i < length else 0
            invalid_offsets.extend(batch_base + i * n_max + j for j in range(invalid_start, n_max))

    result = (
        lengths,
        torch.tensor(dense_offsets, device=mask.device, dtype=torch.int64),
        torch.tensor(invalid_offsets, device=mask.device, dtype=torch.int64),
    )
    if len(_metadata_cache) >= _METADATA_CACHE_LIMIT:
        _metadata_cache.pop(next(iter(_metadata_cache)))
    _metadata_cache[key] = result
    return result


def _group_spans(lengths: tuple[int, ...]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for length in sorted(set(lengths)):
        count = sum(1 for item in lengths if item == length)
        spans.append((offset, length, count))
        offset += count * length * length
    return spans


def _make_groups(
    z: torch.Tensor,
    dense_offsets: torch.Tensor,
    lengths: tuple[int, ...],
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
) -> tuple[list[_Group], torch.Tensor, torch.Tensor]:
    rows = dense_offsets.numel()
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _row_stats_tiled_kernel[(triton.cdiv(rows, _ROW_STAT_BLOCK_ROWS),)](
        z, dense_offsets, mean, rstd, rows, _CHANNELS, eps, _ROW_STAT_BLOCK_ROWS, _CHANNELS, num_warps=8
    )
    groups: list[_Group] = []
    for start, length, count in _group_spans(lengths):
        group_rows = count * length * length
        lhs = torch.empty((count * _CHANNELS, length, length), device=z.device, dtype=z.dtype)
        rhs = torch.empty_like(lhs)
        grid = (triton.cdiv(group_rows, _BLOCK_M), triton.cdiv(2 * _CHANNELS, _BLOCK_N))
        _producer_kernel[grid](
            z,
            dense_offsets,
            mean,
            rstd,
            norm_weight,
            norm_bias,
            gate_weight,
            proj_weight,
            lhs,
            rhs,
            start,
            group_rows,
            length,
            _CHANNELS,
            2 * _CHANNELS,
            _BLOCK_M,
            _BLOCK_N,
            _BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        groups.append(_Group(length=length, start=start, count=count, lhs=lhs, rhs=rhs))
    return groups, mean, rstd


def _contract_and_assemble(groups: list[_Group], rows: int, direction: str) -> torch.Tensor:
    update = torch.empty((rows, _CHANNELS), device=groups[0].lhs.device, dtype=groups[0].lhs.dtype)
    for group in groups:
        if direction == "outgoing":
            contracted = torch.bmm(group.lhs, group.rhs.transpose(1, 2))
        else:
            contracted = torch.bmm(group.lhs.transpose(1, 2), group.rhs)
        group_rows = group.count * group.length * group.length
        grid = (triton.cdiv(group_rows, 16), triton.cdiv(_CHANNELS, 64))
        _assemble_kernel[grid](
            contracted, update, group.start, group_rows, group.length, _CHANNELS, 16, 64, num_warps=4
        )
    return update


def _finish(
    z: torch.Tensor,
    update: torch.Tensor,
    dense_offsets: torch.Tensor,
    invalid_offsets: torch.Tensor,
    input_mean: torch.Tensor,
    input_rstd: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows = dense_offsets.numel()
    output_mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    output_rstd = torch.empty_like(output_mean)
    _update_row_stats_tiled_kernel[(triton.cdiv(rows, _ROW_STAT_BLOCK_ROWS),)](
        update,
        output_mean,
        output_rstd,
        rows,
        _CHANNELS,
        eps,
        _ROW_STAT_BLOCK_ROWS,
        _CHANNELS,
        num_warps=8,
    )
    out = torch.empty_like(z)
    grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(_CHANNELS, _BLOCK_N))
    _output_kernel[grid](
        update,
        z,
        dense_offsets,
        input_mean,
        input_rstd,
        output_mean,
        output_rstd,
        norm_in_weight,
        norm_in_bias,
        norm_out_weight,
        norm_out_bias,
        g_out_weight,
        p_out_weight,
        out,
        rows,
        _CHANNELS,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    if invalid_offsets.numel() > 0:
        zero_grid = (triton.cdiv(invalid_offsets.numel(), 16), triton.cdiv(_CHANNELS, 64))
        _zero_offsets_kernel[zero_grid](
            out, invalid_offsets, invalid_offsets.numel(), _CHANNELS, 16, 64, num_warps=4
        )
    return out


def triangle_update_direct_recompute_gate(
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
    if not triton_triangle_direct_recompute_gate_enabled():
        return None
    if not triton_triangle_direct_recompute_gate_available():
        return None
    if torch.is_grad_enabled() or mask is None:
        return None
    if z.dim() != 4 or z.dtype != torch.bfloat16 or not z.is_cuda or not z.is_contiguous():
        return None
    if z.shape[-1] != _CHANNELS or z.shape[-2] != z.shape[-3] or mask.shape != z.shape[:-1]:
        return None
    if direction not in {"outgoing", "incoming"}:
        return None
    expected_in = (2 * _CHANNELS, _CHANNELS)
    expected_out = (_CHANNELS, _CHANNELS)
    if p_in_weight.shape != expected_in or g_in_weight.shape != expected_in:
        return None
    if p_out_weight.shape != expected_out or g_out_weight.shape != expected_out:
        return None
    if not all(weight.is_cuda for weight in (p_in_weight, g_in_weight, p_out_weight, g_out_weight)):
        return None
    if not all(weight.dtype == z.dtype for weight in (p_in_weight, g_in_weight, p_out_weight, g_out_weight)):
        return None

    lengths, dense_offsets, invalid_offsets = _metadata_from_mask(mask)
    if len(set(lengths)) <= 1 or any(length <= 0 for length in lengths):
        return None
    groups, input_mean, input_rstd = _make_groups(
        z,
        dense_offsets,
        lengths,
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        eps,
    )
    update = _contract_and_assemble(groups, dense_offsets.numel(), direction)
    return _finish(
        z,
        update,
        dense_offsets,
        invalid_offsets,
        input_mean,
        input_rstd,
        norm_in_weight,
        norm_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
        eps,
    )
