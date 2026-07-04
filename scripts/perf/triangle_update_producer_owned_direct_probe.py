#!/usr/bin/env python3
"""Screen a direct producer-owned exact-group triangle update.

The previous upper-bound screen pre-materialized exact-group contractor inputs
before timing.  This script includes the first missing production step: a Triton
input LayerNorm + dual-GEMM producer writes the grouped contractor layout
directly, while also writing row-major ``x_norm`` for the current output gate.

This is still a benchmark, not production code.  It intentionally keeps the
output boundary from ``triangle_update_producer_owned_finish_probe.py`` so the
only new variable is whether direct producer layout pays for itself.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from scripts.perf.triangle_update_compact_segmented_probe import (
    C_Z,
    LONG_LENGTHS,
    SHORT_LENGTHS,
    ContractConfig,
    _compact_row_stats_kernel,
    baseline_update,
    cached_weights,
    compare,
    cuda_time,
    make_inputs,
    make_modules,
    parse_ints,
    valid_from_dense,
)
from scripts.perf.triangle_update_producer_owned_finish_probe import (
    OwnedGroup,
    producer_owned_finish,
)


@triton.jit
def _owned_group_ln_dual_gemm_kernel(
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
    x_norm,
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
    dense_rows = tl.load(dense_offsets + global_rows, mask=offs_m < group_rows, other=0).to(
        tl.int64
    )
    row_mean = tl.load(mean + global_rows, mask=offs_m < group_rows, other=0.0)[:, None]
    row_rstd = tl.load(rstd + global_rows, mask=offs_m < group_rows, other=1.0)[:, None]
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
        values = (values - row_mean) * row_rstd * weight[None, :] + bias[None, :]
        values = values.to(tl.bfloat16)
        tl.store(
            x_norm + global_rows[:, None] * channels + k[None, :],
            values,
            mask=(pid_n == 0) & (offs_m[:, None] < group_rows) & k_mask[None, :],
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
    lhs_ptr = lhs + (record[:, None] * channels + offs_n[None, :]) * pair_rows + pair[:, None]
    rhs_d = offs_n - channels
    rhs_ptr = rhs + (record[:, None] * channels + rhs_d[None, :]) * pair_rows + pair[:, None]
    valid = (offs_m[:, None] < group_rows) & (offs_n[None, :] < out_channels)
    tl.store(lhs_ptr, result, mask=valid & (offs_n[None, :] < channels))
    tl.store(rhs_ptr, result, mask=valid & (offs_n[None, :] >= channels))


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def group_spans(lengths: list[int]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for length in sorted(set(lengths)):
        count = sum(1 for item in lengths if item == length)
        spans.append((offset, length, count))
        offset += count * length * length
    return spans


def grouped_dense_offsets(lengths: list[int]) -> torch.Tensor:
    """Gather valid pair rows by exact length, independent of input order.

    The eventual production path can use any compact order as long as it scatters
    back to the original dense rows. Grouping here keeps one contractor shape per
    exact sequence length, which is the boundary this benchmark is testing.
    """

    n_max = max(lengths)
    offsets: list[int] = []
    for length in sorted(set(lengths)):
        for batch_index, item in enumerate(lengths):
            if item != length:
                continue
            for i in range(length):
                base = batch_index * n_max * n_max + i * n_max
                offsets.extend(base + j for j in range(length))
    return torch.tensor(offsets, device="cuda", dtype=torch.int64)


def direct_owned_groups(
    module: torch.nn.Module,
    z: torch.Tensor,
    dense_offsets: torch.Tensor,
    lengths: list[int],
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[OwnedGroup]]:
    rows = dense_offsets.numel()
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _compact_row_stats_kernel[(rows,)](z, dense_offsets, mean, rstd, rows, C_Z, 1e-5)
    x_norm = torch.empty((rows, C_Z), device=z.device, dtype=z.dtype)
    groups: list[OwnedGroup] = []
    for start, length, count in group_spans(lengths):
        group_rows = count * length * length
        lhs = torch.empty((count * C_Z, length, length), device=z.device, dtype=z.dtype)
        rhs = torch.empty_like(lhs)
        grid = (triton.cdiv(group_rows, 64), triton.cdiv(2 * C_Z, 128))
        _owned_group_ln_dual_gemm_kernel[grid](
            z,
            dense_offsets,
            mean,
            rstd,
            module.layer_norm_in.weight,
            module.layer_norm_in.bias,
            weights["g_in"],
            weights["p_in"],
            lhs,
            rhs,
            x_norm,
            start,
            group_rows,
            length,
            C_Z,
            2 * C_Z,
            64,
            128,
            64,
            num_warps=4,
            num_stages=3,
        )
        groups.append(OwnedGroup(length=length, start=start, count=count, lhs=lhs, rhs=rhs))
    return x_norm, groups


def direct_owned_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    dense_offsets: torch.Tensor,
    lengths: list[int],
    direction: str,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    x_norm, groups = direct_owned_groups(module, z, dense_offsets, lengths, weights)
    return producer_owned_finish(module, z, x_norm, groups, dense_offsets, direction, weights)


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
    dense_offsets = grouped_dense_offsets(lengths)
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
            lambda: direct_owned_groups(module, z, dense_offsets, lengths, weights),
            args.warmup,
            args.iters,
        )
        x_norm, groups = producer_out
        candidate, candidate_ms = cuda_time(
            lambda: direct_owned_update(module, z, dense_offsets, lengths, direction, weights),
            args.warmup,
            args.iters,
        )
        finish_only, finish_ms = cuda_time(
            lambda: producer_owned_finish(module, z, x_norm, groups, dense_offsets, direction, weights),
            args.warmup,
            args.iters,
        )
        ref_valid = valid_from_dense(base_out, dense_offsets)
        results[direction] = {
            "baseline_cueq": {"ms": base_ms},
            "direct_producer_only": {"ms": producer_ms},
            "direct_producer_full": {
                "ms": candidate_ms,
                "speedup_vs_baseline": base_ms / candidate_ms,
                "parity_valid": compare(ref_valid, valid_from_dense(candidate, dense_offsets)),
            },
            "finish_only_reusing_direct_producer": {
                "ms": finish_ms,
                "speedup_vs_baseline": base_ms / finish_ms,
                "parity_valid": compare(ref_valid, valid_from_dense(finish_only, dense_offsets)),
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
