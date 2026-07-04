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


def producer_owned_finish(
    module: torch.nn.Module,
    z: torch.Tensor,
    x_norm: torch.Tensor,
    groups: list[OwnedGroup],
    dense_offsets: torch.Tensor,
    direction: str,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    contracted = contract_groups(groups, direction)
    update = assemble_update(groups, contracted, dense_offsets.numel())
    return compact_finish_from_update(module, z, x_norm, update, dense_offsets, weights)


def producer_owned_output_only(
    module: torch.nn.Module,
    z: torch.Tensor,
    x_norm: torch.Tensor,
    groups: list[OwnedGroup],
    contracted: list[torch.Tensor],
    dense_offsets: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    update = assemble_update(groups, contracted, dense_offsets.numel())
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
                module, z, x_norm, groups, dense_offsets, direction, weights
            ),
            args.warmup,
            args.iters,
        )
        output_out, output_ms = cuda_time(
            lambda: producer_owned_output_only(
                module, z, x_norm, groups, contracted, dense_offsets, weights
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
