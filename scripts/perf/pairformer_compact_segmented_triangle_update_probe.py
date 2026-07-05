#!/usr/bin/env python3
"""Full PairformerBlock gate for compact segmented triangle multiplication.

The compact segmented triangle-update screen showed a real short-bucket win but
a hard long-bucket loss.  This script asks the next promotion question: does the
short-bucket triangle-mul win survive a complete v2 PairformerBlock once later
attention, pair transition, and token paths are charged?

Important: the lower-level screen compares only valid rows and intentionally does
not materialize invalid padded output rows.  A full block cannot do that because
later projections may read padded rows before applying masks.  This gate zeros
invalid rows after each compact triangle update and charges that cost.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from pairformer_varlen_triangle_attention_probe import (
    TensorPair,
    autocast_context,
    compare_outputs,
    cuda_time,
    finish_block,
    make_block,
    make_inputs,
    parse_ints,
    run_baseline,
    str_bool,
)
from scripts.perf.triangle_update_compact_segmented_probe import (
    ContractConfig,
    cached_weights,
    cueq_producer_update,
    make_metadata,
    triton_producer_update,
)
from scripts.perf.triangle_update_producer_owned_direct_probe import (
    direct_owned_update,
    grouped_dense_offsets,
)


@dataclass(frozen=True)
class TriangleUpdatePlan:
    dense_offsets: torch.Tensor
    descriptors: torch.Tensor | None
    invalid_offsets: torch.Tensor | None
    weights_out: dict[str, torch.Tensor]
    weights_in: dict[str, torch.Tensor]


def parse_producers(text: str) -> list[str]:
    producers = [item.strip() for item in text.split(",") if item.strip()]
    bad = sorted(set(producers) - {"triton", "cueq", "direct"})
    if bad:
        raise argparse.ArgumentTypeError(f"unknown producer(s): {', '.join(bad)}")
    return producers


def clone_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.clone()


def zero_invalid_pairs(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Materialize the dense Pairformer contract after a compact valid-row update."""
    return z.masked_fill(~mask.bool().unsqueeze(-1), 0.0)


@triton.jit
def _zero_dense_offsets_kernel(
    z,
    dense_offsets,
    rows,
    channels: tl.constexpr,
    block_rows: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    c = tl.program_id(1) * block_c + tl.arange(0, block_c)
    dense_row = tl.load(dense_offsets + row, mask=row < rows, other=0).to(tl.int64)
    mask = (row[:, None] < rows) & (c[None, :] < channels)
    tl.store(z + dense_row[:, None] * channels + c[None, :], 0.0, mask=mask)


def make_invalid_dense_offsets(lengths: list[int]) -> torch.Tensor:
    n_max = max(lengths)
    offsets: list[int] = []
    for batch_index, length in enumerate(lengths):
        batch_base = batch_index * n_max * n_max
        for i in range(n_max):
            invalid_start = length if i < length else 0
            offsets.extend(batch_base + i * n_max + j for j in range(invalid_start, n_max))
    return torch.tensor(offsets, device="cuda", dtype=torch.int64)


def zero_invalid_offsets_(z: torch.Tensor, dense_offsets: torch.Tensor | None) -> torch.Tensor:
    if dense_offsets is None or dense_offsets.numel() == 0:
        return z
    rows = dense_offsets.numel()
    grid = (triton.cdiv(rows, 16), triton.cdiv(z.shape[-1], 64))
    _zero_dense_offsets_kernel[grid](z, dense_offsets, rows, z.shape[-1], 16, 64, num_warps=4)
    return z


def compact_triangle_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    config: ContractConfig,
    direction: str,
    producer: str,
    weights: dict[str, torch.Tensor],
    dense_offsets: torch.Tensor,
    descriptors: torch.Tensor | None,
    invalid_offsets: torch.Tensor | None,
    direct_row_stats: str,
    direct_row_stat_block_rows: int,
    direct_finish: str,
) -> torch.Tensor:
    if producer == "direct":
        out = direct_owned_update(
            module,
            z,
            dense_offsets,
            lengths,
            direction,
            weights,
            row_stats=direct_row_stats,
            row_stat_block_rows=direct_row_stat_block_rows,
            finish=direct_finish,
        )
        return zero_invalid_offsets_(out, invalid_offsets)

    if descriptors is None:
        raise RuntimeError("compact CUEQ/Triton producers require descriptors")
    update_fn = triton_producer_update if producer == "triton" else cueq_producer_update
    out = update_fn(
        module,
        z,
        dense_offsets,
        descriptors,
        config,
        direction,
        weights,
    )
    return zero_invalid_pairs(out, mask)


def run_compact_triangles(
    block,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    config: ContractConfig,
    producer: str,
    plan: TriangleUpdatePlan,
    direct_row_stats: str,
    direct_row_stat_block_rows: int,
    direct_finish: str,
) -> TensorPair:
    s = clone_optional(s)
    z = z.clone()
    z = compact_triangle_update(
        block.tri_mul_out,
        z,
        mask,
        lengths,
        config,
        "outgoing",
        producer,
        plan.weights_out,
        plan.dense_offsets,
        plan.descriptors,
        plan.invalid_offsets,
        direct_row_stats,
        direct_row_stat_block_rows,
        direct_finish,
    )
    z = compact_triangle_update(
        block.tri_mul_in,
        z,
        mask,
        lengths,
        config,
        "incoming",
        producer,
        plan.weights_in,
        plan.dense_offsets,
        plan.descriptors,
        plan.invalid_offsets,
        direct_row_stats,
        direct_row_stat_block_rows,
        direct_finish,
    )
    z += block.tri_att_start(
        z,
        mask=mask,
        triangle_attention="cuequivariance",
        inplace_safe=True,
        chunk_size=None,
    )
    z += block.tri_att_end._cueq_ending_contiguous_forward(z, mask)
    return finish_block(block, s, z, mask)


def make_triangle_update_plan(
    block,
    lengths: list[int],
    config: ContractConfig,
    producer: str,
) -> TriangleUpdatePlan:
    if producer == "direct":
        dense_offsets = grouped_dense_offsets(lengths)
        descriptors = None
        invalid_offsets = make_invalid_dense_offsets(lengths)
    else:
        dense_offsets, descriptors = make_metadata(lengths, config)
        invalid_offsets = None
    return TriangleUpdatePlan(
        dense_offsets=dense_offsets,
        descriptors=descriptors,
        invalid_offsets=invalid_offsets,
        weights_out=cached_weights(block.tri_mul_out),
        weights_in=cached_weights(block.tri_mul_in),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
    )
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--producer", type=parse_producers, default=parse_producers("triton,cueq"))
    parser.add_argument("--config", type=ContractConfig.parse, default=ContractConfig.parse("32x32x64x1x4x3"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument("--direct-row-stats", choices=["scalar", "tiled"], default="tiled")
    parser.add_argument("--direct-row-stat-block-rows", type=int, default=8)
    parser.add_argument(
        "--direct-finish",
        choices=["current", "fused_output", "recompute_gate"],
        default="current",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    block = make_block(args)
    s, z, mask = make_inputs(args.lengths, args)

    with torch.inference_mode(), autocast_context(args.input_dtype):
        baseline_out, baseline_timing = cuda_time(
            lambda: run_baseline(block, s, z, mask),
            args.warmup,
            args.iters,
        )
        candidates = []
        for producer in args.producer:
            plan = make_triangle_update_plan(block, args.lengths, args.config, producer)
            candidate_out, timing = cuda_time(
                lambda producer=producer, plan=plan: run_compact_triangles(
                    block,
                    s,
                    z,
                    mask,
                    args.lengths,
                    args.config,
                    producer,
                    plan,
                    args.direct_row_stats,
                    args.direct_row_stat_block_rows,
                    args.direct_finish,
                ),
                args.warmup,
                args.iters,
            )
            candidates.append(
                {
                    "variant": f"compact_triangle_update_{producer}",
                    "producer": producer,
                    "config": args.config.label(),
                    "direct_row_stats": args.direct_row_stats if producer == "direct" else None,
                    "direct_row_stat_block_rows": (
                        args.direct_row_stat_block_rows if producer == "direct" else None
                    ),
                    "direct_finish": args.direct_finish if producer == "direct" else None,
                    "timing": timing,
                    "speedup_vs_baseline": baseline_timing["ms"] / timing["ms"],
                    "parity_vs_baseline_valid": compare_outputs(
                        baseline_out,
                        candidate_out,
                        args.lengths,
                    ),
                }
            )

    best = max(candidates, key=lambda row: row["speedup_vs_baseline"])
    payload = {
        "args": {
            **vars(args),
            "config": args.config.label(),
            "lengths": args.lengths,
        },
        "baseline": baseline_timing,
        "candidates": candidates,
        "best": best,
        "device": torch.cuda.get_device_name(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        from pathlib import Path

        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
