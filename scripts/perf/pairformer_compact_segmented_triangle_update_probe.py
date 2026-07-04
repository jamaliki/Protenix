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
from typing import Optional

import torch

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


def compact_triangle_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    config: ContractConfig,
    direction: str,
    producer: str,
) -> torch.Tensor:
    weights = cached_weights(module)
    if producer == "direct":
        dense_offsets = grouped_dense_offsets(lengths)
        out = direct_owned_update(module, z, dense_offsets, lengths, direction, weights)
        return zero_invalid_pairs(out, mask)

    dense_offsets, descriptors = make_metadata(lengths, config)
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
    )
    z = compact_triangle_update(
        block.tri_mul_in,
        z,
        mask,
        lengths,
        config,
        "incoming",
        producer,
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
            candidate_out, timing = cuda_time(
                lambda producer=producer: run_compact_triangles(
                    block,
                    s,
                    z,
                    mask,
                    args.lengths,
                    args.config,
                    producer,
                ),
                args.warmup,
                args.iters,
            )
            candidates.append(
                {
                    "variant": f"compact_triangle_update_{producer}",
                    "producer": producer,
                    "config": args.config.label(),
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
