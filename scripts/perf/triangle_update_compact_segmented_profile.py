#!/usr/bin/env python3
"""Profiler view for the compact segmented triangle-update screen.

The timing screen answers whether the larger compact boundary wins.  This helper
answers why a candidate wins or loses by printing the dominant CUDA events for
one baseline call and one compact segmented call under the same shape.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from scripts.perf.triangle_update_compact_segmented_probe import (
    ContractConfig,
    DTYPE,
    LONG_LENGTHS,
    SHORT_LENGTHS,
    baseline_update,
    cached_weights,
    cueq_producer_update,
    make_inputs,
    make_metadata,
    make_modules,
    parse_ints,
    triton_producer_update,
)


def profile_one(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    limit: int,
) -> dict[str, Any]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as prof:
            out = fn()
            torch.cuda.synchronize()
    del out
    rows = []
    for item in prof.key_averages().table(sort_by="self_cuda_time_total").splitlines():
        if item.strip():
            rows.append(item)
    top_events = []
    for event in prof.key_averages():
        cuda_total = getattr(event, "cuda_time_total", 0.0) or getattr(
            event, "device_time_total", 0.0
        )
        self_cuda = getattr(event, "self_cuda_time_total", 0.0) or getattr(
            event, "self_device_time_total", 0.0
        )
        if cuda_total <= 0 and self_cuda <= 0:
            continue
        top_events.append(
            {
                "key": event.key,
                "count": event.count,
                "cuda_total_ms": cuda_total / 1000.0,
                "self_cuda_ms": self_cuda / 1000.0,
            }
        )
    top_events.sort(key=lambda row: row["self_cuda_ms"], reverse=True)
    return {"top_events": top_events[:limit], "table": rows[: limit + 8]}


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument("--producer", choices=["triton", "cueq"], default="triton")
    parser.add_argument(
        "--contractor",
        choices=["triton", "exact_group_bmm"],
        default="triton",
        help=(
            "Contraction backend inside the compact update. Use "
            "'exact_group_bmm' to profile the optimistic exact-length bmm "
            "bridge from the full-update screen."
        ),
    )
    parser.add_argument(
        "--config",
        type=ContractConfig.parse,
        default=ContractConfig.parse("32x32x64x1x4x3"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
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
    module = modules[args.direction]
    z, mask = make_inputs(lengths, args.seed)
    dense_offsets, descriptors = make_metadata(lengths, args.config)
    weights = cached_weights(module)
    candidate_fn = triton_producer_update if args.producer == "triton" else cueq_producer_update

    baseline = profile_one(
        lambda: baseline_update(module, z, mask),
        warmup=args.warmup,
        limit=args.limit,
    )
    candidate = profile_one(
        lambda: candidate_fn(
            module,
            z,
            dense_offsets,
            descriptors,
            args.config,
            args.direction,
            weights,
            args.contractor,
            lengths,
        ),
        warmup=args.warmup,
        limit=args.limit,
    )
    payload = {
        "args": {
            **vars(args),
            "config": args.config.label(),
            "lengths": lengths,
        },
        "shape": {
            "batch": len(lengths),
            "n_max": max(lengths),
            "valid_rows": int(dense_offsets.numel()),
            "descriptor_count": int(descriptors.shape[0]),
        },
        "baseline_cueq": baseline,
        "compact_segmented": candidate,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
