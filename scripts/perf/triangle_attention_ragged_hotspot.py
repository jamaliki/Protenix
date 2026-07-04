#!/usr/bin/env python3
"""Ragged-screen the v2 CUEQ triangle-attention boundary.

Sam's production-like workload batches many different sequence lengths with
``N_sample=1..5``.  The current pairformer keeps that as one large padded
``B x N_max x N_max`` problem, which preserves large vendor CUEQ/cuDNN launches
but does attention work for padded rows and columns.

This script is a lower-bound screen, not a production path.  It compares the
current single padded triangle-attention module against smaller "islands" that
reuse the same weights at shorter physical lengths.  Islands pay extra Python
launches and copies, so a win here means padding headroom is real.  A loss
means a production fix must be a true one-launch segmented/vendor-quality
kernel, not Python sub-batching.

The valid outputs are compared against the padded result.  Exact-length GPU
reductions are not guaranteed to be bitwise identical to padded reductions with
masked zeros, so the drift columns are evidence about numerical movement, not
necessarily mask leakage.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Callable

import torch

from protenix.model.triangular.triangular import (
    TriangleAttention,
    cueq_bool_mask_enabled,
    cueq_ending_contiguous_producer_enabled,
)


TensorFn = Callable[[], torch.Tensor]
DTYPE = torch.bfloat16


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def make_modules(args: argparse.Namespace) -> dict[str, TriangleAttention]:
    modules = {
        "start": TriangleAttention(
            c_in=args.c_z,
            c_hidden=args.c_hidden_pair_att,
            no_heads=args.no_heads_pair,
            starting=True,
        )
        .cuda()
        .eval(),
        "end": TriangleAttention(
            c_in=args.c_z,
            c_hidden=args.c_hidden_pair_att,
            no_heads=args.no_heads_pair,
            starting=False,
        )
        .cuda()
        .eval(),
    }

    # Several AF3/Protenix output projections are intentionally initialized to
    # zero.  That is correct for the model, but it can hide real memory and GEMM
    # traffic in a synthetic hotspot.  Randomize only all-zero matrices so the
    # benchmark exercises the production topology without changing shapes.
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 8191)
        with torch.no_grad():
            for module in modules.values():
                for parameter in module.parameters():
                    if parameter.ndim >= 2 and not torch.any(parameter):
                        parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return modules


def make_padded_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(lengths)
    z = torch.zeros(
        len(lengths),
        n_max,
        n_max,
        args.c_z,
        device="cuda",
        dtype=DTYPE,
    )
    mask = torch.zeros(len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length,
            length,
            args.c_z,
            device="cuda",
            dtype=DTYPE,
            generator=generator,
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def exact_length_groups(lengths: list[int]) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        groups[length].append(index)
    return [groups[length] for length in sorted(groups)]


def split_bucket_groups(lengths: list[int], limits: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    start = 0
    for limit in sorted(limits):
        group = [index for index, length in enumerate(lengths) if start < length <= limit]
        if group:
            groups.append(group)
        start = limit
    tail = [index for index, length in enumerate(lengths) if length > start]
    if tail:
        groups.append(tail)
    return groups


def group_stats(groups: list[list[int]], lengths: list[int]) -> dict[str, object]:
    scheduled_pairs = 0
    scheduled_cubic = 0
    for group in groups:
        n_group = max(lengths[index] for index in group)
        scheduled_pairs += len(group) * n_group * n_group
        scheduled_cubic += len(group) * n_group * n_group * n_group

    valid_pairs = sum(length * length for length in lengths)
    valid_cubic = sum(length * length * length for length in lengths)
    return {
        "groups": [[lengths[index] for index in group] for group in groups],
        "num_groups": len(groups),
        "valid_pairs": valid_pairs,
        "scheduled_pairs": scheduled_pairs,
        "valid_pair_efficiency": valid_pairs / scheduled_pairs,
        "ideal_pair_work_speedup": scheduled_pairs / valid_pairs,
        "valid_cubic": valid_cubic,
        "scheduled_cubic": scheduled_cubic,
        "valid_cubic_efficiency": valid_cubic / scheduled_cubic,
        "ideal_attention_work_speedup": scheduled_cubic / valid_cubic,
    }


def case_from_indices(
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    n_group = max(lengths[index] for index in indices)
    z_group = torch.zeros(
        len(indices),
        n_group,
        n_group,
        z.shape[-1],
        device=z.device,
        dtype=z.dtype,
    )
    mask_group = torch.zeros(
        len(indices),
        n_group,
        n_group,
        device=mask.device,
        dtype=mask.dtype,
    )
    group_lengths = []
    for out_index, source_index in enumerate(indices):
        length = lengths[source_index]
        group_lengths.append(length)
        z_group[out_index, :length, :length] = z[source_index, :length, :length]
        mask_group[out_index, :length, :length] = mask[source_index, :length, :length]
    return z_group, mask_group, group_lengths


def flatten_valid(z_values: list[torch.Tensor], lengths: list[int]) -> torch.Tensor:
    chunks = []
    for z, length in zip(z_values, lengths, strict=True):
        chunks.append(z[:length, :length].reshape(-1, z.shape[-1]))
    return torch.cat(chunks, dim=0)


def run_attention(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Match PairformerBlock's inference boundary: triangle attention produces an
    # update that is immediately added to the pair representation.  Including
    # the residual add charges the padded HBM pass that a segmented boundary
    # would also need to address or fuse away.
    base = z.clone()
    return base + module(base, mask=mask, triangle_attention="cuequivariance")


def run_padded_valid(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    out = run_attention(module, z, mask)
    return flatten_valid(list(out), lengths)


def run_grouped(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    groups: list[list[int]],
) -> torch.Tensor:
    # Preserve input order in the flattened comparison even when groups are
    # sorted by bucket.  This keeps the script safe for non-sorted test queues.
    output_by_index: list[torch.Tensor | None] = [None] * len(lengths)
    for indices in groups:
        z_group, mask_group, _ = case_from_indices(z, mask, lengths, indices)
        out_group = run_attention(module, z_group, mask_group)
        for out_index, source_index in enumerate(indices):
            output_by_index[source_index] = out_group[out_index]
    if any(value is None for value in output_by_index):
        raise RuntimeError("grouping did not cover every input row")
    return flatten_valid([value for value in output_by_index if value is not None], lengths)


def cuda_time(
    fn: TensorFn,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))

    median_ms = statistics.median(samples)
    stats = {
        "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    return out, median_ms, stats


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
    )
    parser.add_argument("--bucket-limits", type=parse_ints, default=parse_ints("76"))
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_ragged_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    modules = make_modules(args)
    z, mask = make_padded_inputs(args.lengths, args)
    groups_by_name = {
        "padded_all": [list(range(len(args.lengths)))],
        "split_buckets": split_bucket_groups(args.lengths, args.bucket_limits),
        "exact_length_groups": exact_length_groups(args.lengths),
    }

    results: dict[str, object] = {}
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        for direction, module in modules.items():
            variants: dict[str, TensorFn] = {
                "padded_all": lambda module=module: run_padded_valid(
                    module,
                    z,
                    mask,
                    args.lengths,
                ),
                "split_buckets": lambda module=module: run_grouped(
                    module,
                    z,
                    mask,
                    args.lengths,
                    groups_by_name["split_buckets"],
                ),
                "exact_length_groups": lambda module=module: run_grouped(
                    module,
                    z,
                    mask,
                    args.lengths,
                    groups_by_name["exact_length_groups"],
                ),
            }
            reference, padded_ms, padded_stats = cuda_time(
                variants["padded_all"],
                args.warmup,
                args.iters,
            )
            direction_results: dict[str, object] = {
                "padded_all": {
                    "ms": padded_ms,
                    "speedup_vs_padded": 1.0,
                    **padded_stats,
                }
            }
            for name in ("split_buckets", "exact_length_groups"):
                output, ms, timing_stats = cuda_time(
                    variants[name],
                    args.warmup,
                    args.iters,
                )
                direction_results[name] = {
                    "ms": ms,
                    "speedup_vs_padded": padded_ms / ms,
                    **timing_stats,
                    "valid_vs_padded": compare(reference, output),
                }
            results[direction] = direction_results

    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "fixed_dtype": str(DTYPE),
                "feature_flags": {
                    "cueq_bool_mask_enabled": cueq_bool_mask_enabled(),
                    "cueq_ending_contiguous_producer_enabled": (
                        cueq_ending_contiguous_producer_enabled()
                    ),
                },
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                "group_stats": {
                    name: group_stats(groups, args.lengths)
                    for name, groups in groups_by_name.items()
                },
                "results": results,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
