#!/usr/bin/env python3
"""Ragged-screen the v2 CUEQ triangle-multiplication boundary.

This is not a production path.  It times the current single padded CUEQ launch
against exact-length or smaller-bucket "islands" using the same weights.  The
islands pay extra launches and copies, so they are a lower-bound probe: if they
win anyway, padding dominates; if they lose while pair efficiency is poor, the
needed fix is a one-launch segmented kernel rather than more Python bucketing.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from collections.abc import Callable

import torch

from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


TensorFn = Callable[[], torch.Tensor]
DTYPE = torch.bfloat16
C_Z = 256


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def make_modules(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    modules = {
        "out": TriangleMultiplicationOutgoing(C_Z, C_Z).cuda().eval(),
        "in": TriangleMultiplicationIncoming(C_Z, C_Z).cuda().eval(),
    }
    # AF3 zero-initializes some final projections.  That hides real work in a
    # synthetic hotspot, so randomize only all-zero matrices.
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1729)
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
    z = torch.zeros(len(lengths), n_max, n_max, C_Z, device="cuda", dtype=DTYPE)
    mask = torch.zeros(len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
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
    padded_pairs = 0
    for group in groups:
        n_group = max(lengths[index] for index in group)
        padded_pairs += len(group) * n_group * n_group
    valid_pairs = sum(length * length for length in lengths)
    return {
        "groups": [[lengths[index] for index in group] for group in groups],
        "num_groups": len(groups),
        "valid_pairs": valid_pairs,
        "padded_pairs": padded_pairs,
        "valid_pair_efficiency": valid_pairs / padded_pairs,
        "ideal_pair_work_speedup": padded_pairs / valid_pairs,
    }


def case_from_indices(
    z: torch.Tensor,
    lengths: list[int],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    n_group = max(lengths[index] for index in indices)
    z_group = torch.zeros(len(indices), n_group, n_group, C_Z, device=z.device, dtype=z.dtype)
    mask_group = torch.zeros(len(indices), n_group, n_group, device=z.device, dtype=z.dtype)
    group_lengths = []
    for out_index, source_index in enumerate(indices):
        length = lengths[source_index]
        group_lengths.append(length)
        z_group[out_index, :length, :length] = z[source_index, :length, :length]
        mask_group[out_index, :length, :length] = 1
    return z_group, mask_group, group_lengths


def flatten_valid(z_values: list[torch.Tensor], lengths: list[int]) -> torch.Tensor:
    chunks = []
    for z, length in zip(z_values, lengths, strict=True):
        chunks.append(z[:length, :length].reshape(-1, z.shape[-1]))
    return torch.cat(chunks, dim=0)


def run_tri_mul(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    return module(
        z.clone(),
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )


def run_padded_valid(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    args: argparse.Namespace,
) -> torch.Tensor:
    out = run_tri_mul(module, z, mask, args)
    return flatten_valid(list(out), lengths)


def run_grouped(
    module: torch.nn.Module,
    z: torch.Tensor,
    lengths: list[int],
    groups: list[list[int]],
    args: argparse.Namespace,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    output_lengths: list[int] = []
    for indices in groups:
        z_group, mask_group, group_lengths = case_from_indices(z, lengths, indices)
        out_group = run_tri_mul(module, z_group, mask_group, args)
        chunks.extend(list(out_group))
        output_lengths.extend(group_lengths)
    return flatten_valid(chunks, output_lengths)


def cuda_time(fn: TensorFn, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        end.synchronize()
    return out, start.elapsed_time(end) / iters


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_multiplication_ragged_hotspot requires CUDA")
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
                    module, z, mask, args.lengths, args
                ),
                "split_buckets": lambda module=module: run_grouped(
                    module,
                    z,
                    args.lengths,
                    groups_by_name["split_buckets"],
                    args,
                ),
                "exact_length_groups": lambda module=module: run_grouped(
                    module,
                    z,
                    args.lengths,
                    groups_by_name["exact_length_groups"],
                    args,
                ),
            }
            reference, padded_ms = cuda_time(
                variants["padded_all"],
                args.warmup,
                args.iters,
            )
            direction_results: dict[str, object] = {
                "padded_all": {"ms": padded_ms, "speedup_vs_padded": 1.0}
            }
            for name in ("split_buckets", "exact_length_groups"):
                output, ms = cuda_time(variants[name], args.warmup, args.iters)
                direction_results[name] = {
                    "ms": ms,
                    "speedup_vs_padded": padded_ms / ms,
                    "valid_vs_padded": compare(reference, output),
                }
            results[direction] = direction_results

    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "fixed_shape": {"c_z": C_Z, "dtype": str(DTYPE)},
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
