#!/usr/bin/env python3
"""Lower-bound screen for ragged pairformer execution.

Mixed-token campaign batches are length-sorted, then padded to the longest
record in each B16 group.  That keeps diffusion/atom work well batched, but the
pairformer still runs triangle and attention kernels on padded token squares.

This script asks whether the padding waste is large enough to justify a real
segmented pairformer kernel.  It compares the current one-padded-block path
with idealized "islands" that run exact or smaller padded token groups using
the same PairformerBlock weights.  The islands are not a production path: they
pay extra launches and do not preserve cross-block layout.  They are a
decisive screen for headroom.  If even this lower bound is flat, a larger CuTe
rewrite is unlikely to recover the missing 10x.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterable

import torch

from protenix.model.modules.pairformer import PairformerBlock


TensorPair = tuple[torch.Tensor, torch.Tensor]
VariantFn = Callable[[], TensorPair]


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


@contextmanager
def cuda_autocast(dtype_name: str):
    if dtype_name == "float32":
        with nullcontext():
            yield
    else:
        with torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name)):
            yield


def make_block(args: argparse.Namespace) -> PairformerBlock:
    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 10007)
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_padded_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    max_tokens = max(lengths)
    batch = len(lengths)
    s = torch.zeros(batch, max_tokens, args.c_s, device="cuda", dtype=dtype)
    z = torch.zeros(batch, max_tokens, max_tokens, args.c_z, device="cuda", dtype=dtype)
    mask = torch.zeros(batch, max_tokens, max_tokens, device="cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for index, length in enumerate(lengths):
        s[index, :length] = torch.randn(
            length, args.c_s, device="cuda", dtype=dtype, generator=generator
        )
        z[index, :length, :length] = torch.randn(
            length, length, args.c_z, device="cuda", dtype=dtype, generator=generator
        )
        mask[index, :length, :length] = 1
    return s, z, mask


def run_block(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> TensorPair:
    return block(
        s.clone(),
        z.clone(),
        pair_mask=mask,
        triangle_multiplicative=args.triangle_multiplicative,
        triangle_attention=args.triangle_attention,
        inplace_safe=True,
        chunk_size=args.chunk_size,
    )


def case_from_indices(
    s: torch.Tensor,
    z: torch.Tensor,
    lengths: list[int],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    dtype = s.dtype
    max_tokens = max(lengths[index] for index in indices)
    s_group = torch.zeros(len(indices), max_tokens, s.shape[-1], device=s.device, dtype=dtype)
    z_group = torch.zeros(
        len(indices), max_tokens, max_tokens, z.shape[-1], device=z.device, dtype=dtype
    )
    mask_group = torch.zeros(len(indices), max_tokens, max_tokens, device=z.device, dtype=dtype)
    group_lengths = []
    for out_index, source_index in enumerate(indices):
        length = lengths[source_index]
        group_lengths.append(length)
        s_group[out_index, :length] = s[source_index, :length]
        z_group[out_index, :length, :length] = z[source_index, :length, :length]
        mask_group[out_index, :length, :length] = 1
    return s_group, z_group, mask_group, group_lengths


def exact_length_groups(lengths: list[int]) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        groups[length].append(index)
    return [groups[length] for length in sorted(groups)]


def split_bucket_groups(lengths: list[int], limits: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    start = 0
    sorted_limits = sorted(limits)
    for limit in sorted_limits:
        group = [index for index, length in enumerate(lengths) if start < length <= limit]
        if group:
            groups.append(group)
        start = limit
    tail = [index for index, length in enumerate(lengths) if length > start]
    if tail:
        groups.append(tail)
    return groups


def singleton_groups(lengths: list[int]) -> list[list[int]]:
    return [[index] for index in range(len(lengths))]


def flatten_valid(tensors: Iterable[torch.Tensor], lengths: Iterable[int], pair: bool) -> torch.Tensor:
    chunks = []
    for tensor, length in zip(tensors, lengths, strict=True):
        if pair:
            chunks.append(tensor[:length, :length].reshape(-1, tensor.shape[-1]))
        else:
            chunks.append(tensor[:length])
    return torch.cat(chunks, dim=0)


def run_grouped(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    lengths: list[int],
    groups: list[list[int]],
    args: argparse.Namespace,
) -> TensorPair:
    s_chunks = []
    z_chunks = []
    output_lengths = []
    for indices in groups:
        s_group, z_group, mask_group, group_lengths = case_from_indices(s, z, lengths, indices)
        s_out, z_out = run_block(block, s_group, z_group, mask_group, args)
        s_chunks.extend(s_out)
        z_chunks.extend(z_out)
        output_lengths.extend(group_lengths)
    return (
        flatten_valid(s_chunks, output_lengths, pair=False),
        flatten_valid(z_chunks, output_lengths, pair=True),
    )


def run_padded_valid(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    args: argparse.Namespace,
) -> TensorPair:
    s_out, z_out = run_block(block, s, z, mask, args)
    return (
        flatten_valid(s_out, lengths, pair=False),
        flatten_valid(z_out, lengths, pair=True),
    )


def cuda_time(fn: VariantFn, warmup: int, iters: int) -> tuple[TensorPair, float]:
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


def group_stats(groups: list[list[int]], lengths: list[int]) -> dict[str, object]:
    padded_pairs = 0
    for group in groups:
        max_tokens = max(lengths[index] for index in group)
        padded_pairs += len(group) * max_tokens * max_tokens
    valid_pairs = sum(length * length for length in lengths)
    return {
        "groups": [[lengths[index] for index in group] for group in groups],
        "num_groups": len(groups),
        "padded_pairs": padded_pairs,
        "valid_pairs": valid_pairs,
        "valid_pair_efficiency": valid_pairs / padded_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
    )
    parser.add_argument("--bucket-limits", type=parse_ints, default=parse_ints("76"))
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_ragged_islands_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    block = make_block(args)
    s, z, mask = make_padded_inputs(args.lengths, args)

    groups_by_name = {
        "padded_all": [list(range(len(args.lengths)))],
        "split_buckets": split_bucket_groups(args.lengths, args.bucket_limits),
        "exact_length_groups": exact_length_groups(args.lengths),
        "singletons": singleton_groups(args.lengths),
    }
    variants: dict[str, VariantFn] = {
        "padded_all": lambda: run_padded_valid(block, s, z, mask, args.lengths, args),
        **{
            name: (lambda groups=groups: run_grouped(block, s, z, args.lengths, groups, args))
            for name, groups in groups_by_name.items()
            if name != "padded_all"
        },
    }

    results = {}
    with cuda_autocast(args.compute_dtype):
        reference, padded_ms = cuda_time(variants["padded_all"], args.warmup, args.iters)
        results["padded_all"] = {"ms": padded_ms, "speedup_vs_padded": 1.0}
        for name, fn in variants.items():
            if name == "padded_all":
                continue
            output, ms = cuda_time(fn, args.warmup, args.iters)
            results[name] = {
                "ms": ms,
                "speedup_vs_padded": padded_ms / ms,
                "s_vs_padded_valid": compare(reference[0], output[0]),
                "z_vs_padded_valid": compare(reference[1], output[1]),
            }

    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
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
