#!/usr/bin/env python3
"""CUTLASS3 screen for producer-owned exact-length triangle groups.

The record-L screen launched one rank-4 SM90 GEMM per sequence record.  It
removed the old 4096-problem grouped-GEMM overhead, but sixteen record launches
were still not good enough.  This screen tests the next fusion boundary:

    if the fused input producer writes records with the same exact length into
    a record-major layout, can one rank-4 GEMM per exact-length group win?

The benchmark pre-materializes that producer-owned layout as ``[R * D, N, N]``
for each exact length.  Pack time is intentionally not included in the main
candidate row; a production implementation would have the LayerNorm/dual-GEMM
producer write this layout directly.  If this contraction-only screen loses,
another stock CUTLASS adapter is not the right v2 path.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from scripts.perf.transition_output_epilogue_cutlass_candidate import (
    _header_include_paths,
)
from scripts.perf.triangle_contraction_cutlass3_grouped_probe import (
    DTYPE,
    LONG_LENGTHS,
    SHORT_LENGTHS,
    cuda_time,
    efficiency,
    make_operands,
    parse_ints,
    tflops,
    torch_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT = None
_CUTLASS_MULTIPLE = 8


@dataclass(frozen=True)
class ExactGroup:
    length: int
    indices: tuple[int, ...]
    lhs: torch.Tensor
    rhs: torch.Tensor


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    elif "9.0a" not in arch_list and "90a" not in arch_list:
        raise RuntimeError("CUTLASS GMMA probes require TORCH_CUDA_ARCH_LIST=9.0a")

    cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
    if not cutlass_include:
        raise RuntimeError("CUTLASS_INCLUDE_DIR must point at CUTLASS/CuTe headers")

    _EXT = load(
        name="protenix_triangle_contraction_cutlass3_exact_group_lbatched_probe",
        sources=[
            str(
                REPO_ROOT
                / "scripts/perf/triangle_contraction_cutlass3_exact_group_lbatched_probe.cu"
            )
        ],
        extra_include_paths=_header_include_paths(cutlass_include),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def pad_length(length: int) -> int:
    return ((length + _CUTLASS_MULTIPLE - 1) // _CUTLASS_MULTIPLE) * _CUTLASS_MULTIPLE


def group_tensor(source: torch.Tensor, indices: list[int], length: int) -> torch.Tensor:
    """Return exact group storage as [records * features, Npad, Npad]."""
    features = source.shape[0]
    n_pad = pad_length(length)
    grouped = source.new_zeros(len(indices), features, n_pad, n_pad)
    # source is [D, B, Nmax, Nmax]; producer-owned layout is [B_group, D, N, N].
    grouped[:, :, :length, :length] = source[:, indices, :length, :length].permute(
        1, 0, 2, 3
    )
    return grouped.reshape(len(indices) * features, n_pad, n_pad).contiguous()


def make_exact_groups(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lengths: list[int],
) -> list[ExactGroup]:
    groups: list[ExactGroup] = []
    for length in sorted(set(lengths)):
        indices = [index for index, item in enumerate(lengths) if item == length]
        groups.append(
            ExactGroup(
                length=length,
                indices=tuple(indices),
                lhs=group_tensor(lhs, indices, length),
                rhs=group_tensor(rhs, indices, length),
            )
        )
    return groups


def cutlass_group_contract(groups: list[ExactGroup], direction: str) -> list[torch.Tensor]:
    outgoing = direction == "outgoing"
    return [
        extension().forward(group.lhs, group.rhs, group.length, outgoing)
        for group in groups
    ]


def torch_group_contract(groups: list[ExactGroup], direction: str) -> list[torch.Tensor]:
    if direction == "outgoing":
        return [
            torch.bmm(group.lhs, group.rhs.transpose(1, 2))
            for group in groups
        ]
    return [
        torch.bmm(group.lhs.transpose(1, 2), group.rhs)
        for group in groups
    ]


def group_reference(
    ref: torch.Tensor,
    group: ExactGroup,
) -> torch.Tensor:
    length = group.length
    features = ref.shape[0]
    indices = list(group.indices)
    return (
        ref[:, indices, :length, :length]
        .permute(1, 0, 2, 3)
        .reshape(len(indices) * features, length, length)
    )


def group_error(
    ref: torch.Tensor,
    groups: list[ExactGroup],
    outputs: list[torch.Tensor],
) -> dict[str, float]:
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    for group, output in zip(groups, outputs, strict=True):
        length = group.length
        diff = (
            group_reference(ref, group).float()
            - output[:, :length, :length].float()
        ).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        total_abs += float(diff.sum().item())
        count += diff.numel()
    return {"max_abs": max_abs, "mean_abs": total_abs / max(count, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
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
    lhs, rhs = make_operands(lengths, features=args.features, seed=args.seed)
    groups = make_exact_groups(lhs, rhs, lengths)
    n_max = max(lengths)
    dense_work = len(lengths) * n_max**3
    exact_work = sum(length**3 for length in lengths)

    ref, torch_ms = cuda_time(
        lambda: torch_contract(lhs, rhs, args.direction),
        warmup=args.warmup,
        iters=args.iters,
    )
    torch_groups, torch_group_ms = cuda_time(
        lambda: torch_group_contract(groups, args.direction),
        warmup=args.warmup,
        iters=args.iters,
    )
    cutlass_groups, cutlass_ms = cuda_time(
        lambda: cutlass_group_contract(groups, args.direction),
        warmup=args.warmup,
        iters=args.iters,
    )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "features": args.features,
            "batch": len(lengths),
            "n_max": n_max,
            "num_exact_length_groups": len(groups),
            "num_cutlass_launches": len(groups),
            "lengths": lengths,
            "group_sizes": {str(group.length): len(group.indices) for group in groups},
            **efficiency(lengths),
        },
        "dense_torch_einsum": {
            "ms": torch_ms,
            "logical_tflops_dense": tflops(args.features, dense_work, torch_ms),
        },
        "exact_group_torch_bmm": {
            "ms": torch_group_ms,
            "speedup_vs_dense_torch": torch_ms / torch_group_ms,
            "logical_tflops_exact": tflops(args.features, exact_work, torch_group_ms),
            "parity": group_error(ref, groups, torch_groups),
        },
        "exact_group_lbatched_cutlass3": {
            "ms": cutlass_ms,
            "speedup_vs_dense_torch": torch_ms / cutlass_ms,
            "speedup_vs_exact_group_torch": torch_group_ms / cutlass_ms,
            "logical_tflops_exact": tflops(args.features, exact_work, cutlass_ms),
            "parity": group_error(ref, groups, cutlass_groups),
            "candidate_assumption": "input producer writes [records_in_group * c_z, N, N] directly",
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
