#!/usr/bin/env python3
"""Descriptor-scheduled Triton screen for ragged triangle contraction.

The first Triton segmented contraction probe used a rectangular `B * Nmax`
tile grid and masked work outside each record's valid length.  That avoided
host-built descriptors, but it also meant short records still launched invalid
tiles.  This probe tests the next scheduler shape: build a compact list of
valid `(batch, tile_i, tile_j)` tiles once, then run one Triton launch over that
list.

This is still only a contraction-core screen.  A production Protenix-v2 win
would need this idea fused into the larger triangle-multiplication update, so
LayerNorm, gated projections, contraction, output normalization, and residual
store stay in one native boundary.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None

from scripts.perf.triangle_contraction_triton_segmented_probe import (
    KernelConfig,
    cuda_time,
    efficiency,
    invalid_abs_sum,
    make_operands,
    parse_configs,
    parse_ints,
    tflops,
    torch_contract,
    valid_error,
)


SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]


if triton is not None and tl is not None:

    @triton.jit
    def _descriptor_triangle_contract_kernel(
        lhs,
        rhs,
        descriptors,
        out,
        features: tl.constexpr,
        batch: tl.constexpr,
        max_n: tl.constexpr,
        direction: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        desc_id = tl.program_id(0)
        d_block = tl.program_id(1)
        b = tl.load(descriptors + desc_id * 4 + 0).to(tl.int64)
        tile_i = tl.load(descriptors + desc_id * 4 + 1).to(tl.int64)
        tile_j = tl.load(descriptors + desc_id * 4 + 2).to(tl.int64)

        offs_i = tile_i * block_m + tl.arange(0, block_m)
        offs_j = tile_j * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        # Descriptors include only tiles whose origin is inside the valid
        # square.  Boundary masks still handle partial tiles at the record end.
        length = tl.load(descriptors + desc_id * 4 + 3).to(tl.int64)
        valid_ij = (offs_i[:, None] < length) & (offs_j[None, :] < length)

        for local_d in tl.static_range(0, block_d):
            d = d_block * block_d + local_d
            base = ((d * batch + b) * max_n) * max_n
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k0 in range(0, max_n, block_k):
                k = k0 + offs_k
                valid_k = k < length
                if direction == 0:
                    a = tl.load(
                        lhs + base + offs_i[:, None] * max_n + k[None, :],
                        mask=(d < features) & (offs_i[:, None] < length) & valid_k[None, :],
                        other=0.0,
                    )
                    b_tile = tl.load(
                        rhs + base + offs_j[:, None] * max_n + k[None, :],
                        mask=(d < features) & (offs_j[:, None] < length) & valid_k[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(a, tl.trans(b_tile))
                else:
                    a = tl.load(
                        lhs + base + k[None, :] * max_n + offs_i[:, None],
                        mask=(d < features) & valid_k[None, :] & (offs_i[:, None] < length),
                        other=0.0,
                    )
                    b_tile = tl.load(
                        rhs + base + k[:, None] * max_n + offs_j[None, :],
                        mask=(d < features) & valid_k[:, None] & (offs_j[None, :] < length),
                        other=0.0,
                    )
                    acc += tl.dot(a, b_tile)

            tl.store(
                out + base + offs_i[:, None] * max_n + offs_j[None, :],
                tl.where(valid_ij, acc, 0.0),
                mask=(d < features) & valid_ij,
            )


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def make_descriptors(lengths: list[int], config: KernelConfig) -> torch.Tensor:
    rows: list[tuple[int, int, int, int]] = []
    for batch_index, length in enumerate(lengths):
        tiles_m = (length + config.block_m - 1) // config.block_m
        tiles_n = (length + config.block_n - 1) // config.block_n
        for tile_i in range(tiles_m):
            for tile_j in range(tiles_n):
                rows.append((batch_index, tile_i, tile_j, length))
    return torch.tensor(rows, device="cuda", dtype=torch.int32)


def descriptor_contract(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    descriptors: torch.Tensor,
    direction: str,
    config: KernelConfig,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required")
    features, batch, max_n, _ = lhs.shape
    out = torch.zeros_like(lhs)
    grid = (descriptors.shape[0], triton.cdiv(features, config.block_d))
    _descriptor_triangle_contract_kernel[grid](
        lhs,
        rhs,
        descriptors,
        out,
        features,
        batch,
        max_n,
        0 if direction == "outgoing" else 1,
        config.block_m,
        config.block_n,
        config.block_k,
        config.block_d,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out


def tile_stats(lengths: list[int], config: KernelConfig) -> dict[str, Any]:
    max_n = max(lengths)
    rectangular = len(lengths) * ((max_n + config.block_m - 1) // config.block_m) * (
        (max_n + config.block_n - 1) // config.block_n
    )
    compact = sum(
        ((length + config.block_m - 1) // config.block_m)
        * ((length + config.block_n - 1) // config.block_n)
        for length in lengths
    )
    return {
        "rectangular_tiles": rectangular,
        "descriptor_tiles": compact,
        "tile_efficiency": compact / rectangular,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=parse_configs(
            "16x16x32x1x4x3,32x32x64x1x4x3,32x32x64x2x4x3,"
            "32x32x64x4x4x3,64x32x64x1x4x3,32x64x64x1x4x3"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if triton is None or tl is None:
        raise RuntimeError("triangle_contraction_triton_descriptors_probe requires Triton")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True

    lengths = lengths_from_args(args)
    lhs, rhs = make_operands(lengths, features=args.features, seed=args.seed)
    max_n = max(lengths)
    dense_work = len(lengths) * max_n**3
    exact_work = sum(length**3 for length in lengths)
    timing = {"warmup": args.warmup, "iters": args.iters}

    ref, torch_ms = cuda_time(lambda: torch_contract(lhs, rhs, args.direction), **timing)
    candidates = {}
    for config in args.configs:
        descriptors = make_descriptors(lengths, config)
        row = {"config": vars(config), "tiles": tile_stats(lengths, config)}
        try:
            candidate, ms = cuda_time(
                lambda config=config, descriptors=descriptors: descriptor_contract(
                    lhs, rhs, descriptors, args.direction, config
                ),
                **timing,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path.
            row["error"] = repr(exc)
        else:
            row.update(
                {
                    "ms": ms,
                    "logical_tflops_exact": tflops(args.features, exact_work, ms),
                    "speedup_vs_dense_torch": torch_ms / ms,
                    "parity": valid_error(ref, candidate, lengths),
                    "invalid_abs_sum": invalid_abs_sum(candidate, lengths),
                }
            )
        candidates[config.label()] = row

    result = {
        "args": {**vars(args), "configs": [config.label() for config in args.configs], "lengths": lengths},
        "device": torch.cuda.get_device_name(),
        "shape": {
            "features": args.features,
            "batch": len(lengths),
            "n_max": max_n,
            **efficiency(lengths),
        },
        "dense_torch_einsum": {
            "ms": torch_ms,
            "logical_tflops_dense": tflops(args.features, dense_work, torch_ms),
        },
        "triton_descriptor_segmented": candidates,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
