#!/usr/bin/env python3
"""Triton segmented screen for the v2 triangle contraction core.

The CUTLASS grouped-GEMM probes were useful, but their unit of scheduling was
still one tiny GEMM per ``(feature, record)`` pair.  For Protenix-v2 that means
4096 small problems at ``c_z=256, B=16`` before any useful math starts.  This
probe asks a more targeted question: can a single Triton launch schedule the
same ragged contraction directly from record lengths, without per-GEMM problem
metadata, and recover enough of the padding waste to justify fusing the larger
triangle-multiplication boundary?

This is intentionally still a contraction-core probe, not a production path.
Promotion still requires fusing more of the CUEQ triangle-multiplication update
because the contraction is only one part of the measured v2 triangle-mul time.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None


DTYPE = torch.bfloat16
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    block_d: int
    num_warps: int
    num_stages: int

    @classmethod
    def parse(cls, text: str) -> "KernelConfig":
        try:
            bm, bn, bk, bd, nw, ns = (int(item) for item in text.split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "kernel configs must look like BMxBNxBKxBDxWARPSxSTAGES"
            ) from exc
        if min(bm, bn, bk, bd, nw, ns) <= 0:
            raise argparse.ArgumentTypeError("kernel config values must be positive")
        return cls(bm, bn, bk, bd, nw, ns)

    def label(self) -> str:
        return (
            f"m{self.block_m}_n{self.block_n}_k{self.block_k}_"
            f"d{self.block_d}_w{self.num_warps}_s{self.num_stages}"
        )


if triton is not None and tl is not None:

    @triton.jit
    def _segmented_triangle_contract_kernel(
        lhs,
        rhs,
        lengths,
        out,
        features: tl.constexpr,
        batch: tl.constexpr,
        max_n: tl.constexpr,
        direction: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        block_d: tl.constexpr,
        tiles_n: tl.constexpr,
    ) -> None:
        tile = tl.program_id(0)
        d_block = tl.program_id(1)
        b = tl.program_id(2)

        tile_i = tile // tiles_n
        tile_j = tile - tile_i * tiles_n
        offs_i = tile_i * block_m + tl.arange(0, block_m)
        offs_j = tile_j * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        length = tl.load(lengths + b)

        # Keep the grid rectangular to avoid host-built tile descriptor arrays.
        # Programs outside a record's valid square only zero their tile and
        # return; the real fused kernel would replace this with compact tile
        # descriptors once the math schedule is proven useful.
        valid_ij = (offs_i[:, None] < length) & (offs_j[None, :] < length)
        in_physical_tile = (offs_i[:, None] < max_n) & (offs_j[None, :] < max_n)

        for local_d in tl.static_range(0, block_d):
            d = d_block * block_d + local_d
            base = ((d * batch + b) * max_n) * max_n
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k0 in range(0, max_n, block_k):
                k = k0 + offs_k
                valid_k = k < length
                if direction == 0:  # outgoing: sum_k lhs[i, k] * rhs[j, k]
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
                else:  # incoming: sum_k lhs[k, i] * rhs[k, j]
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

            values = tl.where(valid_ij, acc, 0.0)
            tl.store(
                out + base + offs_i[:, None] * max_n + offs_j[None, :],
                values,
                mask=(d < features) & in_physical_tile,
            )


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_configs(text: str) -> list[KernelConfig]:
    return [KernelConfig.parse(item.strip()) for item in text.split(",") if item.strip()]


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def cuda_time(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def make_operands(lengths: list[int], *, features: int, seed: int):
    n_max = max(lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lhs = torch.zeros(features, len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    rhs = torch.zeros_like(lhs)
    for batch_index, length in enumerate(lengths):
        shape = (features, length, length)
        lhs[:, batch_index, :length, :length] = torch.randn(
            shape, device="cuda", dtype=DTYPE, generator=generator
        )
        rhs[:, batch_index, :length, :length] = torch.randn(
            shape, device="cuda", dtype=DTYPE, generator=generator
        )
    return lhs.contiguous(), rhs.contiguous()


def torch_contract(lhs: torch.Tensor, rhs: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "outgoing":
        return torch.einsum("dbik,dbjk->dbij", lhs, rhs)
    return torch.einsum("dbki,dbkj->dbij", lhs, rhs)


def triton_contract(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lengths: torch.Tensor,
    direction: str,
    config: KernelConfig,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required")
    features, batch, max_n, _ = lhs.shape
    out = torch.empty_like(lhs)
    tiles_m = triton.cdiv(max_n, config.block_m)
    tiles_n = triton.cdiv(max_n, config.block_n)
    grid = (
        tiles_m * tiles_n,
        triton.cdiv(features, config.block_d),
        batch,
    )
    _segmented_triangle_contract_kernel[grid](
        lhs,
        rhs,
        lengths,
        out,
        features,
        batch,
        max_n,
        0 if direction == "outgoing" else 1,
        config.block_m,
        config.block_n,
        config.block_k,
        config.block_d,
        tiles_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out


def valid_error(ref: torch.Tensor, candidate: torch.Tensor, lengths: list[int]) -> dict[str, float]:
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    for batch_index, length in enumerate(lengths):
        diff = (
            ref[:, batch_index, :length, :length].float()
            - candidate[:, batch_index, :length, :length].float()
        ).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        total_abs += float(diff.sum().item())
        count += diff.numel()
    return {"max_abs": max_abs, "mean_abs": total_abs / max(count, 1)}


def invalid_abs_sum(candidate: torch.Tensor, lengths: list[int]) -> float:
    invalid = 0.0
    for batch_index, length in enumerate(lengths):
        invalid += float(candidate[:, batch_index, length:, :].float().abs().sum().item())
        invalid += float(candidate[:, batch_index, :length, length:].float().abs().sum().item())
    return invalid


def efficiency(lengths: list[int]) -> dict[str, float]:
    n_max = max(lengths)
    valid_cubic = sum(length**3 for length in lengths)
    padded_cubic = len(lengths) * n_max**3
    return {
        "valid_cubic_efficiency": valid_cubic / padded_cubic,
        "ideal_contraction_work_speedup": padded_cubic / valid_cubic,
    }


def tflops(features: int, work_units: int, ms: float) -> float:
    return (2 * features * work_units) / (ms * 1e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=parse_configs("16x16x32x1x4x3,32x32x32x1x4x3,32x32x64x1x4x3"),
        help="Comma-separated BMxBNxBKxBDxWARPSxSTAGES Triton configs.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if triton is None or tl is None:
        raise RuntimeError("triangle_contraction_triton_segmented_probe requires Triton")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True

    lengths = lengths_from_args(args)
    lengths_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    lhs, rhs = make_operands(lengths, features=args.features, seed=args.seed)
    n_max = max(lengths)
    dense_work = len(lengths) * n_max**3
    exact_work = sum(length**3 for length in lengths)
    timing = {"warmup": args.warmup, "iters": args.iters}

    ref, torch_ms = cuda_time(lambda: torch_contract(lhs, rhs, args.direction), **timing)
    candidates = {}
    for config in args.configs:
        row = {
            "config": {
                "block_m": config.block_m,
                "block_n": config.block_n,
                "block_k": config.block_k,
                "block_d": config.block_d,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
            },
        }
        try:
            candidate, ms = cuda_time(
                lambda config=config: triton_contract(
                    lhs,
                    rhs,
                    lengths_tensor,
                    args.direction,
                    config,
                ),
                **timing,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path.
            # Triton config exploration should be failure-tolerant: an illegal
            # tile is evidence, not a reason to lose all other timings from the
            # same queued H100 job.
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
        "args": {
            **vars(args),
            "configs": [config.label() for config in args.configs],
            "lengths": lengths,
        },
        "device": torch.cuda.get_device_name(),
        "shape": {
            "features": args.features,
            "batch": len(lengths),
            "n_max": n_max,
            "lengths": lengths,
            **efficiency(lengths),
        },
        "dense_torch_einsum": {
            "ms": torch_ms,
            "logical_tflops_dense": tflops(args.features, dense_work, torch_ms),
        },
        "triton_segmented": candidates,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
