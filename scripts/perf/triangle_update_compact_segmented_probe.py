#!/usr/bin/env python3
"""Screen a compact segmented v2 triangle-multiplication update boundary.

Earlier probes established two constraints:

* splitting a batch into many smaller CUEQ calls loses to launch fragmentation;
* compacting only the row-wise work loses if we scatter back to dense before the
  triangular contraction.

This benchmark tests the next larger boundary.  It keeps Protenix's public ABI
(`z` is still dense padded `[B, Nmax, Nmax, 256]`), but internally uses compact
valid pair rows:

1. read valid dense rows and run input LayerNorm + dual gated projection;
2. run the triangular contraction directly in compact per-record storage;
3. run output LayerNorm/projection/gate on compact rows and scatter valid rows
   back to dense output while adding the residual.

It is intentionally a screen, not production code.  Invalid output rows are not
materialized because Pairformer masks them; parity is measured on valid rows.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

try:
    import triton
    import triton.language as tl
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from cuequivariance_ops_torch.gated_gemm_torch import (
        fused_sigmoid_gated_dual_gemm_dual_x,
    )
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None
    layer_norm_transpose = None
    fused_sigmoid_gated_dual_gemm_dual_x = None


DTYPE = torch.bfloat16
C_Z = 256
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]


@dataclass(frozen=True)
class ContractConfig:
    block_m: int
    block_n: int
    block_k: int
    block_d: int
    num_warps: int
    num_stages: int

    @classmethod
    def parse(cls, text: str) -> "ContractConfig":
        try:
            bm, bn, bk, bd, nw, ns = (int(item) for item in text.split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "configs must look like BMxBNxBKxBDxWARPSxSTAGES"
            ) from exc
        if min(bm, bn, bk, bd, nw, ns) <= 0:
            raise argparse.ArgumentTypeError("config values must be positive")
        return cls(bm, bn, bk, bd, nw, ns)

    def label(self) -> str:
        return (
            f"m{self.block_m}_n{self.block_n}_k{self.block_k}_"
            f"d{self.block_d}_w{self.num_warps}_s{self.num_stages}"
        )


if triton is not None and tl is not None:

    @triton.jit
    def _compact_row_stats_kernel(
        z,
        dense_offsets,
        mean,
        rstd,
        rows: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, 256)
        dense_row = tl.load(dense_offsets + row).to(tl.int64)
        mask = offs < channels
        values = tl.load(z + dense_row * channels + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        mu = tl.sum(values, axis=0) / channels
        centered = tl.where(mask, values - mu, 0.0)
        var = tl.sum(centered * centered, axis=0) / channels
        tl.store(mean + row, mu)
        tl.store(rstd + row, tl.rsqrt(var + eps))

    @triton.jit
    def _compact_ln_dual_gemm_kernel(
        z,
        dense_offsets,
        mean,
        rstd,
        norm_weight,
        norm_bias,
        gate_weight,
        proj_weight,
        ab,
        x_norm,
        rows,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
        offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
        offs_k = tl.arange(0, block_k).to(tl.int64)
        dense_rows = tl.load(dense_offsets + offs_m, mask=offs_m < rows, other=0).to(
            tl.int64
        )
        row_mean = tl.load(mean + offs_m, mask=offs_m < rows, other=0.0)[:, None]
        row_rstd = tl.load(rstd + offs_m, mask=offs_m < rows, other=1.0)[:, None]
        acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
        acc_proj = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k0 in range(0, channels, block_k):
            k = k0 + offs_k
            k_mask = k < channels
            values = tl.load(
                z + dense_rows[:, None] * channels + k[None, :],
                mask=(offs_m[:, None] < rows) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
            bias = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
            values = (values - row_mean) * row_rstd * weight[None, :] + bias[None, :]
            values = values.to(tl.bfloat16)
            tl.store(
                x_norm + offs_m[:, None] * channels + k[None, :],
                values,
                mask=(pid_n == 0) & (offs_m[:, None] < rows) & k_mask[None, :],
            )
            gate_w = tl.load(
                gate_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < out_channels) & k_mask[None, :],
                other=0.0,
            )
            proj_w = tl.load(
                proj_weight + offs_n[:, None] * channels + k[None, :],
                mask=(offs_n[:, None] < out_channels) & k_mask[None, :],
                other=0.0,
            )
            acc_gate += tl.dot(values, tl.trans(gate_w))
            acc_proj += tl.dot(values, tl.trans(proj_w))
        result = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj
        tl.store(
            ab + offs_n[None, :] * rows + offs_m[:, None],
            result,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_channels),
        )

    @triton.jit
    def _compact_contract_kernel(
        ab,
        descriptors,
        update,
        rows: tl.constexpr,
        features: tl.constexpr,
        max_n: tl.constexpr,
        direction: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        desc_id = tl.program_id(0)
        d_block = tl.program_id(1)
        record_offset = tl.load(descriptors + desc_id * 4 + 0).to(tl.int64)
        length = tl.load(descriptors + desc_id * 4 + 1).to(tl.int64)
        tile_i = tl.load(descriptors + desc_id * 4 + 2).to(tl.int64)
        tile_j = tl.load(descriptors + desc_id * 4 + 3).to(tl.int64)
        offs_i = tile_i * block_m + tl.arange(0, block_m)
        offs_j = tile_j * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        valid_ij = (offs_i[:, None] < length) & (offs_j[None, :] < length)
        out_rows = record_offset + offs_i[:, None] * length + offs_j[None, :]

        for local_d in tl.static_range(0, block_d):
            d = d_block * block_d + local_d
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k0 in range(0, max_n, block_k):
                k = k0 + offs_k
                valid_k = k < length
                if direction == 0:
                    a_rows = record_offset + offs_i[:, None] * length + k[None, :]
                    b_rows = record_offset + offs_j[:, None] * length + k[None, :]
                    a = tl.load(
                        ab + d * rows + a_rows,
                        mask=(d < features) & (offs_i[:, None] < length) & valid_k[None, :],
                        other=0.0,
                    )
                    b = tl.load(
                        ab + (features + d) * rows + b_rows,
                        mask=(d < features) & (offs_j[:, None] < length) & valid_k[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(a, tl.trans(b))
                else:
                    a_rows = record_offset + k[None, :] * length + offs_i[:, None]
                    b_rows = record_offset + k[:, None] * length + offs_j[None, :]
                    a = tl.load(
                        ab + d * rows + a_rows,
                        mask=(d < features) & valid_k[None, :] & (offs_i[:, None] < length),
                        other=0.0,
                    )
                    b = tl.load(
                        ab + (features + d) * rows + b_rows,
                        mask=(d < features) & valid_k[:, None] & (offs_j[None, :] < length),
                        other=0.0,
                    )
                    acc += tl.dot(a, b)
            tl.store(
                update + out_rows * features + d,
                acc,
                mask=(d < features) & valid_ij,
            )

    @triton.jit
    def _scatter_residual_kernel(
        compact_delta,
        z,
        dense_offsets,
        out,
        rows,
        channels: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        c = tl.program_id(1) * block_c + tl.arange(0, block_c)
        dense_row = tl.load(dense_offsets + row).to(tl.int64)
        mask = (row < rows) & (c < channels)
        delta = tl.load(compact_delta + row * channels + c, mask=mask, other=0.0)
        residual = tl.load(z + dense_row * channels + c, mask=mask, other=0.0)
        tl.store(out + dense_row * channels + c, delta + residual, mask=mask)


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_configs(text: str) -> list[ContractConfig]:
    return [ContractConfig.parse(item.strip()) for item in text.split(",") if item.strip()]


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def make_modules(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    modules = {
        "outgoing": TriangleMultiplicationOutgoing(C_Z, C_Z).cuda().eval(),
        "incoming": TriangleMultiplicationIncoming(C_Z, C_Z).cuda().eval(),
    }
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1729)
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return modules


def make_inputs(lengths: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(lengths)
    z = torch.zeros(len(lengths), n_max, n_max, C_Z, device="cuda", dtype=DTYPE)
    mask = torch.zeros(len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def make_metadata(lengths: list[int], config: ContractConfig) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(lengths)
    dense_offsets: list[int] = []
    descriptors: list[tuple[int, int, int, int]] = []
    compact_offset = 0
    for batch_index, length in enumerate(lengths):
        for i in range(length):
            base = batch_index * n_max * n_max + i * n_max
            dense_offsets.extend(base + j for j in range(length))
        tiles_i = (length + config.block_m - 1) // config.block_m
        tiles_j = (length + config.block_n - 1) // config.block_n
        descriptors.extend(
            (compact_offset, length, tile_i, tile_j)
            for tile_i in range(tiles_i)
            for tile_j in range(tiles_j)
        )
        compact_offset += length * length
    return (
        torch.tensor(dense_offsets, device="cuda", dtype=torch.int64),
        torch.tensor(descriptors, device="cuda", dtype=torch.int64),
    )


def cached_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        "p_in": torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0)
        .to(device="cuda", dtype=DTYPE)
        .contiguous(),
        "g_in": torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0)
        .to(device="cuda", dtype=DTYPE)
        .contiguous(),
        "p_out": module.linear_z.weight.to(device="cuda", dtype=DTYPE).contiguous(),
        "g_out": module.linear_g.weight.to(device="cuda", dtype=DTYPE).contiguous(),
    }


def compact_segmented_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    dense_offsets: torch.Tensor,
    descriptors: torch.Tensor,
    config: ContractConfig,
    direction: str,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    rows = dense_offsets.numel()
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    ab = torch.empty((2 * C_Z, rows), device=z.device, dtype=z.dtype)
    x_norm = torch.empty((rows, C_Z), device=z.device, dtype=z.dtype)
    _compact_row_stats_kernel[(rows,)](z, dense_offsets, mean, rstd, rows, C_Z, 1e-5)
    grid = (triton.cdiv(rows, 64), triton.cdiv(2 * C_Z, 128))
    _compact_ln_dual_gemm_kernel[grid](
        z,
        dense_offsets,
        mean,
        rstd,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        weights["g_in"],
        weights["p_in"],
        ab,
        x_norm,
        rows,
        C_Z,
        2 * C_Z,
        64,
        128,
        64,
        num_warps=4,
        num_stages=3,
    )

    update = torch.empty((rows, C_Z), device=z.device, dtype=z.dtype)
    contract_grid = (descriptors.shape[0], triton.cdiv(C_Z, config.block_d))
    _compact_contract_kernel[contract_grid](
        ab,
        descriptors,
        update,
        rows,
        C_Z,
        z.shape[1],
        0 if direction == "outgoing" else 1,
        config.block_m,
        config.block_n,
        config.block_k,
        config.block_d,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    update_norm = layer_norm_transpose(
        update,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        eps=1e-5,
        layout="nd->nd",
    )
    compact_delta = fused_sigmoid_gated_dual_gemm_dual_x(
        x_norm,
        update_norm,
        weights["g_out"],
        weights["p_out"],
    )
    out = torch.empty_like(z)
    scatter_grid = (rows, triton.cdiv(C_Z, 64))
    _scatter_residual_kernel[scatter_grid](compact_delta, z, dense_offsets, out, rows, C_Z, 64)
    return out


def baseline_update(module: torch.nn.Module, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return module(
        z.clone(),
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )


def valid_from_dense(z: torch.Tensor, dense_offsets: torch.Tensor) -> torch.Tensor:
    return z.reshape(-1, z.shape[-1]).index_select(0, dense_offsets)


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def cuda_time(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=DTYPE):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="both")
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=parse_configs("16x16x32x1x4x3,32x32x64x1x4x3,32x32x64x2x4x3"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if triton is None or tl is None or layer_norm_transpose is None:
        raise RuntimeError("Triton and CUEQ torch ops are required")
    if fused_sigmoid_gated_dual_gemm_dual_x is None:
        raise RuntimeError("CUEQ dual-input gated GEMM is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    lengths = lengths_from_args(args)
    modules = make_modules(args)
    z, mask = make_inputs(lengths, args.seed)
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
        direction_results: dict[str, Any] = {
            "baseline_cueq": {"ms": base_ms},
            "compact_segmented": {},
        }
        for config in args.configs:
            dense_offsets, descriptors = make_metadata(lengths, config)
            candidate, ms = cuda_time(
                lambda config=config, dense_offsets=dense_offsets, descriptors=descriptors: compact_segmented_update(
                    module,
                    z,
                    dense_offsets,
                    descriptors,
                    config,
                    direction,
                    weights,
                ),
                args.warmup,
                args.iters,
            )
            direction_results["compact_segmented"][config.label()] = {
                "config": vars(config),
                "descriptor_count": int(descriptors.shape[0]),
                "valid_rows": int(dense_offsets.numel()),
                "ms": ms,
                "speedup_vs_baseline": base_ms / ms,
                "parity_valid": compare(
                    valid_from_dense(base_out, dense_offsets),
                    valid_from_dense(candidate, dense_offsets),
                ),
            }
        results[direction] = direction_results

    n_max = max(lengths)
    payload = {
        "args": {**vars(args), "configs": [config.label() for config in args.configs], "lengths": lengths},
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": len(lengths),
            "n_max": n_max,
            "c_z": C_Z,
            "valid_pair_efficiency": sum(length * length for length in lengths) / (len(lengths) * n_max * n_max),
            "valid_cubic_efficiency": sum(length**3 for length in lengths) / (len(lengths) * n_max**3),
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
