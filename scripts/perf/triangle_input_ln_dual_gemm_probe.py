#!/usr/bin/env python3
"""Probe a fused LayerNorm + dual-GEMM producer for v2 triangle multiplication.

CUEQ triangle multiplication is fast because its dense contraction and gated
GEMMs are good kernels.  The remaining cost around the input producer is still
not free: CUEQ first normalizes the full padded pair tensor, writes that
normalized tensor, then the fused gated dual-GEMM reads it back and writes the
`[2D, B, N, N]` projection layout consumed by the triangular contraction.

This benchmark tests a larger fusion boundary without changing the contraction:

1. compute one mean/rstd pair per `[B, N, N]` row;
2. run a Triton dual-GEMM that applies LayerNorm inside the dot-product loop;
3. write the exact same transposed CUEQ projection layout.

It is intentionally a screen, not production code.  A candidate must beat the
CUEQ producer here before it is worth wiring into the full triangle update.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl

from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

try:
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm
except ImportError:  # pragma: no cover - optional GPU dependency.
    layer_norm_transpose = None
    fused_sigmoid_gated_dual_gemm = None


Config = tuple[int, int, int, int, int]
TensorFn = Callable[[], torch.Tensor]
DTYPE = torch.bfloat16
C_Z = 256
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]
DEFAULT_CONFIGS: list[Config] = [
    (16, 64, 64, 4, 3),
    (16, 128, 64, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 3),
    (32, 64, 128, 4, 3),
    (32, 128, 64, 8, 3),
]


def parse_config(spec: str) -> Config:
    parts = tuple(int(part) for part in spec.split("x"))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("expected BMxBNxBKxWARPSxSTAGES")
    return parts  # type: ignore[return-value]


def config_name(config: Config) -> str:
    bm, bn, bk, nw, ns = config
    return f"m{bm}_n{bn}_k{bk}_w{nw}_s{ns}"


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


@triton.jit
def _row_stats_kernel(x, mean, rstd, rows: tl.constexpr, c: tl.constexpr, eps: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, 256)
    mask = offs < c
    values = tl.load(x + row * c + offs, mask=mask, other=0.0).to(tl.float32)
    mu = tl.sum(values, axis=0) / c
    centered = tl.where(mask, values - mu, 0.0)
    var = tl.sum(centered * centered, axis=0) / c
    tl.store(mean + row, mu)
    tl.store(rstd + row, tl.rsqrt(var + eps))


@triton.jit
def _ln_dual_gemm_transpose_kernel(
    x,
    mean,
    rstd,
    norm_weight,
    norm_bias,
    gate_weight,
    proj_weight,
    mask_ptr,
    out,
    rows,
    channels: tl.constexpr,
    out_channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
    offs_k = tl.arange(0, block_k).to(tl.int64)

    acc_gate = tl.zeros((block_m, block_n), dtype=tl.float32)
    acc_proj = tl.zeros((block_m, block_n), dtype=tl.float32)
    row_mean = tl.load(mean + offs_m, mask=offs_m < rows, other=0.0)[:, None]
    row_rstd = tl.load(rstd + offs_m, mask=offs_m < rows, other=1.0)[:, None]
    for k0 in range(0, channels, block_k):
        k = k0 + offs_k
        k_mask = k < channels
        values = tl.load(
            x + offs_m[:, None] * channels + k[None, :],
            mask=(offs_m[:, None] < rows) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        b = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        values = (values - row_mean) * row_rstd * w[None, :] + b[None, :]
        # CUEQ's producer feeds BF16 tensor-core GEMMs.  Keep the same contract:
        # accumulate normalization in FP32 for stability, then cast before dot.
        values = values.to(tl.bfloat16)
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

    pair_mask = tl.load(mask_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
    result = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj * pair_mask[:, None]
    tl.store(
        out + offs_n[None, :] * rows + offs_m[:, None],
        result,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_channels),
    )


def make_module(args: argparse.Namespace) -> torch.nn.Module:
    cls = TriangleMultiplicationOutgoing if args.direction == "outgoing" else TriangleMultiplicationIncoming
    module = cls(C_Z, C_Z).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1729)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2 and not torch.any(parameter):
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(lengths: list[int], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = max(lengths)
    z = torch.zeros(len(lengths), tokens, tokens, C_Z, device="cuda", dtype=DTYPE)
    mask = torch.zeros(len(lengths), tokens, tokens, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def cueq_producer(module: torch.nn.Module, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x_norm = layer_norm_transpose(
        z,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        eps=1e-5,
        layout="bijd->bijd",
    )
    p_weight = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0)
    g_weight = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0)
    return fused_sigmoid_gated_dual_gemm(
        x_norm,
        g_weight,
        p_weight,
        mask,
        transpose_out=True,
    )


def triton_producer(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    bm, bn, bk, nw, ns = config
    rows = z.numel() // z.shape[-1]
    mean = torch.empty(rows, device=z.device, dtype=torch.float32)
    rstd = torch.empty(rows, device=z.device, dtype=torch.float32)
    out_channels = 2 * z.shape[-1]
    out = torch.empty((out_channels, rows), device=z.device, dtype=z.dtype)
    p_weight = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0).to(
        device=z.device, dtype=z.dtype
    ).contiguous()
    g_weight = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0).to(
        device=z.device, dtype=z.dtype
    ).contiguous()
    _row_stats_kernel[(rows,)](z, mean, rstd, rows, C_Z, 1e-5)
    grid = (triton.cdiv(rows, bm), triton.cdiv(out_channels, bn))
    _ln_dual_gemm_transpose_kernel[grid](
        z,
        mean,
        rstd,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        g_weight,
        p_weight,
        mask,
        out,
        rows,
        C_Z,
        out_channels,
        bm,
        bn,
        bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out.reshape(out_channels, *z.shape[:3])


def time_cuda(fn: TensorFn, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
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


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument("--configs", nargs="*", type=parse_config)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if layer_norm_transpose is None or fused_sigmoid_gated_dual_gemm is None:
        raise RuntimeError("CUEQ torch ops are required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    lengths = lengths_from_args(args)
    module = make_module(args)
    z, mask = make_inputs(lengths, args)
    configs = args.configs or DEFAULT_CONFIGS

    reference, cueq_ms = time_cuda(lambda: cueq_producer(module, z, mask), args.warmup, args.iters)
    rows: dict[str, Any] = {}
    best_name = ""
    best_ms = float("inf")
    for config in configs:
        name = config_name(config)
        try:
            candidate, ms = time_cuda(
                lambda cfg=config: triton_producer(module, z, mask, cfg),
                args.warmup,
                args.iters,
            )
            rows[name] = {
                "ms": ms,
                "speedup_vs_cueq": cueq_ms / ms,
                "parity": diff_stats(reference, candidate),
            }
            if ms < best_ms:
                best_name, best_ms = name, ms
        except Exception as exc:
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}

    result = {
        "args": {**vars(args), "lengths": lengths, "configs": [config_name(c) for c in configs]},
        "shape": {
            "batch": len(lengths),
            "tokens": max(lengths),
            "rows": len(lengths) * max(lengths) * max(lengths),
            "c_z": C_Z,
        },
        "cueq_ms": cueq_ms,
        "candidates": rows,
        "best": (rows[best_name] | {"name": best_name}) if best_name else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        from pathlib import Path

        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
