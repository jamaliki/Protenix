#!/usr/bin/env python3
"""Screen a larger triangle-multiplication output fusion boundary.

The existing CUEQ triangle-multiplication path is a good dense mainloop, so this
probe deliberately leaves the triangular contraction alone.  It only tests the
next consumer boundary:

    layer_norm_transpose(update_dbij)
    fused_sigmoid_gated_dual_gemm_dual_x(x_norm, update_norm)
    residual add

against a Triton candidate that computes the output LayerNorm row statistics,
normalizes the `D,B,N,N` update tile inside the value GEMM, computes the gate
GEMM from the already-normalized input, and adds the residual in the final
store.  If this cannot beat CUEQ plus the explicit norm/layout pass in a hotspot
screen, it is not worth wiring into the model path.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import (
    fused_sigmoid_gated_dual_gemm_dual_x,
)
from scripts.perf.triangle_input_ln_dual_gemm_probe import (
    C_Z,
    DEFAULT_CONFIGS,
    DTYPE,
    Config,
    config_name,
    diff_stats,
    lengths_from_args,
    parse_config,
    parse_ints,
    time_cuda,
)


@triton.jit
def _dbij_row_stats_kernel(
    update,
    mean,
    rstd,
    rows: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offs = tl.arange(0, 256)
    mask = offs < channels
    # `update` is contiguous `[D, B, N, N]`.  A logical BIJ row is strided by
    # `rows` along D, so loading all channels is a gather rather than one
    # contiguous vector.  That is exactly the layout boundary CUEQ currently
    # resolves with `layer_norm_transpose`.
    values = tl.load(update + offs * rows + row, mask=mask, other=0.0).to(tl.float32)
    mu = tl.sum(values, axis=0) / channels
    centered = tl.where(mask, values - mu, 0.0)
    var = tl.sum(centered * centered, axis=0) / channels
    tl.store(mean + row, mu)
    tl.store(rstd + row, tl.rsqrt(var + eps))


@triton.jit
def _output_ln_dualgemm_residual_kernel(
    x_gate,
    update,
    mean,
    rstd,
    norm_weight,
    norm_bias,
    gate_weight,
    value_weight,
    residual,
    out,
    rows,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    offs_n = (pid_n * block_n + tl.arange(0, block_n)).to(tl.int64)
    offs_k = tl.arange(0, block_k).to(tl.int64)

    gate = tl.zeros((block_m, block_n), dtype=tl.float32)
    value = tl.zeros((block_m, block_n), dtype=tl.float32)
    row_mean = tl.load(mean + offs_m, mask=offs_m < rows, other=0.0)[:, None]
    row_rstd = tl.load(rstd + offs_m, mask=offs_m < rows, other=1.0)[:, None]

    for k0 in range(0, channels, block_k):
        k = k0 + offs_k
        k_mask = k < channels
        xg = tl.load(
            x_gate + offs_m[:, None] * channels + k[None, :],
            mask=(offs_m[:, None] < rows) & k_mask[None, :],
            other=0.0,
        )
        upd = tl.load(
            update + k[None, :] * rows + offs_m[:, None],
            mask=(offs_m[:, None] < rows) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
        b = tl.load(norm_bias + k, mask=k_mask, other=0.0).to(tl.float32)
        upd = (upd - row_mean) * row_rstd * w[None, :] + b[None, :]
        upd = upd.to(tl.bfloat16)

        wg = tl.load(
            gate_weight + offs_n[:, None] * channels + k[None, :],
            mask=(offs_n[:, None] < channels) & k_mask[None, :],
            other=0.0,
        )
        wv = tl.load(
            value_weight + offs_n[:, None] * channels + k[None, :],
            mask=(offs_n[:, None] < channels) & k_mask[None, :],
            other=0.0,
        )
        gate += tl.dot(xg, tl.trans(wg))
        value += tl.dot(upd, tl.trans(wv))

    result = (1.0 / (1.0 + tl.exp(-gate))) * value
    res = tl.load(
        residual + offs_m[:, None] * channels + offs_n[None, :],
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < channels),
        other=0.0,
    )
    tl.store(
        out + offs_m[:, None] * channels + offs_n[None, :],
        result + res,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < channels),
    )


def make_tensors(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    lengths = lengths_from_args(args)
    tokens = max(lengths)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x_gate = torch.zeros(len(lengths), tokens, tokens, C_Z, device="cuda", dtype=DTYPE)
    residual = torch.zeros_like(x_gate)
    update = torch.zeros(C_Z, len(lengths), tokens, tokens, device="cuda", dtype=DTYPE)
    for batch_index, length in enumerate(lengths):
        x_gate[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        residual[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        update[:, batch_index, :length, :length] = torch.randn(
            C_Z, length, length, device="cuda", dtype=DTYPE, generator=generator
        )
    return {
        "x_gate": x_gate,
        "update": update,
        "residual": residual,
        "norm_weight": torch.randn(C_Z, device="cuda", dtype=torch.float32),
        "norm_bias": torch.randn(C_Z, device="cuda", dtype=torch.float32),
        "gate_weight": torch.randn(C_Z, C_Z, device="cuda", dtype=DTYPE),
        "value_weight": torch.randn(C_Z, C_Z, device="cuda", dtype=DTYPE),
    }


def cueq_output_boundary(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    update_norm = layer_norm_transpose(
        tensors["update"],
        tensors["norm_weight"],
        tensors["norm_bias"],
        eps=1e-5,
        layout="dbij->bijd",
    )
    out = fused_sigmoid_gated_dual_gemm_dual_x(
        tensors["x_gate"],
        update_norm,
        tensors["gate_weight"],
        tensors["value_weight"],
    )
    return out + tensors["residual"]


def triton_output_boundary(
    tensors: dict[str, torch.Tensor],
    config: Config,
) -> torch.Tensor:
    bm, bn, bk, nw, ns = config
    x_gate = tensors["x_gate"]
    rows = x_gate.numel() // C_Z
    mean = torch.empty(rows, device=x_gate.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    out = torch.empty_like(x_gate.reshape(rows, C_Z))
    _dbij_row_stats_kernel[(rows,)](tensors["update"], mean, rstd, rows, C_Z, 1e-5)
    grid = (triton.cdiv(rows, bm), triton.cdiv(C_Z, bn))
    _output_ln_dualgemm_residual_kernel[grid](
        x_gate,
        tensors["update"],
        mean,
        rstd,
        tensors["norm_weight"],
        tensors["norm_bias"],
        tensors["gate_weight"],
        tensors["value_weight"],
        tensors["residual"],
        out,
        rows,
        C_Z,
        bm,
        bn,
        bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out.reshape_as(x_gate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--configs", nargs="*", type=parse_config)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    lengths = lengths_from_args(args)
    tensors = make_tensors(args)
    configs = args.configs or DEFAULT_CONFIGS

    reference, cueq_ms = time_cuda(lambda: cueq_output_boundary(tensors), args.warmup, args.iters)
    rows: dict[str, Any] = {}
    best_name = ""
    best_ms = float("inf")
    for config in configs:
        name = config_name(config)
        try:
            candidate, ms = time_cuda(
                lambda cfg=config: triton_output_boundary(tensors, cfg),
                args.warmup,
                args.iters,
            )
        except Exception as exc:
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        rows[name] = {
            "ms": ms,
            "speedup_vs_cueq_boundary": cueq_ms / ms,
            "parity": diff_stats(reference, candidate),
        }
        if ms < best_ms:
            best_name, best_ms = name, ms

    result = {
        "args": {**vars(args), "lengths": lengths, "configs": [config_name(config) for config in configs]},
        "shape": {
            "batch": len(lengths),
            "tokens": max(lengths),
            "rows": len(lengths) * max(lengths) * max(lengths),
            "c_z": C_Z,
        },
        "cueq_boundary_ms": cueq_ms,
        "candidates": rows,
        "best": rows[best_name] | {"name": best_name} if best_name else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
