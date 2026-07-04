#!/usr/bin/env python3
"""Probe residual fusion at the CUEQ triangle-multiplication output gate.

CUEQ triangle multiplication ends with a dual-input gated GEMM:

    update = sigmoid(x_in @ W_gate.T) * (x_out @ W_value.T)

Protenix then immediately adds the residual pair tensor.  For Protenix-v2
(`c_z=256`) that residual add is a full B*N*N*C HBM pass inside the hottest
pairformer component.  This probe tests the exact fusion boundary before any
model-path change: keep the same math, but add the residual in the final store.

This is intentionally an isolated screen.  A promoted implementation would need
to beat CUEQ plus PyTorch's high-bandwidth add in this benchmark and then move a
full PairformerBlock/model gate.
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
from cuequivariance_ops_torch.gated_gemm_torch import (
    fused_sigmoid_gated_dual_gemm_dual_x,
)


Config = tuple[int, int, int, int, int]
TensorFn = Callable[[], torch.Tensor]

DEFAULT_CONFIGS: list[Config] = [
    (32, 64, 64, 4, 3),
    (64, 64, 64, 4, 3),
    (64, 128, 64, 4, 3),
    (128, 64, 64, 4, 3),
    (128, 128, 64, 4, 3),
    (64, 64, 128, 4, 3),
]


def parse_config(spec: str) -> Config:
    parts = tuple(int(part) for part in spec.split("x"))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "configs must be BLOCK_MxBLOCK_NxBLOCK_KxWARPSxSTAGES"
        )
    return parts  # type: ignore[return-value]


def config_name(config: Config) -> str:
    bm, bn, bk, nw, ns = config
    return f"m{bm}_n{bn}_k{bk}_w{nw}_s{ns}"


@triton.jit
def _gate_value_residual_kernel(
    x_gate_ptr,
    x_value_ptr,
    w_gate_ptr,
    w_value_ptr,
    residual_ptr,
    out_ptr,
    m_rows,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
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
    for k0 in range(0, k_in, block_k):
        k = k0 + offs_k
        xg = tl.load(
            x_gate_ptr + offs_m[:, None] * k_in + k[None, :],
            mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_in),
            other=0.0,
        )
        xv = tl.load(
            x_value_ptr + offs_m[:, None] * k_in + k[None, :],
            mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_in),
            other=0.0,
        )
        wg = tl.load(
            w_gate_ptr + offs_n[:, None] * k_in + k[None, :],
            mask=(offs_n[:, None] < n_out) & (k[None, :] < k_in),
            other=0.0,
        )
        wv = tl.load(
            w_value_ptr + offs_n[:, None] * k_in + k[None, :],
            mask=(offs_n[:, None] < n_out) & (k[None, :] < k_in),
            other=0.0,
        )
        gate += tl.dot(xg, tl.trans(wg))
        value += tl.dot(xv, tl.trans(wv))

    out = (1.0 / (1.0 + tl.exp(-gate))) * value
    residual = tl.load(
        residual_ptr + offs_m[:, None] * n_out + offs_n[None, :],
        mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_out),
        other=0.0,
    )
    out += residual
    tl.store(
        out_ptr + offs_m[:, None] * n_out + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_out),
    )


def candidate_gate_residual(
    x_gate: torch.Tensor,
    x_value: torch.Tensor,
    w_gate: torch.Tensor,
    w_value: torch.Tensor,
    residual: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    bm, bn, bk, nw, ns = config
    x_shape = x_gate.shape
    k_in = x_shape[-1]
    n_out = w_gate.shape[0]
    x_gate_2d = x_gate.reshape(-1, k_in).contiguous()
    x_value_2d = x_value.reshape(-1, k_in).contiguous()
    residual_2d = residual.reshape(-1, n_out).contiguous()
    out = torch.empty_like(residual_2d)
    grid = (triton.cdiv(x_gate_2d.shape[0], bm), triton.cdiv(n_out, bn))
    _gate_value_residual_kernel[grid](
        x_gate_2d,
        x_value_2d,
        w_gate.contiguous(),
        w_value.contiguous(),
        residual_2d,
        out,
        x_gate_2d.shape[0],
        n_out,
        k_in,
        bm,
        bn,
        bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out.reshape(*x_shape[:-1], n_out)


def cueq_gate_residual(
    x_gate: torch.Tensor,
    x_value: torch.Tensor,
    w_gate: torch.Tensor,
    w_value: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    out = fused_sigmoid_gated_dual_gemm_dual_x(x_gate, x_value, w_gate, w_value)
    out.add_(residual)
    return out


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


def diff_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=251)
    parser.add_argument("--c-in", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--configs", nargs="*", type=parse_config)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, args.dtype)
    shape = (args.batch, args.tokens, args.tokens, args.c_in)
    x_gate = torch.randn(shape, device="cuda", dtype=dtype)
    x_value = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    w_gate = torch.randn(args.c_in, args.c_in, device="cuda", dtype=dtype)
    w_value = torch.randn(args.c_in, args.c_in, device="cuda", dtype=dtype)

    configs = args.configs or DEFAULT_CONFIGS
    reference, reference_ms = time_cuda(
        lambda: cueq_gate_residual(x_gate, x_value, w_gate, w_value, residual),
        args.warmup,
        args.iters,
    )

    rows: dict[str, Any] = {}
    best_name = ""
    best_ms = float("inf")
    for config in configs:
        name = config_name(config)
        try:
            out, ms = time_cuda(
                lambda cfg=config: candidate_gate_residual(
                    x_gate, x_value, w_gate, w_value, residual, cfg
                ),
                args.warmup,
                args.iters,
            )
        except Exception as exc:
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        rows[name] = {
            "ms": ms,
            "speedup_vs_cueq_plus_add": reference_ms / ms,
            "parity": diff_stats(reference, out),
        }
        if ms < best_ms:
            best_name = name
            best_ms = ms

    result = {
        "args": vars(args) | {"configs": [config_name(config) for config in configs]},
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "m_rows": args.batch * args.tokens * args.tokens,
            "c_in": args.c_in,
        },
        "cueq_plus_residual_ms": reference_ms,
        "candidates": rows,
        "best": rows[best_name] | {"name": best_name} if best_name else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
