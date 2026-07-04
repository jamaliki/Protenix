#!/usr/bin/env python3
"""Screen a fused pair-transition input-projection kernel.

Pairformer ``Transition(c_in=128, n=4)`` computes two projections from the same
large flattened pair tensor and immediately consumes them as ``silu(a) * b``.
This probe tests whether a direct Triton dual-GEMM epilogue can remove that
intermediate HBM traffic on the real pairformer shapes.  It is a benchmark
screen only; production code should be written only if this beats real cuBLAS
module weights, not just a synthetic matmul.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

os.environ.setdefault("LAYERNORM_TYPE", "normal")

from protenix.model.modules.primitives import Transition


TensorFn = Callable[[], torch.Tensor]
Config = tuple[int, int, int, int, int]
DEFAULT_CONFIGS: list[Config] = [
    (16, 64, 64, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 3),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
]


def config_name(config: Config) -> str:
    bm, bn, bk, nw, ns = config
    return f"m{bm}_n{bn}_k{bk}_w{nw}_s{ns}"


def parse_config(spec: str) -> Config:
    parts = tuple(int(part) for part in spec.split("x"))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "configs must be BLOCK_MxBLOCK_NxBLOCK_KxWARPSxSTAGES"
        )
    return parts  # type: ignore[return-value]


@triton.jit
def _dual_projection_silu_kernel(
    y_ptr,
    wa_ptr,
    wb_ptr,
    out_ptr,
    m_rows,
    stride_y_m: tl.constexpr,
    stride_wa_n: tl.constexpr,
    stride_wb_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    n_hidden: tl.constexpr,
    k_input: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    acc_a = tl.zeros((block_m, block_n), dtype=tl.float32)
    acc_b = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k0 in range(0, k_input, block_k):
        k = k0 + offs_k
        y = tl.load(
            y_ptr + offs_m[:, None] * stride_y_m + k[None, :],
            mask=(offs_m[:, None] < m_rows) & (k[None, :] < k_input),
            other=0.0,
        )
        wa = tl.load(
            wa_ptr + offs_n[:, None] * stride_wa_n + k[None, :],
            mask=(offs_n[:, None] < n_hidden) & (k[None, :] < k_input),
            other=0.0,
        )
        wb = tl.load(
            wb_ptr + offs_n[:, None] * stride_wb_n + k[None, :],
            mask=(offs_n[:, None] < n_hidden) & (k[None, :] < k_input),
            other=0.0,
        )
        acc_a += tl.dot(y, tl.trans(wa))
        acc_b += tl.dot(y, tl.trans(wb))

    # Fusing SiLU here avoids storing both pre-activation matrices to HBM.
    out = (acc_a / (1.0 + tl.exp(-acc_a))) * acc_b
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < n_hidden),
    )


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


def make_transition(args: argparse.Namespace) -> Transition:
    module = Transition(c_in=args.c_in, n=args.factor).cuda().eval()
    if args.randomize_zero_weights:
        gen = torch.Generator(device="cuda").manual_seed(args.seed + 1009)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=gen)
    return module


def make_y(module: Transition, args: argparse.Namespace) -> torch.Tensor:
    x = torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_in,
        device="cuda",
        dtype=getattr(torch, args.input_dtype),
    )
    with torch.autocast(device_type="cuda", dtype=getattr(torch, args.compute_dtype)):
        return module.layernorm1(x.reshape(-1, args.c_in))


def two_gemm_hidden(module: Transition, y: torch.Tensor) -> torch.Tensor:
    a = module.linear_no_bias_a(y)
    b = module.linear_no_bias_b(y)
    b *= F.silu(a, inplace=True)
    return b


def dual_gemm_hidden(
    y: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    bm, bn, bk, nw, ns = config
    y = y if y.is_contiguous() else y.contiguous()
    m_rows, k_input = y.shape
    n_hidden = wa.shape[0]
    out = torch.empty((m_rows, n_hidden), dtype=y.dtype, device=y.device)
    grid = (triton.cdiv(m_rows, bm), triton.cdiv(n_hidden, bn))
    _dual_projection_silu_kernel[grid](
        y,
        wa,
        wb,
        out,
        m_rows,
        y.stride(0),
        wa.stride(0),
        wb.stride(0),
        out.stride(0),
        n_hidden,
        k_input,
        bm,
        bn,
        bk,
        num_warps=nw,
        num_stages=ns,
    )
    return out


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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=251)
    parser.add_argument("--c-in", type=int, default=128)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--input-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--configs", nargs="*", type=parse_config)
    parser.add_argument("--output-json")
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    configs = args.configs or DEFAULT_CONFIGS
    module = make_transition(args)
    y = make_y(module, args)
    dtype = y.dtype
    wa = module.linear_no_bias_a.weight.to(dtype=dtype).contiguous()
    wb = module.linear_no_bias_b.weight.to(dtype=dtype).contiguous()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        ref, ref_ms = time_cuda(lambda: two_gemm_hidden(module, y), args.warmup, args.iters)

        rows: dict[str, Any] = {}
        best_name, best_config, best_ms, best_hidden = "", configs[0], float("inf"), None
        for config in configs:
            name = config_name(config)
            try:
                out, ms = time_cuda(
                    lambda cfg=config: dual_gemm_hidden(y, wa, wb, cfg),
                    args.warmup,
                    args.iters,
                )
            except Exception as exc:
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            rows[name] = {
                "ms": ms,
                "speedup_vs_two_gemm": ref_ms / ms,
                "parity": diff_stats(ref, out),
            }
            if ms < best_ms:
                best_name, best_config, best_ms, best_hidden = name, config, ms, out

        assert best_hidden is not None
        default_out, default_total = time_cuda(
            lambda: module.linear_no_bias(two_gemm_hidden(module, y)),
            args.warmup,
            args.iters,
        )
        cand_out, cand_total = time_cuda(
            lambda: module.linear_no_bias(dual_gemm_hidden(y, wa, wb, best_config)),
            args.warmup,
            args.iters,
        )

    result = {
        "args": vars(args) | {"configs": [config_name(config) for config in configs]},
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "m_rows": y.shape[0],
            "c_in": args.c_in,
            "hidden": args.c_in * args.factor,
        },
        "hidden_boundary": {
            "two_gemm_ms": ref_ms,
            "candidates": rows,
            "best": rows[best_name] | {"name": best_name},
        },
        "input_to_output_boundary": {
            "two_gemm_ms": default_total,
            "candidate_ms": cand_total,
            "candidate_speedup_vs_two_gemm": default_total / cand_total,
            "candidate_parity": diff_stats(default_out, cand_out),
        },
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
