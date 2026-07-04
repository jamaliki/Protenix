#!/usr/bin/env python3
"""Probe fusing Pairformer transition's output projection with the residual add.

After the Protenix-v2 CUEQ cache overlay, Nsight Compute shows pair transition
as a clean Protenix-owned remaining boundary:

    hidden = Transition(z)
    z = z + hidden

The transition already uses a Triton dual-GEMM kernel for the two input
projections on large pair batches.  The next cheap question is whether the
output projection can use cuBLAS' beta epilogue via ``torch.addmm``:

    z_flat + hidden @ W_out.T

This keeps the library GEMM instead of replacing it with a hand-written Triton
matmul, but removes the separate full-pair residual-add launch.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch
import torch.nn.functional as F

from protenix.model.modules.fused_elementwise_triton import triton_silu_mul_inplace_y
from protenix.model.modules.primitives import Transition


TensorFn = Callable[[], torch.Tensor]


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
def cuda_autocast(dtype_name: str) -> Iterator[None]:
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def make_transition(args: argparse.Namespace) -> Transition:
    module = Transition(c_in=args.c_in, n=args.factor).cuda().eval()
    if args.randomize_zero_weights:
        gen = torch.Generator(device="cuda").manual_seed(args.seed + 1009)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=gen)
    return module


def make_input(args: argparse.Namespace) -> torch.Tensor:
    dtype = getattr(torch, args.input_dtype)
    return torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_in,
        device="cuda",
        dtype=dtype,
    )


def _hidden(module: Transition, flat: torch.Tensor) -> torch.Tensor:
    y = module.layernorm1(flat)
    hidden = module._fused_input_projection(y)
    if hidden is not None:
        return hidden

    a = module.linear_no_bias_a(y)
    b = module.linear_no_bias_b(y)
    fused_b = triton_silu_mul_inplace_y(a, b)
    if fused_b is None:
        a = F.silu(a, inplace=True)
        b *= a
        return b
    return fused_b


def baseline(module: Transition, x: torch.Tensor) -> torch.Tensor:
    return x + module(x)


def addmm_residual(module: Transition, x: torch.Tensor) -> torch.Tensor:
    prefix_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    hidden = _hidden(module, flat)
    out = torch.addmm(flat, hidden, module.linear_no_bias.weight.t())
    return out.reshape(*prefix_shape, module.c_in)


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


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "allclose_rtol_1e-2_atol_5e-2": bool(
            torch.allclose(candidate.float(), reference.float(), rtol=1e-2, atol=5e-2)
        ),
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=251)
    parser.add_argument("--c-in", type=int, default=256)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Keep the probe on the same high-row transition path used by the v2 block.
    os.environ.setdefault("PROTENIX_TRITON_TRANSITION_DUAL_GEMM", "1")
    os.environ.setdefault("PROTENIX_TRITON_TRANSITION_DUAL_GEMM_MIN_ROWS", "0")

    module = make_transition(args)
    x = make_input(args)
    with cuda_autocast(args.compute_dtype):
        ref, ref_ms = time_cuda(lambda: baseline(module, x), args.warmup, args.iters)
        cand, cand_ms = time_cuda(
            lambda: addmm_residual(module, x),
            args.warmup,
            args.iters,
        )

    result = {
        "args": vars(args),
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "m_rows": args.batch * args.tokens * args.tokens,
            "c_in": args.c_in,
            "hidden": args.c_in * args.factor,
        },
        "timing_ms": {
            "baseline": ref_ms,
            "addmm_residual": cand_ms,
            "speedup": ref_ms / cand_ms,
        },
        "parity": diff_stats(ref, cand),
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
