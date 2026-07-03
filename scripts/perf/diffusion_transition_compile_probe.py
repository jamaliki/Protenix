#!/usr/bin/env python3
"""Screen compile/fusion for the diffusion transition sub-block.

The full diffusion-transformer `torch.compile` screen improved the isolated
24-block boundary, but the real model integration failed because Inductor
rewrote SDPA into a cuDNN plan that was invalid for this runtime.  This probe
keeps attention out of scope and tests the next safest boundary:

    ConditionedTransitionBlock(attn_residual, s) + attn_residual

That expression is pure feed-forward plumbing: adaptive LayerNorm, two input
GEMMs, SiLU/multiply, output GEMM, adaLN-Zero gate, and the block residual.
If compile can fuse useful work here, we can consider wrapping only this
sub-block while leaving eager SDPA untouched.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from protenix.model.modules.transformer import ConditionedTransitionBlock


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class TransitionPlusResidual(nn.Module):
    """Wrapper for the exact diffusion block transition boundary."""

    def __init__(
        self,
        block: ConditionedTransitionBlock,
        use_inner_residual: bool,
    ) -> None:
        super().__init__()
        self.block = block
        self.use_inner_residual = use_inner_residual

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.use_inner_residual:
            return self.block(a=a, s=s, residual=a)
        return self.block(a=a, s=s) + a


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _dtype(args.dtype)
    a = torch.randn(
        args.batch * args.samples,
        args.tokens,
        args.c_a,
        device="cuda",
        dtype=dtype,
    )
    s = torch.randn(
        args.batch * args.samples,
        args.tokens,
        args.c_s,
        device="cuda",
        dtype=dtype,
    )
    return a, s


def _cuda_time(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = fn()
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
    return out, start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _record(
    name: str,
    fn: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    try:
        start = time.perf_counter()
        out, mean_ms, peak_mib = _cuda_time(fn, args.warmup, args.iters)
        elapsed_sec = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        return {"name": name, "ok": False, "error": repr(exc)}
    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "peak_allocated_mib": peak_mib,
        "elapsed_sec_including_compile": elapsed_sec,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }
    if reference is not None:
        row["parity_vs_eager_default"] = _compare(out, reference)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_transition_compile_probe requires CUDA")

    # Match the promoted/default transition path.  The point of this probe is
    # whether compile can fuse the default PyTorch expression, not whether the
    # older global experimental Triton elementwise flags happen to help.
    os.environ["PROTENIX_TRITON_FUSED_ELEMENTWISE"] = "0"
    os.environ["PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION"] = "0"

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = _dtype(args.dtype)
    block = ConditionedTransitionBlock(
        c_a=args.c_a,
        c_s=args.c_s,
        n=args.factor,
    ).cuda().eval()
    a, s = _make_inputs(args)

    eager_default = TransitionPlusResidual(block, use_inner_residual=False).cuda().eval()
    eager_inner_residual = TransitionPlusResidual(
        block, use_inner_residual=True
    ).cuda().eval()
    compiled_default = torch.compile(
        eager_default,
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=False,
    )
    compiled_inner_residual = torch.compile(
        eager_inner_residual,
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=False,
    )

    def call(module: nn.Module) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=dtype != torch.float32,
        ):
            return module(a, s)

    eager_out, eager_ms, eager_peak = _cuda_time(
        lambda: call(eager_default),
        args.warmup,
        args.iters,
    )
    rows: list[dict[str, Any]] = [
        {
            "name": "eager_default_outer_residual",
            "ok": True,
            "mean_ms": eager_ms,
            "peak_allocated_mib": eager_peak,
            "output_shape": list(eager_out.shape),
            "output_dtype": str(eager_out.dtype),
        }
    ]
    rows.append(
        _record(
            "eager_inner_residual",
            lambda: call(eager_inner_residual),
            args,
            reference=eager_out,
        )
    )
    rows.append(
        _record(
            "compiled_default_outer_residual",
            lambda: call(compiled_default),
            args,
            reference=eager_out,
        )
    )
    rows.append(
        _record(
            "compiled_inner_residual",
            lambda: call(compiled_inner_residual),
            args,
            reference=eager_out,
        )
    )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shapes": {"a": list(a.shape), "s": list(s.shape)},
        "env": {
            key: os.environ.get(key)
            for key in (
                "PROTENIX_TRITON_FUSED_ELEMENTWISE",
                "PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION",
            )
        },
        "cases": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
