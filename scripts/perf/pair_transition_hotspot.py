#!/usr/bin/env python3
"""Profile the pairformer Transition boundary for many-sequence inference.

The low-sample design workload is trunk-bound.  After the triangle-attention
layout wins, pair transition is the next large non-attention piece of a
PairformerBlock.  This script isolates the exact `Transition(c_in=128, n=4)`
shape used by the pairformer and compares:

* the normal two-input-GEMM path;
* the existing guarded wide-input-GEMM + split-aware Triton SiLU/multiply path;
* `Transition.forward()` with those flags disabled/enabled.

The result is meant to answer whether there is still a concrete kernel boundary
worth implementing, or whether the remaining time is mostly cuBLAS GEMM.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F

# This harness runs in the inference environment used on Sandpit Tokyo, where
# the optional fast-layernorm extension may not already be built.  Use PyTorch's
# normal LayerNorm by default so the profiler measures pair-transition fusion
# boundaries rather than one-time extension compilation or CUDA_HOME setup.
os.environ.setdefault("LAYERNORM_TYPE", "normal")

from protenix.model.modules.fused_elementwise_triton import triton_silu_mul_split
from protenix.model.modules.primitives import Transition


TensorFn = Callable[[], torch.Tensor]


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


@contextmanager
def env_flag(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@contextmanager
def transition_fusion_env(enabled: bool) -> Iterator[None]:
    value = "1" if enabled else "0"
    with env_flag("PROTENIX_TRITON_FUSED_ELEMENTWISE", value), env_flag(
        "PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION", value
    ):
        yield


def cuda_time(fn: TensorFn, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
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


def event_time(
    name: str,
    fn: TensorFn,
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events.append((name, start, end))
    return out


def make_transition(args: argparse.Namespace) -> Transition:
    module = Transition(c_in=args.c_in, n=args.factor).cuda()
    module.eval()
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 2654435761)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
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


def flat_view(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape


def manual_baseline(module: Transition, x: torch.Tensor) -> torch.Tensor:
    flat, prefix_shape = flat_view(x)
    y = module.layernorm1(flat)
    a = module.linear_no_bias_a(y)
    a = F.silu(a, inplace=True)
    b = module.linear_no_bias_b(y)
    b *= a
    out = module.linear_no_bias(b)
    return out.reshape(*prefix_shape, module.c_in)


def manual_fused_input(module: Transition, x: torch.Tensor) -> torch.Tensor:
    flat, prefix_shape = flat_view(x)
    if not can_use_fused_input_projection(module, flat):
        raise RuntimeError(
            "wide input projection is disabled by Transition guard"
        )
    y = module.layernorm1(flat)
    projected = F.linear(y, module._input_projection_params())
    with transition_fusion_env(True):
        b = triton_silu_mul_split(projected, module.n * module.c_in)
    if b is None:
        raise RuntimeError("triton_silu_mul_split returned None")
    out = module.linear_no_bias(b)
    return out.reshape(*prefix_shape, module.c_in)


def can_use_fused_input_projection(
    module: Transition,
    flat: torch.Tensor,
) -> bool:
    """Ask the model whether the wide-GEMM path is legal for this flattened shape."""
    with torch.inference_mode(), transition_fusion_env(True):
        return bool(module._can_fuse_input_projection(flat))


def manual_timed(
    module: Transition,
    x: torch.Tensor,
    *,
    fused_input: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.inference_mode():
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        flat, prefix_shape = flat_view(x)
        y = event_time("layernorm", lambda: module.layernorm1(flat), events)
        if fused_input:
            if not can_use_fused_input_projection(module, flat):
                raise RuntimeError(
                    "wide input projection is disabled by Transition guard"
                )
            projected = event_time(
                "wide_input_projection",
                lambda: F.linear(y, module._input_projection_params()),
                events,
            )
            hidden = module.n * module.c_in
            b = event_time(
                "split_silu_mul",
                lambda: _split_silu_mul_required(projected, hidden),
                events,
            )
        else:
            a = event_time(
                "input_projection_a", lambda: module.linear_no_bias_a(y), events
            )
            a = event_time("silu_a", lambda: F.silu(a, inplace=True), events)
            b = event_time(
                "input_projection_b", lambda: module.linear_no_bias_b(y), events
            )
            b = event_time("multiply", lambda: b.mul_(a), events)
        out = event_time("output_projection", lambda: module.linear_no_bias(b), events)
        out = out.reshape(*prefix_shape, module.c_in)
        torch.cuda.synchronize()
    return out, {name: start.elapsed_time(end) for name, start, end in events}


def _split_silu_mul_required(projected: torch.Tensor, hidden: int) -> torch.Tensor:
    with transition_fusion_env(True):
        out = triton_silu_mul_split(projected, hidden)
    if out is None:
        raise RuntimeError("triton_silu_mul_split returned None")
    return out


def transition_forward_with_env(
    module: Transition,
    x: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    with transition_fusion_env(enabled):
        return module(x)


def max_mean_abs(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
    }


def profiler_rows(fn: TensorFn, warmup: int, limit: int) -> list[dict[str, Any]]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            fn()
        torch.cuda.synchronize()

    rows = []
    for event in profiler.key_averages():
        device_us = getattr(event, "device_time_total", None)
        if device_us is None:
            device_us = getattr(event, "cuda_time_total", 0.0)
        if device_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "cuda_ms": device_us / 1000.0,
                "calls": event.count,
                "shapes": str(event.input_shapes)[:240],
            }
        )
    rows.sort(key=lambda row: row["cuda_ms"], reverse=True)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--c-in", type=int, default=128)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--input-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    module = make_transition(args)
    x = make_input(args)
    amp = cuda_autocast(args.compute_dtype)

    flat, _ = flat_view(x)
    can_use_wide_input_projection = can_use_fused_input_projection(module, flat)
    variants: dict[str, TensorFn] = {
        "manual_two_input_gemms": lambda: manual_baseline(module, x),
        "forward_env_off": lambda: transition_forward_with_env(
            module, x, enabled=False
        ),
        "forward_env_on": lambda: transition_forward_with_env(module, x, enabled=True),
    }
    if can_use_wide_input_projection:
        variants["manual_wide_input_projection"] = lambda: manual_fused_input(module, x)

    results: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    with amp:
        # First calls can include cuBLAS/TRITON setup.  Warm the manual subrange
        # path once so the reported launch breakdown reflects steady-state work.
        manual_timed(module, x, fused_input=False)
        if can_use_wide_input_projection:
            manual_timed(module, x, fused_input=True)

        reference, baseline_parts = manual_timed(module, x, fused_input=False)
        if can_use_wide_input_projection:
            fused_parts_out, fused_parts = manual_timed(module, x, fused_input=True)
        else:
            fused_parts_out = None
            fused_parts = {}
        for name, fn in variants.items():
            out, ms = cuda_time(fn, args.warmup, args.iters)
            results[name] = {
                "ms": ms,
                "parity_vs_manual_two_input_gemms": max_mean_abs(reference, out),
            }
        if not can_use_wide_input_projection:
            results["manual_wide_input_projection"] = {
                "skipped": True,
                "reason": "disabled by Transition guard for this flattened shape",
            }
        results["manual_parts_two_input_gemms"] = baseline_parts
        results["manual_parts_wide_input_projection"] = fused_parts
        results["wide_input_projection_eligible"] = can_use_wide_input_projection
        if fused_parts_out is not None:
            results["manual_wide_parts_parity"] = max_mean_abs(reference, fused_parts_out)
        if args.profile:
            results["profiler_top_cuda"] = {
                name: profiler_rows(fn, args.warmup, args.profile_top)
                for name, fn in variants.items()
            }

    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "tokens": args.tokens,
            "c_in": args.c_in,
            "transition_factor": args.factor,
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "results": results,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
