#!/usr/bin/env python3
"""Measure the ceiling for fusing Protenix's residual add into CUEQ.

The fast inference path currently calls CUEQ's triangle-multiplication update
and then immediately does ``update.add_(z)``.  That add is a real full-pair HBM
pass, but it is also a simple high-bandwidth kernel.  This screen answers the
specific epilogue question before we write or modify native code:

    If CUEQ accepted a residual tensor and added it in its final store, how much
    triangle-multiplication time could we save at most?

This is deliberately narrower than the older output-boundary prototypes.  It
does not decompose CUEQ into Protenix-side producer/contraction/output pieces;
it measures the native CUEQ call, the explicit residual add, and the implied
speedup ceiling for Sam-style Protenix-v2 buckets.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Keep this probe on the native CUEQ path even if a shell has experimental
# triangle-multiplication flags set from another screen.
os.environ.setdefault("PROTENIX_TRITON_TRIANGLE_LN_DUAL_GEMM", "0")
os.environ.setdefault("PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL", "0")

import torch

from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    kernel_triangular_mult,
)


DTYPE = torch.bfloat16
C_Z = 256
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]
TensorFn = Callable[[], torch.Tensor]


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def make_module(direction: str, seed: int) -> torch.nn.Module:
    cls = (
        TriangleMultiplicationOutgoing
        if direction == "outgoing"
        else TriangleMultiplicationIncoming
    )
    module = cls(C_Z, C_Z).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(seed + 1729)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2 and not torch.any(parameter):
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(lengths: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = max(lengths)
    z = torch.zeros(len(lengths), tokens, tokens, C_Z, device="cuda", dtype=DTYPE)
    mask = torch.zeros(len(lengths), tokens, tokens, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def _cat_weights(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0),
        torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0),
    )


def cueq_update_only(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
    *,
    cached_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if cached_weights is None:
        p_in_weight, g_in_weight = _cat_weights(module)
    else:
        p_in_weight, g_in_weight = cached_weights
    return kernel_triangular_mult(
        z[None],
        direction=direction,
        mask=mask,
        norm_in_weight=module.layer_norm_in.weight,
        norm_in_bias=module.layer_norm_in.bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=module.layer_norm_out.weight,
        norm_out_bias=module.layer_norm_out.bias,
        p_out_weight=module.linear_z.weight,
        g_out_weight=module.linear_g.weight,
        eps=1e-5,
    )[0]


def cueq_update_plus_residual(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
    *,
    cached_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    update = cueq_update_only(
        module,
        z,
        mask,
        direction,
        cached_weights=cached_weights,
    )
    update.add_(z)
    return update


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


def residual_add_ms(z: torch.Tensor, warmup: int, iters: int) -> float:
    # Reusing the same large tensor is intentional: these v2 tensors exceed L2,
    # so the timing is a bandwidth measurement of the explicit residual pass.
    update = torch.randn_like(z)
    _, ms = time_cuda(lambda: update.add_(z), warmup, iters)
    return ms


def measure_case(args: argparse.Namespace, direction: str) -> dict[str, Any]:
    module = make_module(direction, args.seed)
    lengths = lengths_from_args(args)
    z, mask = make_inputs(lengths, args.seed)
    cached_weights = _cat_weights(module)

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        torch.cuda.reset_peak_memory_stats()
        _, update_ms = time_cuda(
            lambda: cueq_update_only(module, z, mask, direction),
            args.warmup,
            args.iters,
        )
        _, cached_update_ms = time_cuda(
            lambda: cueq_update_only(
                module,
                z,
                mask,
                direction,
                cached_weights=cached_weights,
            ),
            args.warmup,
            args.iters,
        )
        _, update_plus_add_ms = time_cuda(
            lambda: cueq_update_plus_residual(module, z, mask, direction),
            args.warmup,
            args.iters,
        )
        add_ms = residual_add_ms(z, args.warmup, args.iters)

    no_add_ceiling = update_plus_add_ms / max(update_ms, 1e-9)
    cached_no_add_ceiling = update_plus_add_ms / max(cached_update_ms, 1e-9)
    add_fraction = add_ms / max(update_plus_add_ms, 1e-9)
    return {
        "direction": direction,
        "cueq_update_ms": update_ms,
        "cueq_update_cached_input_weights_ms": cached_update_ms,
        "cueq_update_plus_residual_ms": update_plus_add_ms,
        "residual_add_only_ms": add_ms,
        "residual_add_fraction_of_plus_residual": add_fraction,
        "max_speedup_if_native_epilogue_were_free": no_add_ceiling,
        "max_speedup_if_native_epilogue_and_cached_input_weights_were_free": (
            cached_no_add_ceiling
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument(
        "--direction",
        choices=["outgoing", "incoming", "both"],
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_residual_add_ceiling requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    lengths = lengths_from_args(args)
    directions = (
        ["outgoing", "incoming"] if args.direction == "both" else [args.direction]
    )
    cases = [measure_case(args, direction) for direction in directions]
    tokens = max(lengths)
    pair_rows = len(lengths) * tokens * tokens
    pair_tensor_mib = pair_rows * C_Z * torch.finfo(DTYPE).bits / 8 / 2**20
    result = {
        "args": {**vars(args), "lengths": lengths},
        "shape": {
            "batch": len(lengths),
            "tokens": tokens,
            "pair_rows": pair_rows,
            "c_z": C_Z,
            "pair_tensor_mib": pair_tensor_mib,
        },
        "cases": cases,
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
