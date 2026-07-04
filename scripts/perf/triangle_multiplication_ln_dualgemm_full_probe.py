#!/usr/bin/env python3
"""Full triangle-multiplication screen for the fused LN + dual-GEMM producer.

The input-producer probe can look artificially good because CUEQ's final
output gate also needs the normalized input tensor.  This script tests the
realistic boundary: the Triton producer writes both the projected `[2D,B,N,N]`
layout and the normalized `[B,N,N,D]` tensor, then the rest of triangle
multiplication is identical to the current CUEQ/PyTorch composition.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch

from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm_dual_x
from scripts.perf.triangle_input_ln_dual_gemm_probe import (
    Config,
    DEFAULT_CONFIGS,
    DTYPE,
    config_name,
    diff_stats,
    lengths_from_args,
    make_inputs,
    make_module,
    parse_config,
    parse_ints,
    time_cuda,
    triton_producer,
)


TensorFn = Callable[[], torch.Tensor]


def run_baseline(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return module(
        z.clone(),
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )


def run_candidate(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    config: Config,
    *,
    direction: str,
) -> torch.Tensor:
    ab, x_norm = triton_producer(module, z, mask, config, return_norm=True)
    a, b = torch.chunk(ab, 2, dim=0)
    if direction == "outgoing":
        update = torch.einsum("dbik,dbjk->dbij", a, b)
    elif direction == "incoming":
        update = torch.einsum("dbki,dbkj->dbij", a, b)
    else:
        raise ValueError("direction must be outgoing or incoming")
    update = layer_norm_transpose(
        update,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        eps=1e-5,
        layout="dbij->bijd",
    )
    out = fused_sigmoid_gated_dual_gemm_dual_x(
        x_norm,
        update,
        module.linear_g.weight,
        module.linear_z.weight,
    )
    return out + z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
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
    module = make_module(args)
    z, mask = make_inputs(lengths, args)
    configs = args.configs or DEFAULT_CONFIGS

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        reference, baseline_ms = time_cuda(
            lambda: run_baseline(module, z, mask),
            args.warmup,
            args.iters,
        )
        rows: dict[str, Any] = {}
        best_name = ""
        best_ms = float("inf")
        for config in configs:
            name = config_name(config)
            try:
                candidate, ms = time_cuda(
                    lambda cfg=config: run_candidate(
                        module,
                        z,
                        mask,
                        cfg,
                        direction=args.direction,
                    ),
                    args.warmup,
                    args.iters,
                )
                rows[name] = {
                    "ms": ms,
                    "speedup_vs_baseline": baseline_ms / ms,
                    "parity": diff_stats(reference, candidate),
                }
                if ms < best_ms:
                    best_name, best_ms = name, ms
            except Exception as exc:
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"}

    result = {
        "args": {
            **vars(args),
            "lengths": lengths,
            "configs": [config_name(config) for config in configs],
        },
        "shape": {
            "batch": len(lengths),
            "tokens": max(lengths),
            "rows": len(lengths) * max(lengths) * max(lengths),
            "c_z": z.shape[-1],
        },
        "baseline_ms": baseline_ms,
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
