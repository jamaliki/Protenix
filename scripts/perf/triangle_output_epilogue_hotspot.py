#!/usr/bin/env python3
"""Screen a fused triangle-attention output epilogue.

The promoted triangle-attention path already keeps the fast CUEQ/cuDNN
attention mainloop.  Nsight Compute shows the remaining local boundary is the
output epilogue:

    gate = linear_g(pair)
    gated = sigmoid(gate) * heads_first.transpose/flatten()
    update = gated @ W_out.T
    out = residual + update

This script tests the next fusion boundary without touching model code.  It
precomputes a real CUEQ attention output and gate for representative pairformer
shapes, then compares the current Triton gate/flatten + cuBLAS output GEMM +
residual add against one Triton matmul epilogue.  A candidate must beat the full
boundary, not just delete a launch on paper.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from protenix.model.modules.fused_elementwise_triton import (
    triton_sigmoid_mul_heads_first,
)
from protenix.model.triangular.layers import cuequivariance_triangular_attn
from protenix.model.triangular.output_epilogue_triton import (
    triton_gate_linear_residual_heads_first,
)
from protenix.model.triangular.triangular import TriangleAttention
from protenix.model.utils import permute_final_dims


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
def cuda_autocast(dtype_name: str):
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def candidate_gate_linear_residual(
    gate: torch.Tensor,
    heads_first: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    out = triton_gate_linear_residual_heads_first(
        gate,
        heads_first,
        weight,
        residual,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        require_env=False,
    )
    if out is None:
        raise RuntimeError("production Triton epilogue helper rejected fixture")
    return out


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
        starting=True,
    ).cuda()
    module.eval()
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 65537)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_fixture(
    module: TriangleAttention,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(
        args.batch, args.tokens, args.tokens, args.c_z, device="cuda", dtype=dtype
    )
    mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)

    x_norm = module.layer_norm(x)
    triangle_bias = permute_final_dims(module.linear(x_norm), (2, 0, 1)).unsqueeze(-4)
    q, k, v = module.mha._prep_qkv(
        x_norm,
        x_norm,
        apply_scale=False,
        cueq_heads_first=True,
    )
    mask_bool = (mask == 1)[..., :, None, None, :]
    heads = cuequivariance_triangular_attn(
        q,
        k,
        v,
        triangle_bias.float(),
        mask_bool,
        1.0 / math.sqrt(module.c_hidden),
    )
    if isinstance(heads, tuple):
        heads = heads[0]
    gate = module.mha.linear_g(x_norm).contiguous()
    weight = module.mha.linear_o.weight.to(dtype=dtype).contiguous()
    return {
        "gate": gate,
        "heads": heads.contiguous(),
        "weight": weight,
        "residual": x.contiguous(),
    }


def baseline_boundary(fixture: dict[str, torch.Tensor]) -> torch.Tensor:
    gated = triton_sigmoid_mul_heads_first(fixture["gate"], fixture["heads"])
    if gated is None:
        heads = fixture["heads"].transpose(-2, -3)
        gate = torch.sigmoid(fixture["gate"]).view(heads.shape)
        gated = (heads * gate).flatten(-2)
    projected = F.linear(gated, fixture["weight"])
    return projected + fixture["residual"]


def baseline_subranges(
    fixture: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def timed(name: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        events.append((name, start, end))
        return out

    gated = timed(
        "gate_flatten",
        lambda: triton_sigmoid_mul_heads_first(fixture["gate"], fixture["heads"]),
    )
    projected = timed(
        "output_projection",
        lambda: F.linear(gated, fixture["weight"]),
    )
    out = timed("residual_add", lambda: projected + fixture["residual"])
    torch.cuda.synchronize()
    return out, {name: start.elapsed_time(end) for name, start, end in events}


def cuda_time(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
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


def max_mean_abs(
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


def parse_configs(value: str) -> list[tuple[int, int, int, int]]:
    configs = []
    for part in value.split(";"):
        if not part.strip():
            continue
        block_m, block_n, block_k, warps = [int(x) for x in part.split(",")]
        configs.append((block_m, block_n, block_k, warps))
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--input-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument(
        "--configs",
        default="16,32,32,4;16,64,32,4;32,32,32,4;32,64,32,4;32,64,64,4",
        help="Semicolon-separated block_m,block_n,block_k,num_warps configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_output_epilogue_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    module = make_module(args)
    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        fixture = make_fixture(module, args)
        reference, baseline_ranges = baseline_subranges(fixture)
        _, baseline_ms = cuda_time(
            lambda: baseline_boundary(fixture),
            args.warmup,
            args.iters,
        )

        candidates: list[dict[str, Any]] = []
        for block_m, block_n, block_k, warps in parse_configs(args.configs):
            out, ms = cuda_time(
                lambda bm=block_m, bn=block_n, bk=block_k, nw=warps: (
                    candidate_gate_linear_residual(
                        fixture["gate"],
                        fixture["heads"],
                        fixture["weight"],
                        fixture["residual"],
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                        num_warps=nw,
                    )
                ),
                args.warmup,
                args.iters,
            )
            candidates.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "num_warps": warps,
                    "ms": ms,
                    "speedup_vs_baseline": baseline_ms / ms,
                    "parity": max_mean_abs(reference, out),
                }
            )

    candidates.sort(key=lambda row: row["ms"])
    result = {
        "args": vars(args),
        "baseline": {"ms": baseline_ms, "subranges_ms": baseline_ranges},
        "candidates": candidates,
        "device": torch.cuda.get_device_name(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
