#!/usr/bin/env python3
"""Screen torch.compile for static-shape diffusion trunk hotspots."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.modules.transformer import (
    AttentionPairBias,
    DiffusionTransformer,
    DiffusionTransformerBlock,
)
from protenix.model.utils import permute_final_dims


def cuda_time(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
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


class TrunkAttentionTarget(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.efficient_fusion = bool(args.efficient_fusion)
        self.module = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=args.heads,
            c_a=args.c_token,
            c_s=args.c_s,
            c_z=args.c_z,
            cross_attention_mode=False,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        a_norm = self.module.layernorm_a(a=a, s=s)
        return self.module.standard_multihead_attention(
            q=a_norm,
            kv=a_norm,
            z=z,
            enable_efficient_fusion=self.efficient_fusion,
        )


class TrunkTransitionTarget(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.block = DiffusionTransformerBlock(
            n_heads=args.heads,
            c_a=args.c_token,
            c_s=args.c_s,
            c_z=args.c_z,
            cross_attention_mode=False,
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.block.conditioned_transition_block(a=a, s=s)


class TrunkBlockTarget(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.efficient_fusion = bool(args.efficient_fusion)
        self.block = DiffusionTransformerBlock(
            n_heads=args.heads,
            c_a=args.c_token,
            c_s=args.c_s,
            c_z=args.c_z,
            cross_attention_mode=False,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            a=a,
            s=s,
            z=z,
            enable_efficient_fusion=self.efficient_fusion,
        )[0]


class TrunkStackTarget(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.efficient_fusion = bool(args.efficient_fusion)
        self.stack = DiffusionTransformer(
            n_blocks=args.stack_blocks,
            n_heads=args.heads,
            c_a=args.c_token,
            c_s=args.c_s,
            c_z=args.c_z,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.stack(
            a=a,
            s=s,
            z=z,
            enable_efficient_fusion=self.efficient_fusion,
        )


def make_target(args: argparse.Namespace) -> nn.Module:
    targets = {
        "attention": TrunkAttentionTarget,
        "transition": TrunkTransitionTarget,
        "block": TrunkBlockTarget,
        "stack": TrunkStackTarget,
    }
    return targets[args.target](args).cuda().eval()


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    a = torch.randn(
        args.samples,
        args.tokens,
        args.c_token,
        device=device,
        dtype=dtype,
    )
    s = torch.randn(
        args.samples,
        args.tokens,
        args.c_s,
        device=device,
        dtype=dtype,
    )
    if args.target == "transition":
        return a, s
    if args.efficient_fusion:
        z = torch.randn(
            args.z_samples,
            args.c_z,
            args.tokens,
            args.tokens,
            device=device,
            dtype=dtype,
        )
    else:
        z = torch.randn(
            args.z_samples,
            args.tokens,
            args.tokens,
            args.c_z,
            device=device,
            dtype=dtype,
        )
    return a, s, z


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs_error": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs_error": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["attention", "transition", "block", "stack"],
        default="block",
    )
    parser.add_argument("--samples", type=int, default=640)
    parser.add_argument("--z-samples", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--c-token", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--stack-blocks", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile-warmup", type=int, default=3)
    parser.add_argument("--compile-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--efficient-fusion", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mode", default="reduce-overhead")
    parser.add_argument("--fullgraph", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("trunk_compile_hotspot requires CUDA")
    if not hasattr(torch, "compile"):
        raise RuntimeError("this torch build does not expose torch.compile")

    torch.manual_seed(args.seed)
    args.efficient_fusion = bool(args.efficient_fusion)
    target = make_target(args)
    inputs = make_inputs(args)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    def eager_run() -> torch.Tensor:
        with amp:
            return target(*inputs)

    eager_out, eager_ms = cuda_time(
        eager_run,
        warmup=args.warmup,
        iters=args.iters,
    )
    torch.cuda.reset_peak_memory_stats()
    compile_start = time.perf_counter()
    compiled_target = torch.compile(
        target,
        mode=args.mode,
        fullgraph=bool(args.fullgraph),
    )
    compile_construct_sec = time.perf_counter() - compile_start

    def compiled_run() -> torch.Tensor:
        with amp:
            return compiled_target(*inputs)

    compiled_out, compiled_ms = cuda_time(
        compiled_run,
        warmup=args.compile_warmup,
        iters=args.compile_iters,
    )
    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "input_shapes": [list(t.shape) for t in inputs],
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "speedup": eager_ms / compiled_ms,
        "compile_construct_sec": compile_construct_sec,
        "parity": compare(compiled_out, eager_out),
        "eager_all_finite": bool(torch.isfinite(eager_out).all().item()),
        "compiled_all_finite": bool(torch.isfinite(compiled_out).all().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
