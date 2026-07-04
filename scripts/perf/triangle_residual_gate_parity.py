#!/usr/bin/env python3
"""Parity check for the opt-in triangle residual-gate fusion.

The fused path intentionally changes only the final store of CUEQ triangle
multiplication: instead of materialising ``update`` and then launching a second
kernel for ``update + residual``, it adds the residual inside the output-gate
Triton kernel.  This script compares the full PairformerBlock output with the
flag disabled/enabled so we catch mistakes that an isolated GEMM benchmark
would miss.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from contextlib import contextmanager
from typing import Iterator

import torch

from protenix.model.modules.pairformer import PairformerBlock


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
def triangle_residual_gate(enabled: bool) -> Iterator[None]:
    old = os.environ.get("PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL")
    os.environ["PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL"] = (
        "1" if enabled else "0"
    )
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL", None)
        else:
            os.environ["PROTENIX_TRITON_TRIANGLE_OUTPUT_GATE_RESIDUAL"] = old


@contextmanager
def cuda_autocast(dtype_name: str) -> Iterator[None]:
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def make_block(args: argparse.Namespace) -> PairformerBlock:
    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()

    # Some Protenix projections are deliberately zero-initialised.  For a parity
    # test that should exercise the residual-gate path, randomise only those
    # zero matrices while leaving normal initialisation untouched.
    with torch.no_grad():
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 104729)
        for parameter in block.parameters():
            if parameter.ndim >= 2 and not torch.any(parameter):
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_inputs(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    device = torch.device("cuda")
    s = torch.randn(args.batch, args.tokens, args.c_s, device=device, dtype=dtype)
    z = torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_z,
        device=device,
        dtype=dtype,
    )
    pair_mask = torch.ones(
        args.batch,
        args.tokens,
        args.tokens,
        device=device,
        dtype=dtype,
    )
    return s, z, pair_mask


def run_block(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with triangle_residual_gate(fused), torch.inference_mode(), cuda_autocast(
        args.compute_dtype
    ):
        return block(
            s.clone(),
            z.clone(),
            pair_mask,
            triangle_multiplicative="cuequivariance",
            triangle_attention="cuequivariance",
            inplace_safe=True,
            chunk_size=None,
        )


def time_block(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    fused: bool,
) -> float:
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_block(block, s, z, pair_mask, args, fused=fused)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_block(block, s, z, pair_mask, args, fused=fused)
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / args.iters


def diff_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int | bool]:
    diff = (candidate.float() - reference.float()).abs()
    ref_abs = reference.float().abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "allclose_rtol_1e-2_atol_5e-2": bool(
            torch.allclose(candidate.float(), reference.float(), rtol=1e-2, atol=5e-2)
        ),
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "max_ref_abs": float(ref_abs.max().item()),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=251)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_residual_gate_parity requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    block = make_block(args)
    s, z, pair_mask = make_inputs(args)

    base_s, base_z = run_block(block, s, z, pair_mask, args, fused=False)
    fused_s, fused_z = run_block(block, s, z, pair_mask, args, fused=True)
    torch.cuda.synchronize()

    base_ms = time_block(block, s, z, pair_mask, args, fused=False)
    fused_ms = time_block(block, s, z, pair_mask, args, fused=True)
    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "timing_ms": {
            "baseline": base_ms,
            "fused": fused_ms,
            "speedup": base_ms / fused_ms,
        },
        "parity": {
            "s": diff_stats(base_s, fused_s),
            "z": diff_stats(base_z, fused_z),
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
