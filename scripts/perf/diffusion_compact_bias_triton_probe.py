#!/usr/bin/env python3
"""Screen compact-bias Triton attention for the diffusion transformer.

Benchmark-only: keep the fast flat GEMM/MLP ABI, store one pair bias per
record, and use a custom attention kernel that maps flat sample lanes back to
their record when loading bias.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable
from typing import Any

import torch

from diffusion_pair_bias_cache_probe import (
    _autocast,
    _cached_transformer,
    _cached_transformer_compact_bias,
    _compare,
    _dtype,
    _precompute_biases,
    _precompute_compact_biases,
    _time,
)
from diffusion_transformer_compile_probe import _make_inputs
from protenix.model.modules.diffusion_attention_triton import (
    triton_compact_bias_attention,
)
from protenix.model.modules.fused_elementwise_triton import (
    fused_sigmoid_mul,
    fused_sigmoid_mul_add,
)
from protenix.model.modules.transformer import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
    fused_attention_residual_enabled,
    fused_transition_residual_enabled,
)


TensorFn = Callable[[], torch.Tensor]


def _compact_bias_for_kernel(bias: torch.Tensor) -> torch.Tensor:
    if bias.dim() == 5:
        if bias.shape[1] != 1:
            raise ValueError("compact bias must have singleton sample axis")
        return bias[:, 0].contiguous()
    if bias.dim() != 4:
        raise ValueError("compact bias must be [B,H,N,N] or [B,1,H,N,N]")
    return bias.contiguous()


def _cached_attention_block_compact_triton(
    block: DiffusionTransformerBlock,
    a: torch.Tensor,
    s: torch.Tensor,
    bias: torch.Tensor,
    samples: int,
    *,
    block_m: int,
    num_warps: int,
    dot_input_precision: str,
) -> torch.Tensor:
    """Block forward with flat GEMMs and compact-bias Triton attention."""

    apb = block.attention_pair_bias
    q_x = apb.layernorm_a(a=a, s=s)
    q, k, v = apb.attention._prep_qkv(q_x=q_x, kv_x=q_x, apply_scale=True)
    attn = triton_compact_bias_attention(
        q,
        k,
        v,
        _compact_bias_for_kernel(bias),
        sample_count=samples,
        block_m=block_m,
        num_warps=num_warps,
        dot_input_precision=dot_input_precision,
    )
    if attn is None:
        raise RuntimeError("compact-bias Triton attention rejected this shape")
    attn = apb.attention._wrap_up(attn.transpose(-2, -3), q_x)

    gate = apb.linear_a_last(s)
    if fused_attention_residual_enabled() and isinstance(block.drop_path, torch.nn.Identity):
        attn_out = fused_sigmoid_mul_add(gate, attn, a)
    else:
        attn_out = fused_sigmoid_mul(gate, attn) + a

    if fused_transition_residual_enabled() and isinstance(block.drop_path, torch.nn.Identity):
        return block.conditioned_transition_block(a=attn_out, s=s, residual=attn_out)
    return block.conditioned_transition_block(a=attn_out, s=s) + attn_out


def _cached_transformer_compact_triton(
    model: DiffusionTransformer,
    a: torch.Tensor,
    s: torch.Tensor,
    biases: list[torch.Tensor],
    samples: int,
    *,
    block_m: int,
    num_warps: int,
    dot_input_precision: str,
) -> torch.Tensor:
    for block, bias in zip(model.blocks, biases):
        a = _cached_attention_block_compact_triton(
            block,
            a,
            s,
            bias,
            samples,
            block_m=block_m,
            num_warps=num_warps,
            dot_input_precision=dot_input_precision,
        )
    return a


def _case(
    name: str,
    fn: TensorFn,
    args: argparse.Namespace,
    reference: torch.Tensor,
    baseline_ms: float,
) -> dict[str, Any]:
    try:
        out, ms, peak = _time(fn, args.warmup, args.iters)
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        return {"name": name, "ok": False, "error": repr(exc)}
    return {
        "name": name,
        "ok": True,
        "mean_ms": ms,
        "speedup_vs_expanded_cache": baseline_ms / ms,
        "peak_allocated_mib": peak,
        "parity_vs_expanded_cache": _compare(out, reference),
    }


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=parse_int_list, default=[124, 220])
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--block-m", type=parse_int_list, default=[16, 32])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument(
        "--dot-input-precision",
        choices=["tf32", "tf32x3", "ieee"],
        default="tf32",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def run_shape(args: argparse.Namespace, tokens: int) -> dict[str, Any]:
    torch.manual_seed(args.seed + tokens)
    dtype = _dtype(args.dtype)
    model = DiffusionTransformer(
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        n_blocks=args.blocks,
        n_heads=args.heads,
    ).cuda().eval()
    inputs = _make_inputs(
        argparse.Namespace(
            batch=args.batch,
            samples=args.samples,
            tokens=tokens,
            c_a=args.c_a,
            c_s=args.c_s,
            c_z=args.c_z,
            dtype=args.dtype,
            valid_fraction=args.valid_fraction,
        )
    )

    with _autocast(dtype), torch.inference_mode():
        expanded_biases = _precompute_biases(
            model, inputs["z"], inputs["token_mask"], args.samples
        )
        compact_biases = _precompute_compact_biases(
            model, inputs["z"], inputs["token_mask"], args.samples
        )

    expanded_bytes = sum(b.numel() * b.element_size() for b in expanded_biases)
    compact_bytes = sum(b.numel() * b.element_size() for b in compact_biases)

    def expanded_cache() -> torch.Tensor:
        with _autocast(dtype):
            return _cached_transformer(
                model, inputs["a"], inputs["s"], expanded_biases
            )

    expanded_out, expanded_ms, expanded_peak = _time(
        expanded_cache, args.warmup, args.iters
    )

    def compact_rank5() -> torch.Tensor:
        with _autocast(dtype):
            return _cached_transformer_compact_bias(
                model, inputs["a"], inputs["s"], compact_biases, args.samples
            )

    rows = [
        {
            "name": "expanded_bias_rank4_sdpa",
            "ok": True,
            "mean_ms": expanded_ms,
            "speedup_vs_expanded_cache": 1.0,
            "peak_allocated_mib": expanded_peak,
        },
        _case(
            "compact_bias_rank5_sdpa",
            compact_rank5,
            args,
            expanded_out,
            expanded_ms,
        ),
    ]
    for block_m in args.block_m:

        def compact_triton(block_m: int = block_m) -> torch.Tensor:
            with _autocast(dtype):
                return _cached_transformer_compact_triton(
                    model,
                    inputs["a"],
                    inputs["s"],
                    compact_biases,
                    args.samples,
                    block_m=block_m,
                    num_warps=args.num_warps,
                    dot_input_precision=args.dot_input_precision,
                )

        rows.append(
            _case(
                f"compact_bias_triton_block_m{block_m}",
                compact_triton,
                args,
                expanded_out,
                expanded_ms,
            )
        )

    return {
        "tokens": tokens,
        "batch": args.batch,
        "samples": args.samples,
        "head_dim": args.c_a // args.heads,
        "c_z": args.c_z,
        "expanded_bias_mib": expanded_bytes / 2**20,
        "compact_bias_mib": compact_bytes / 2**20,
        "cases": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_compact_bias_triton_probe requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    results = [run_shape(args, tokens) for tokens in args.tokens]
    print(
        json.dumps(
            {
                "args": vars(args),
            "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
