#!/usr/bin/env python3
"""Screen the Pairformer ending-triangle-attention layout boundary.

Ending triangle attention is starting triangle attention over transposed pair
axes.  The historical block made that explicit by copying ``z`` into the
transposed layout before the attention and copying it back afterwards.  Those
copies are pure HBM traffic, so this screen compares them with the
layout-aware ``TriangleAttentionEndingNode`` path using identical weights.

This is an isolated filter only: a win here still needs an end-to-end inference
gate on the many-sequence, low-sample workload.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch

from protenix.model.modules.pairformer import PairformerBlock
from protenix.model.triangular.triangular import TriangleAttention


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
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype else nullcontext()
    with ctx:
        yield


def make_candidate_block(args: argparse.Namespace) -> PairformerBlock:
    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 104729)
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_materialized_block(args, template, state_dict) -> PairformerBlock:
    block = make_candidate_block(args)
    ending = template.tri_att_end
    block.tri_att_end = TriangleAttention(
        c_in=ending.c_in,
        c_hidden=ending.c_hidden,
        no_heads=ending.no_heads,
    ).cuda()
    block.load_state_dict(state_dict)
    block.eval()
    return block


def make_inputs(args: argparse.Namespace):
    dtype = getattr(torch, args.input_dtype)
    s = torch.randn(args.batch, args.tokens, args.c_s, device="cuda", dtype=dtype)
    z_shape = (args.batch, args.tokens, args.tokens, args.c_z)
    z = torch.randn(*z_shape, device="cuda", dtype=dtype)
    pair_mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)
    return s, z, pair_mask


def timed_step(name, fn, events) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events.append((name, start, end))
    return out


def run_block(
    block: PairformerBlock,
    s0: torch.Tensor,
    z0: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    layout_aware_ending: bool,
):
    s = s0.clone()
    z = z0.clone()
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    z = timed_step(
        "tri_mul_out",
        lambda: block.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=True,
            _add_with_inplace=True,
            triangle_multiplicative=args.triangle_multiplicative,
        ),
        events,
    )
    z = timed_step(
        "tri_mul_in",
        lambda: block.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=True,
            _add_with_inplace=True,
            triangle_multiplicative=args.triangle_multiplicative,
        ),
        events,
    )
    z = timed_step(
        "tri_att_start",
        lambda: z
        + block.tri_att_start(
            z,
            mask=pair_mask,
            triangle_attention=args.triangle_attention,
            inplace_safe=True,
            chunk_size=args.chunk_size,
        ),
        events,
    )

    if layout_aware_ending:
        z = timed_step(
            "tri_att_end_layout_aware",
            lambda: z
            + block.tri_att_end(
                z,
                mask=pair_mask,
                triangle_attention=args.triangle_attention,
                inplace_safe=True,
                chunk_size=args.chunk_size,
            ),
            events,
        )
    else:
        z = timed_step(
            "materialize_to_ending_layout",
            lambda: z.transpose(-2, -3).contiguous(),
            events,
        )
        z = timed_step(
            "tri_att_end",
            lambda: z
            + block.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2),
                triangle_attention=args.triangle_attention,
                inplace_safe=True,
                chunk_size=args.chunk_size,
            ),
            events,
        )
        z = timed_step(
            "materialize_from_ending_layout",
            lambda: z.transpose(-2, -3).contiguous(),
            events,
        )

    z = timed_step("pair_transition", lambda: z + block.pair_transition(z), events)
    s = timed_step(
        "token_attention",
        lambda: s + block.attention_pair_bias(s, None, z),
        events,
    )
    s = timed_step("single_transition", lambda: s + block.single_transition(s), events)

    torch.cuda.synchronize()
    return s, z, {name: start.elapsed_time(end) for name, start, end in events}


def measure(
    label: str,
    fn,
    args: argparse.Namespace,
) -> dict[str, object]:
    totals: dict[str, float] = defaultdict(float)
    for _ in range(args.warmup):
        fn()
    torch.cuda.reset_peak_memory_stats()
    last_s = last_z = None
    for _ in range(args.iters):
        last_s, last_z, timings = fn()
        for name, elapsed_ms in timings.items():
            totals[name] += elapsed_ms

    mean_ms = {name: total / args.iters for name, total in totals.items()}
    total_ms = sum(mean_ms.values())
    return {
        "name": label,
        "mean_ms": mean_ms,
        "total_ms": total_ms,
        "percent": {name: 100.0 * value / total_ms for name, value in mean_ms.items()},
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "_last_s": last_s,
        "_last_z": last_z,
    }


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return float(finite.max().item()) if finite.numel() else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--chunk-size", type=int, default=None)
    dtypes = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_ending_layout_screen requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    candidate = make_candidate_block(args)
    materialized = make_materialized_block(args, candidate, candidate.state_dict())
    s, z, pair_mask = make_inputs(args)

    with torch.inference_mode(), cuda_autocast(args.compute_dtype):
        materialized_row = measure(
            "materialized_transpose",
            lambda: run_block(materialized, s, z, pair_mask, args, layout_aware_ending=False),
            args,
        )
        ending_node_row = measure(
            "ending_node",
            lambda: run_block(candidate, s, z, pair_mask, args, layout_aware_ending=True),
            args,
        )

    materialized_s = materialized_row.pop("_last_s")
    materialized_z = materialized_row.pop("_last_z")
    ending_node_s = ending_node_row.pop("_last_s")
    ending_node_z = ending_node_row.pop("_last_z")

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "materialized_transpose": materialized_row,
        "ending_node": ending_node_row,
        "speedup": materialized_row["total_ms"] / ending_node_row["total_ms"],
        "parity": {
            "s_max_abs_error": max_abs_error(materialized_s, ending_node_s),
            "z_max_abs_error": max_abs_error(materialized_z, ending_node_z),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
