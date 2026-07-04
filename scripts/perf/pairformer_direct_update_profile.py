#!/usr/bin/env python3
"""Profile the direct producer-owned triangle-update PairformerBlock path.

The direct update gate is now positive but small on the long Protenix-v2 bucket.
This script records Chrome traces for the baseline CUEQ block and the direct
producer-owned block with explicit stage scopes.  Use it to decide whether the
next useful boundary is producer/contraction/output fusion, invalid-row handling,
triangle attention, or pair transition.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from pairformer_compact_segmented_triangle_update_probe import (
    TriangleUpdatePlan,
    compact_triangle_update,
    make_triangle_update_plan,
)
from pairformer_varlen_triangle_attention_probe import (
    TensorPair,
    autocast_context,
    cuda_time,
    finish_block,
    make_block,
    make_inputs,
    parse_ints,
    run_baseline,
    str_bool,
)
from scripts.perf.triangle_update_compact_segmented_probe import ContractConfig

LONG_LENGTHS = "136,136,160,160,172,172,184,184,196,196,208,208,216,216,220,220"


def clone_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.clone()


def run_direct_scoped(
    block,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: list[int],
    config: ContractConfig,
    plan: TriangleUpdatePlan,
) -> TensorPair:
    with record_function("direct/clone_inputs"):
        s = clone_optional(s)
        z = z.clone()
    with record_function("direct/triangle_mul_out"):
        z = compact_triangle_update(
            block.tri_mul_out,
            z,
            mask,
            lengths,
            config,
            "outgoing",
            "direct",
            plan.weights_out,
            plan.dense_offsets,
            plan.descriptors,
            plan.invalid_offsets,
        )
    with record_function("direct/triangle_mul_in"):
        z = compact_triangle_update(
            block.tri_mul_in,
            z,
            mask,
            lengths,
            config,
            "incoming",
            "direct",
            plan.weights_in,
            plan.dense_offsets,
            plan.descriptors,
            plan.invalid_offsets,
        )
    with record_function("direct/triangle_attention_start"):
        z += block.tri_att_start(
            z,
            mask=mask,
            triangle_attention="cuequivariance",
            inplace_safe=True,
            chunk_size=None,
        )
    with record_function("direct/triangle_attention_end"):
        z += block.tri_att_end._cueq_ending_contiguous_forward(z, mask)
    with record_function("direct/finish_block"):
        return finish_block(block, s, z, mask)


def profile_case(
    name: str,
    fn: Callable[[], TensorPair],
    out_dir: Path,
    warmup: int,
    iters: int,
    top_rows: int,
) -> dict[str, str | float]:
    _out, timing = cuda_time(fn, warmup, iters)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()
            prof.step()

    trace = out_dir / f"{name}_trace.json"
    top = out_dir / f"{name}_top_cuda.txt"
    prof.export_chrome_trace(str(trace))
    top.write_text(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=top_rows) + "\n",
        encoding="utf-8",
    )
    return {
        "ms": timing["ms"],
        "mean_ms": timing["mean_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
        "trace": str(trace),
        "top_cuda": str(top),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_ints, default=parse_ints(LONG_LENGTHS))
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--config", type=ContractConfig.parse, default=ContractConfig.parse("32x32x64x1x4x3"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument("--top-rows", type=int, default=60)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["PROTENIX_PROFILE_PAIRFORMER_SCOPES"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block = make_block(args)
    s, z, mask = make_inputs(args.lengths, args)
    plan = make_triangle_update_plan(block, args.lengths, args.config, "direct")

    with torch.inference_mode(), autocast_context(args.input_dtype):
        baseline = profile_case(
            "baseline_cueq_block",
            lambda: run_baseline(block, s, z, mask),
            out_dir,
            args.warmup,
            args.iters,
            args.top_rows,
        )
        direct = profile_case(
            "direct_block",
            lambda: run_direct_scoped(block, s, z, mask, args.lengths, args.config, plan),
            out_dir,
            args.warmup,
            args.iters,
            args.top_rows,
        )

    summary = {
        "args": {**vars(args), "config": args.config.label()},
        "baseline": baseline,
        "direct": direct,
        "speedup": baseline["ms"] / direct["ms"],
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    (out_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
