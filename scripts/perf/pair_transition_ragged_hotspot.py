#!/usr/bin/env python3
"""Screen compact valid-pair execution for pairformer Transition.

Mixed-length campaign batches pad each sorted batch to its largest token count.
Pair transition is row-wise over `(i, j)` pair features, so it is the safest
first screen for a future ragged pairformer layout.

The variants separate the ideal compact compute win from gather/scatter costs.
If only `compact_transition_only` wins, this local island is not enough: the
real boundary would be a larger ragged pairformer layout that avoids scattering
between blocks.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

# The Tokyo inference environment may not have the optional fast-layernorm
# extension prebuilt.  This screen is about pair-transition layout economics,
# not one-time extension compilation, so use PyTorch LayerNorm unless the caller
# deliberately overrides it.
os.environ.setdefault("LAYERNORM_TYPE", "normal")

from protenix.model.modules.fused_elementwise_triton import triton_silu_mul_inplace_y
from protenix.model.modules.primitives import Transition

TensorFn = Callable[[], torch.Tensor]


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


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item) for item in text.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive comma-separated ints")
    return lengths


def make_transition(args: argparse.Namespace) -> Transition:
    module = Transition(c_in=args.c_in, n=args.factor).cuda().eval()
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 0x9E3779B1)
        with torch.no_grad():
            for parameter in module.parameters():
                # The AF3 transition output projection is zero-initialized.  A
                # zero update is correct for training init, but it hides the
                # real GEMM and parity behavior in an inference microbenchmark.
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_padded_input(
    lengths: list[int],
    *,
    c_in: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    max_tokens = max(lengths)
    z = torch.zeros(len(lengths), max_tokens, max_tokens, c_in, device="cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for batch_idx, length in enumerate(lengths):
        shape = (length, length, c_in)
        z[batch_idx, :length, :length] = torch.randn(
            shape, device="cuda", dtype=dtype, generator=generator
        )
    return z


def make_valid_index(lengths: list[int]) -> torch.Tensor:
    max_tokens = max(lengths)
    chunks = []
    for batch_idx, length in enumerate(lengths):
        row = torch.arange(length, device="cuda")
        i, j = torch.meshgrid(row, row, indexing="ij")
        chunks.append((batch_idx * max_tokens * max_tokens + i * max_tokens + j).reshape(-1))
    return torch.cat(chunks)


def transition_rows_no_chunk(module: Transition, x: torch.Tensor) -> torch.Tensor:
    """Run the same inference math as Transition.forward without shape chunking.

    `Transition.forward()` chunks based on `x.shape[-2]` before flattening.  For
    normal `[B, N, N, C]`, this is just `N`; for compact `[valid_pairs, C]`, it
    would be the full row count.  A real compact pairformer boundary would use
    one row-wise GEMM, so this helper avoids that artifact.
    """
    y = module.layernorm1(x)
    b = module._fused_input_projection(y)
    if b is None:
        a = module.linear_no_bias_a(y)
        b = module.linear_no_bias_b(y)
        fused_b = triton_silu_mul_inplace_y(a, b)
        if fused_b is None:
            a = F.silu(a, inplace=True)
            b *= a
        else:
            b = fused_b
    return module.linear_no_bias(b)

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


def profiler_rows(fn: TensorFn, warmup: int, limit: int) -> list[dict[str, object]]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities, record_shapes=True, with_stack=False
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


def max_mean_abs(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
        help="Comma-separated token lengths in one sorted campaign batch.",
    )
    parser.add_argument("--c-in", type=int, default=128)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--compute-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=12)
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    parser.add_argument("--fused-elementwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-input-projection", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ["PROTENIX_TRITON_FUSED_ELEMENTWISE"] = (
        "1" if args.fused_elementwise else "0"
    )
    os.environ["PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION"] = (
        "1" if args.fused_input_projection else "0"
    )
    os.environ.setdefault("PROTENIX_FUSED_TRANSITION_INPUT_MAX_ELEMENTS", "4000000000")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    dtype = getattr(torch, args.input_dtype)
    module = make_transition(args)
    z = make_padded_input(args.lengths, c_in=args.c_in, dtype=dtype, seed=args.seed)
    flat = z.reshape(-1, args.c_in)
    valid_index = make_valid_index(args.lengths)
    compact = flat.index_select(0, valid_index)

    def padded_full() -> torch.Tensor:
        return z + module(z)

    def compact_transition_only() -> torch.Tensor:
        return compact + transition_rows_no_chunk(module, compact)

    def compact_gather_transition() -> torch.Tensor:
        gathered = flat.index_select(0, valid_index)
        return gathered + transition_rows_no_chunk(module, gathered)

    def compact_round_trip() -> torch.Tensor:
        gathered = flat.index_select(0, valid_index)
        out = gathered + transition_rows_no_chunk(module, gathered)
        out_flat = torch.zeros_like(flat)
        out_flat.index_copy_(0, valid_index, out)
        return out_flat.reshape_as(z)

    variants: dict[str, TensorFn] = {
        "padded_full": padded_full,
        "compact_transition_only": compact_transition_only,
        "compact_gather_transition": compact_gather_transition,
        "compact_round_trip": compact_round_trip,
    }

    with cuda_autocast(args.compute_dtype), torch.inference_mode():
        reference = padded_full()
        reference_valid = reference.reshape(-1, args.c_in).index_select(0, valid_index)
        results: dict[str, Any] = {}
        torch.cuda.reset_peak_memory_stats()
        for name, fn in variants.items():
            out, ms = cuda_time(fn, args.warmup, args.iters)
            if out.shape == reference.shape:
                candidate_valid = out.reshape(-1, args.c_in).index_select(0, valid_index)
            else:
                candidate_valid = out
            results[name] = {
                "ms": ms,
                "speedup_vs_padded": None,
                "parity_valid_pairs": max_mean_abs(reference_valid, candidate_valid),
            }

        padded_ms = results["padded_full"]["ms"]
        for row in results.values():
            row["speedup_vs_padded"] = padded_ms / row["ms"]

        if args.profile:
            results["profiler_top_cuda"] = {
                name: profiler_rows(fn, args.warmup, args.profile_top)
                for name, fn in variants.items()
            }

    valid_pairs = sum(length * length for length in args.lengths)
    padded_pairs = len(args.lengths) * max(args.lengths) ** 2
    row = {
        "args": {
            **vars(args),
            "lengths": args.lengths,
        },
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": len(args.lengths),
            "max_tokens": max(args.lengths),
            "valid_pairs": valid_pairs,
            "padded_pairs": padded_pairs,
            "valid_pair_efficiency": valid_pairs / padded_pairs,
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
