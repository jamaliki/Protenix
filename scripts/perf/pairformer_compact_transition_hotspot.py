#!/usr/bin/env python3
"""Gate compact valid-pair execution inside a full PairformerBlock.

`pair_transition` is row-wise over pair cells, so mixed-length batches do not
need to run that MLP on padded `(i, j)` rows.  This script measures the honest
block-level effect of the opt-in production path:

1. Run the normal dense PairformerBlock.
2. Run the same block with `PROTENIX_COMPACT_PAIR_TRANSITION=1`.
3. Compare only valid pair/token rows, because padded rows are implementation
   detail and are masked from later model reductions.

The result tells us whether the local transition win is large enough to justify
an end-to-end inference gate.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import torch

import protenix.model.modules.pairformer as pairformer_mod
from protenix.model.modules.pairformer import PairformerBlock


TensorPair = tuple[torch.Tensor, torch.Tensor]
BlockFn = Callable[[], TensorPair]


def parse_lengths(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated lengths")
    return values


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
def autocast(dtype_name: str):
    if dtype_name == "float32":
        with nullcontext():
            yield
    else:
        with torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name)):
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
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 17)
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    max_tokens = max(lengths)
    batch = len(lengths)
    s = torch.zeros(batch, max_tokens, args.c_s, device="cuda", dtype=dtype)
    z = torch.zeros(batch, max_tokens, max_tokens, args.c_z, device="cuda", dtype=dtype)
    pair_mask = torch.zeros(batch, max_tokens, max_tokens, device="cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for index, length in enumerate(lengths):
        s[index, :length] = torch.randn(
            length, args.c_s, device="cuda", dtype=dtype, generator=generator
        )
        z[index, :length, :length] = torch.randn(
            length, length, args.c_z, device="cuda", dtype=dtype, generator=generator
        )
        pair_mask[index, :length, :length] = 1
    return s, z, pair_mask


def flatten_valid(tensor: torch.Tensor, lengths: list[int], pair: bool) -> torch.Tensor:
    chunks = []
    for row, length in zip(tensor, lengths, strict=True):
        if pair:
            chunks.append(row[:length, :length].reshape(-1, row.shape[-1]))
        else:
            chunks.append(row[:length])
    return torch.cat(chunks, dim=0)


def run_block(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
    compact: bool,
) -> TensorPair:
    os.environ["PROTENIX_COMPACT_PAIR_TRANSITION"] = "1" if compact else "0"
    # The module caches environment parsing for production speed.  Reset it so
    # one process can benchmark both variants without re-importing the model.
    pairformer_mod._COMPACT_PAIR_TRANSITION = None  # pylint: disable=protected-access
    s_out, z_out = block(
        s.clone(),
        z.clone(),
        pair_mask=pair_mask,
        triangle_multiplicative=args.triangle_multiplicative,
        triangle_attention=args.triangle_attention,
        inplace_safe=True,
        chunk_size=args.chunk_size,
    )
    return (
        flatten_valid(s_out, args.lengths, pair=False),
        flatten_valid(z_out, args.lengths, pair=True),
    )


def cuda_time(fn: BlockFn, warmup: int, iters: int) -> tuple[TensorPair, float]:
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


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
    )
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_compact_transition_hotspot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    block = make_block(args)
    s, z, pair_mask = make_inputs(args.lengths, args)

    with autocast(args.compute_dtype):
        baseline, baseline_ms = cuda_time(
            lambda: run_block(block, s, z, pair_mask, args, compact=False),
            args.warmup,
            args.iters,
        )
        candidate, candidate_ms = cuda_time(
            lambda: run_block(block, s, z, pair_mask, args, compact=True),
            args.warmup,
            args.iters,
        )

    valid_pairs = sum(length * length for length in args.lengths)
    padded_pairs = len(args.lengths) * max(args.lengths) ** 2
    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "device": torch.cuda.get_device_name(),
                "shape": {
                    "batch": len(args.lengths),
                    "max_tokens": max(args.lengths),
                    "valid_pairs": valid_pairs,
                    "padded_pairs": padded_pairs,
                    "valid_pair_efficiency": valid_pairs / padded_pairs,
                },
                "results": {
                    "dense": {"ms": baseline_ms},
                    "compact_pair_transition": {
                        "ms": candidate_ms,
                        "speedup_vs_dense": baseline_ms / candidate_ms,
                        "s_valid_parity": compare(baseline[0], candidate[0]),
                        "z_valid_parity": compare(baseline[1], candidate[1]),
                    },
                },
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
