#!/usr/bin/env python3
"""CUDA-event breakdown for one PairformerBlock.

The full inference timing showed that high-B, low-sample throughput is now
limited by pairformer.  This script times the block's real sequence of
triangle/pair/single operations so kernel work can target the dominant boundary
instead of the already-fast diffusion path.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator

import torch

from protenix.model.modules.pairformer import PairformerBlock
from protenix.model.triangular.triangular import (
    cueq_ending_contiguous_producer_enabled,
)


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
def cuda_autocast(dtype_name: str) -> Iterator[None]:
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
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
        args.batch, args.tokens, args.tokens, args.c_z, device=device, dtype=dtype
    )
    mask_dtype = torch.bool if args.pair_mask_dtype == "bool" else dtype
    pair_mask = torch.ones(
        args.batch, args.tokens, args.tokens, device=device, dtype=mask_dtype
    )
    return s, z, pair_mask


def timed_step(
    name: str,
    fn: Callable[[], torch.Tensor],
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events.append((name, start, end))
    return out


def cuda_profiler_capture(
    name: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, dict[str, float]]],
    *,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a narrow CUDA-profiler capture for Nsight Compute.

    CUDA events rank the PairformerBlock's Python-visible stages, but the next
    same-token/variable-atom bottleneck is a kernel-boundary question.  Launch
    NCU with ``--profile-from-start off`` and it will ignore setup work until
    ``cudaProfilerStart`` is called here.  Capturing only the warmed block keeps
    the report focused enough to decide whether the next kernel should target
    attention wrapper traffic, triangle multiplication, transitions, or copies.
    """
    for _ in range(warmup):
        s_out, z_out, _ = fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(name)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        s_out, z_out, _ = fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    return s_out, z_out


def run_once(
    block: PairformerBlock,
    s0: torch.Tensor,
    z0: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    s = s0.clone()
    z = z0.clone()
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    use_contiguous_ending = (
        args.triangle_attention == "cuequivariance"
        and args.chunk_size is None
        and cueq_ending_contiguous_producer_enabled()
    )

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
    if use_contiguous_ending:
        z = timed_step(
            "tri_att_end_contiguous_producer",
            lambda: z
            + block.tri_att_end._cueq_ending_contiguous_forward(z, pair_mask),
            events,
        )
    else:
        z = timed_step(
            "transpose_after_start",
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
            "transpose_after_end",
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
    timings = {name: start.elapsed_time(end) for name, start, end in events}
    return s, z, timings


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
    dtype_choices = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--pair-mask-dtype", choices=["same", "bool"], default="same")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--ncu-capture",
        action="store_true",
        help=(
            "Run only a warmed PairformerBlock CUDA-profiler capture for Nsight "
            "Compute. Launch NCU with --profile-from-start off to collect just "
            "this range."
        ),
    )
    parser.add_argument(
        "--ncu-iters",
        type=int,
        default=1,
        help="PairformerBlock calls inside the CUDA-profiler capture range.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_block_breakdown requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)
    block = make_block(args)
    s, z, pair_mask = make_inputs(args)

    totals: dict[str, float] = defaultdict(float)
    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        if args.ncu_capture:
            s_out, z_out = cuda_profiler_capture(
                f"pairformer_block_B{args.batch}_N{args.tokens}",
                lambda: run_once(block, s, z, pair_mask, args),
                warmup=args.warmup,
                iters=args.ncu_iters,
            )
            print(
                json.dumps(
                    {
                        "args": vars(args),
                        "capture_shapes": {
                            "s": list(s_out.shape),
                            "z": list(z_out.shape),
                        },
                        "device": torch.cuda.get_device_name(),
                        "torch": {
                            "version": torch.__version__,
                            "cuda": torch.version.cuda,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        for _ in range(args.warmup):
            run_once(block, s, z, pair_mask, args)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            _, _, timings = run_once(block, s, z, pair_mask, args)
            for name, elapsed_ms in timings.items():
                totals[name] += elapsed_ms

    mean_ms = {name: total / args.iters for name, total in totals.items()}
    total_ms = sum(mean_ms.values())
    row = {
        "args": vars(args),
        "mean_ms": mean_ms,
        "total_ms": total_ms,
        "percent": {name: 100.0 * value / total_ms for name, value in mean_ms.items()},
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
