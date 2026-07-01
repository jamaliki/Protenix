#!/usr/bin/env python3
"""Screen CUDA Graph replay for static-shape diffusion trunk hotspots."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from typing import Callable

import torch

from scripts.perf.trunk_compile_hotspot import compare, make_inputs, make_target


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


def capture_cuda_graph(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(warmup):
            static_out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_out = fn()
    capture_sec = time.perf_counter() - start

    def replay() -> torch.Tensor:
        graph.replay()
        return static_out

    return replay, static_out, capture_sec


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
    parser.add_argument("--graph-warmup", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--efficient-fusion", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("trunk_cuda_graph_hotspot requires CUDA")

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
    graph_run, graph_out, capture_sec = capture_cuda_graph(
        eager_run,
        warmup=args.graph_warmup,
    )
    graph_out, graph_ms = cuda_time(
        graph_run,
        warmup=args.graph_warmup,
        iters=args.graph_iters,
    )
    row = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "input_shapes": [list(t.shape) for t in inputs],
        "eager_ms": eager_ms,
        "cuda_graph_ms": graph_ms,
        "speedup": eager_ms / graph_ms,
        "capture_sec": capture_sec,
        "parity": compare(graph_out, eager_out),
        "eager_all_finite": bool(torch.isfinite(eager_out).all().item()),
        "graph_all_finite": bool(torch.isfinite(graph_out).all().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "timestamp": time.time(),
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
