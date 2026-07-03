#!/usr/bin/env python3
"""Screen CUDA graph replay at the diffusion-token transformer boundary.

CUDA graphs remove Python dispatch and per-op launch setup from repeated
fixed-shape work.  That is attractive for Protenix inference because the same
denoising shape is replayed for many diffusion steps.  The catch is that a
graph can only replay against the same tensor addresses: if we capture only the
token transformer, changing inputs must first be copied into static buffers.

This probe therefore reports three progressively more realistic cases:

* eager: current PyTorch execution;
* graph_replay_only: the best possible boundary win if upstream producers wrote
  directly into the graph's static input buffers;
* graph_copy_changed_inputs: replay plus device copies for `a`, `s`, and `z`,
  which change across denoising steps in the current model boundary.

The input layout matches the promoted mixed-sequence, low-sample inference path:
`a` and `s` are flattened over record/sample lanes, while `z` remains at record
grain so each block projects the pair bias once per protein.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import time
from collections.abc import Iterable
from typing import Any

import torch

from diffusion_transformer_compile_probe import (
    FlatDiffusionTransformer,
    _compare,
    _dtype,
    _make_inputs,
)
from protenix.model.modules.transformer import DiffusionTransformer


def _autocast(dtype: torch.dtype) -> torch.amp.autocast_mode.autocast:
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=dtype != torch.float32,
    )


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def _time_eager(
    wrapper: FlatDiffusionTransformer,
    inputs: dict[str, torch.Tensor],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        with _autocast(dtype):
            out = wrapper(**inputs)
        for _ in range(warmup):
            with _autocast(dtype):
                out = wrapper(**inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            with _autocast(dtype):
                out = wrapper(**inputs)
        end.record()
        end.synchronize()
    return out, start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _capture_graph(
    wrapper: FlatDiffusionTransformer,
    static_inputs: dict[str, torch.Tensor],
    dtype: torch.dtype,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, float]:
    torch.cuda.reset_peak_memory_stats()

    # Warmup must happen on a side stream before capture so allocator and cuDNN
    # workspaces are materialized outside the graph's private memory pool.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    warmup_start = time.perf_counter()
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(warmup):
            with _autocast(dtype):
                out = wrapper(**static_inputs)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        with _autocast(dtype):
            out = wrapper(**static_inputs)
    torch.cuda.synchronize()
    return graph, out, time.perf_counter() - warmup_start


def _copy_inputs(
    static_inputs: dict[str, torch.Tensor],
    source_inputs: dict[str, torch.Tensor],
    keys: Iterable[str],
) -> None:
    for key in keys:
        static_inputs[key].copy_(source_inputs[key])


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    output: torch.Tensor,
    static_inputs: dict[str, torch.Tensor],
    source_inputs: dict[str, torch.Tensor],
    copy_keys: tuple[str, ...],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            _copy_inputs(static_inputs, source_inputs, copy_keys)
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _copy_inputs(static_inputs, source_inputs, copy_keys)
            graph.replay()
        end.record()
        end.synchronize()
    return output, start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _case_row(
    name: str,
    mean_ms: float,
    peak_mib: float,
    reference_ms: float,
    output: torch.Tensor,
    reference: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "speedup_vs_eager": reference_ms / mean_ms,
        "peak_allocated_mib": peak_mib,
        "parity_vs_eager": _compare(output, reference),
    }
    if extra:
        row.update(extra)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--graph-warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_transformer_cudagraph_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = _dtype(args.dtype)

    model = DiffusionTransformer(
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        n_blocks=args.blocks,
        n_heads=args.heads,
    ).cuda().eval()
    wrapper = FlatDiffusionTransformer(model, samples=args.samples).cuda().eval()
    inputs = _make_inputs(args)

    eager_out, eager_ms, eager_peak = _time_eager(
        wrapper=wrapper,
        inputs=inputs,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
    )
    rows: list[dict[str, Any]] = [
        {
            "name": "eager",
            "ok": True,
            "mean_ms": eager_ms,
            "speedup_vs_eager": 1.0,
            "peak_allocated_mib": eager_peak,
            "output_shape": list(eager_out.shape),
            "output_dtype": str(eager_out.dtype),
        }
    ]

    static_inputs = _clone_inputs(inputs)
    source_inputs = _make_inputs(args)
    try:
        graph, graph_out, capture_sec = _capture_graph(
            wrapper=wrapper,
            static_inputs=static_inputs,
            dtype=dtype,
            warmup=args.graph_warmup,
        )
        graph_out, graph_ms, graph_peak = _time_graph(
            graph=graph,
            output=graph_out,
            static_inputs=static_inputs,
            source_inputs=source_inputs,
            copy_keys=(),
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(
            _case_row(
                "graph_replay_only",
                graph_ms,
                graph_peak,
                eager_ms,
                graph_out,
                eager_out,
                {"capture_sec": capture_sec},
            )
        )

        changed_keys = ("a", "s", "z")
        copy_out, copy_ms, copy_peak = _time_graph(
            graph=graph,
            output=graph_out,
            static_inputs=static_inputs,
            source_inputs=source_inputs,
            copy_keys=changed_keys,
            warmup=args.warmup,
            iters=args.iters,
        )
        reference_after_copy, _, _ = _time_eager(
            wrapper=wrapper,
            inputs={**inputs, **{key: source_inputs[key] for key in changed_keys}},
            dtype=dtype,
            warmup=1,
            iters=1,
        )
        rows.append(
            _case_row(
                "graph_copy_changed_inputs",
                copy_ms,
                copy_peak,
                eager_ms,
                copy_out,
                reference_after_copy,
                {"copy_keys": changed_keys},
            )
        )
    except Exception as exc:  # noqa: BLE001 - report graph failures as data.
        rows.append({"name": "cuda_graph", "ok": False, "error": repr(exc)})

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "cases": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
