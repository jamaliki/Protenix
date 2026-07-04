#!/usr/bin/env python3
"""Screen CUDA graph replay at the wide Protenix-v2 Pairformer boundary.

Sam-style inference is a low-sample, many-record workload.  After batching and
the H100 CUEQ cache overlay, the remaining v2 bottleneck is real wide-``z``
Pairformer work.  This probe checks whether CUDA graphs can remove enough
Python/launch overhead to justify graph plumbing before we commit to a larger
CuTe triangle-update kernel.

CUDA graphs replay against fixed tensor addresses.  The useful question is not
only "is graph replay faster?", but "is it still faster after changed inputs are
copied into graph-owned buffers?"  The reported rows separate the optimistic
no-copy ceiling from the realistic ``s/z`` copy boundary and the conservative
``s/z/mask`` copy boundary.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import time
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from protenix.model.modules.pairformer import PairformerStack

TensorDict = dict[str, torch.Tensor]
Output = tuple[torch.Tensor, torch.Tensor]
ALL_KEYS = ("s", "z", "pair_mask")


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("expected positive comma-separated lengths")
    return lengths


@contextmanager
def autocast(dtype_name: str):
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def make_stack(args: argparse.Namespace) -> PairformerStack:
    stack = PairformerStack(
        n_blocks=args.blocks,
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        blocks_per_ckpt=None,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda().eval()

    # Synthetic-only: zero output projections hide parity mistakes and can
    # perturb autotuning.  Randomizing them keeps the screen representative
    # without changing production initialization.
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 7919)
            for parameter in stack.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return stack


def make_inputs(args: argparse.Namespace) -> TensorDict:
    dtype = getattr(torch, args.input_dtype)
    mask_dtype = torch.bool if args.pair_mask_dtype == "bool" else dtype
    s = torch.zeros(args.batch, args.tokens, args.c_s, device="cuda", dtype=dtype)
    z = torch.zeros(
        args.batch, args.tokens, args.tokens, args.c_z, device="cuda", dtype=dtype
    )
    pair_mask = torch.zeros(
        args.batch, args.tokens, args.tokens, device="cuda", dtype=mask_dtype
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for index, length in enumerate(args.lengths):
        s[index, :length] = torch.randn(
            length, args.c_s, device="cuda", dtype=dtype, generator=generator
        )
        z[index, :length, :length] = torch.randn(
            length, length, args.c_z, device="cuda", dtype=dtype, generator=generator
        )
        pair_mask[index, :length, :length] = True if mask_dtype == torch.bool else 1
    return {"s": s, "z": z, "pair_mask": pair_mask}


def clone_inputs(inputs: TensorDict) -> TensorDict:
    return {key: tensor.clone() for key, tensor in inputs.items()}


def copy_inputs(dst: TensorDict, src: TensorDict, keys: Iterable[str]) -> None:
    for key in keys:
        dst[key].copy_(src[key])


def run_stack(stack: PairformerStack, inputs: TensorDict, args: argparse.Namespace) -> Output:
    return stack(
        inputs["s"],
        inputs["z"],
        pair_mask=inputs["pair_mask"],
        triangle_multiplicative=args.triangle_multiplicative,
        triangle_attention=args.triangle_attention,
        inplace_safe=args.inplace_safe,
        chunk_size=args.chunk_size,
    )


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (reference.float() - candidate.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
    }


def parity(reference: Output, candidate: Output) -> dict[str, dict[str, float]]:
    return {
        "s": diff_stats(reference[0], candidate[0]),
        "z": diff_stats(reference[1], candidate[1]),
    }


def reference_once(stack: PairformerStack, source: TensorDict, args: argparse.Namespace) -> Output:
    with torch.inference_mode(), autocast(args.compute_dtype):
        return run_stack(stack, clone_inputs(source), args)


def time_eager(
    stack: PairformerStack,
    source: TensorDict,
    args: argparse.Namespace,
    copy_keys: tuple[str, ...],
) -> tuple[Output, float, float]:
    work = clone_inputs(source)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), autocast(args.compute_dtype):
        for _ in range(args.warmup):
            copy_inputs(work, source, copy_keys)
            output = run_stack(stack, work, args)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(args.iters):
            copy_inputs(work, source, copy_keys)
            output = run_stack(stack, work, args)
        end.record()
        end.synchronize()
    return output, start.elapsed_time(end) / args.iters, torch.cuda.max_memory_allocated() / 2**20


def capture_graph(
    stack: PairformerStack,
    static: TensorDict,
    source: TensorDict,
    args: argparse.Namespace,
) -> tuple[torch.cuda.CUDAGraph, Output, float]:
    # Side-stream warmup keeps allocator, cuDNN, and CUEQ autotune setup out of
    # the graph pool.  Inputs are reset before every warmup to avoid numerical
    # drift from inplace Pairformer updates.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = time.perf_counter()
    with torch.cuda.stream(stream), torch.inference_mode(), autocast(args.compute_dtype):
        for _ in range(args.graph_warmup):
            copy_inputs(static, source, ALL_KEYS)
            run_stack(stack, static, args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    copy_inputs(static, source, ALL_KEYS)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph), autocast(args.compute_dtype):
        output = run_stack(stack, static, args)
    torch.cuda.synchronize()
    return graph, output, time.perf_counter() - start


def time_graph(
    graph: torch.cuda.CUDAGraph,
    output: Output,
    static: TensorDict,
    source: TensorDict,
    args: argparse.Namespace,
    copy_keys: tuple[str, ...],
) -> tuple[Output, float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.warmup):
            copy_inputs(static, source, copy_keys)
            graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(args.iters):
            copy_inputs(static, source, copy_keys)
            graph.replay()
        end.record()
        end.synchronize()
    return output, start.elapsed_time(end) / args.iters, torch.cuda.max_memory_allocated() / 2**20


def result_row(
    name: str,
    output: Output,
    mean_ms: float,
    peak_mib: float,
    *,
    copy_keys: tuple[str, ...],
    reference_ms: float | None = None,
    reference: Output | None = None,
    capture_sec: float | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "copy_keys": list(copy_keys),
        "peak_allocated_mib": peak_mib,
        "output_shape": [list(tensor.shape) for tensor in output],
    }
    if reference_ms is not None:
        item["speedup_vs_reference"] = reference_ms / mean_ms
    if reference is not None:
        item["parity_vs_eager_reset"] = parity(reference, output)
    else:
        item["parity_note"] = "not meaningful because inplace replay starts from mutated z"
    if capture_sec is not None:
        item["capture_sec"] = capture_sec
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("40,40,52,52,64,64,76,76,88,88,100,100,112,112,124,124"),
    )
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--inplace-safe", type=str_bool, default=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--input-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--pair-mask-dtype", choices=["same", "bool"], default="same")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--graph-warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.batch = len(args.lengths)
    args.tokens = max(args.lengths)
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_cudagraph_probe requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.backends.cudnn.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    stack = make_stack(args)
    source = make_inputs(args)
    reference = reference_once(stack, source, args)

    eager_specs = {
        "eager_no_reset": (),
        "eager_copy_s_z": ("s", "z"),
        "eager_copy_all_inputs": ALL_KEYS,
    }
    cases: list[dict[str, Any]] = []
    eager_ms: dict[str, float] = {}
    for name, keys in eager_specs.items():
        output, mean_ms, peak_mib = time_eager(stack, source, args, keys)
        eager_ms[name] = mean_ms
        cases.append(
            result_row(
                name,
                output,
                mean_ms,
                peak_mib,
                copy_keys=keys,
                reference=reference if keys else None,
            )
        )

    try:
        graph, graph_output, capture_sec = capture_graph(stack, clone_inputs(source), source, args)
        graph_specs = {
            "graph_replay_only": ((), "eager_no_reset"),
            "graph_copy_s_z": (("s", "z"), "eager_copy_s_z"),
            "graph_copy_all_inputs": (ALL_KEYS, "eager_copy_all_inputs"),
        }
        for name, (keys, eager_name) in graph_specs.items():
            output, mean_ms, peak_mib = time_graph(graph, graph_output, clone_inputs(source), source, args, keys)
            cases.append(
                result_row(
                    name,
                    output,
                    mean_ms,
                    peak_mib,
                    copy_keys=keys,
                    reference_ms=eager_ms[eager_name],
                    reference=reference if keys else None,
                    capture_sec=capture_sec if name == "graph_replay_only" else None,
                )
            )
    except Exception as exc:  # noqa: BLE001 - graph failures are useful evidence.
        cases.append({"name": "cuda_graph_capture", "ok": False, "error": repr(exc)})

    print(
        json.dumps(
            {
                "args": vars(args),
                "input_shapes": {key: list(value.shape) for key, value in source.items()},
                "pair_token_efficiency": sum(length * length for length in args.lengths)
                / (args.batch * args.tokens * args.tokens),
                "cases": cases,
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
