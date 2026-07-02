#!/usr/bin/env python3
"""Benchmark the transition output-GEMM epilogue boundary.

This is the smallest reusable screen for the next native CuTe/CUTLASS attempt.
The target expression in ``ConditionedTransitionBlock`` is:

    projected = b @ W_out.T
    out = sigmoid(gate) * projected + residual

PyTorch/cuBLAS already computes the GEMM very well, and the promoted Triton
kernel already fuses the elementwise gate/residual expression.  A custom kernel
is only worth keeping if it preserves cuBLAS-class GEMM speed while folding the
gate/residual epilogue into the GEMM store.  This harness times the baseline
subranges and can compare an injected candidate op with exactly the same
fixtures.

Pass ``--candidate path_or_module:function`` to compare a custom op with
signature:

    fn(b, weight, gate, residual) -> out

where ``b`` is ``[..., hidden]``, ``weight`` is ``[c_a, hidden]``, and
``gate`` may either match the output shape or have first dimension ``1`` for
the diffusion sample-broadcast case.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the screen focused on the transition boundary, not layernorm setup.
os.environ.setdefault("LAYERNORM_TYPE", "normal")

TensorFn = Callable[[], torch.Tensor]
CandidateFn = Callable[..., torch.Tensor]


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


@contextmanager
def env_flag(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def load_candidate(spec: str) -> CandidateFn:
    module_name, sep, function_name = spec.partition(":")
    if not sep or not module_name or not function_name:
        raise ValueError("--candidate must be path_or_module:function")

    path = Path(module_name)
    if path.exists():
        import_name = f"_protenix_epilogue_candidate_{abs(hash(path.resolve()))}"
        module_spec = importlib.util.spec_from_file_location(import_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not import candidate file {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


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


def event_time(
    name: str,
    fn: TensorFn,
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events.append((name, start, end))
    return out


def make_block(args: argparse.Namespace) -> Any:
    from protenix.model.modules.transformer import ConditionedTransitionBlock

    block = ConditionedTransitionBlock(c_a=args.c_a, c_s=args.c_s, n=args.factor).cuda()
    block.eval()
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 17)
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_fixture(block: Any, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    hidden = args.factor * args.c_a
    common = {"device": "cuda", "dtype": dtype, "generator": generator}
    b = torch.randn(args.samples, args.tokens, hidden, **common)
    residual = torch.randn(args.samples, args.tokens, args.c_a, **common)
    s = torch.randn(args.gate_batch, args.tokens, args.c_s, **common)

    # The actual block computes this gate before the output projection.  Keep it
    # fixed so the custom kernel screen only measures GEMM epilogue work.
    gate = block.linear_s(s)
    return {
        "b": b,
        "weight": block.linear_nobias_b.weight,
        "gate": gate,
        "residual": residual,
    }


def baseline_output(
    b: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    from protenix.model.modules.fused_elementwise_triton import fused_sigmoid_mul_add

    projected = F.linear(b, weight)
    with env_flag("PROTENIX_TRITON_FUSED_ELEMENTWISE", "1"):
        return fused_sigmoid_mul_add(gate.contiguous(), projected, residual)


def baseline_subranges(
    b: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    from protenix.model.modules.fused_elementwise_triton import fused_sigmoid_mul_add

    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        projected = event_time(
            "output_projection_gemm", lambda: F.linear(b, weight), events
        )
        with env_flag("PROTENIX_TRITON_FUSED_ELEMENTWISE", "1"):
            out = event_time(
                "sigmoid_mul_add_epilogue",
                lambda: fused_sigmoid_mul_add(gate.contiguous(), projected, residual),
                events,
            )
        torch.cuda.synchronize()
    return out, {name: start.elapsed_time(end) for name, start, end in events}


def call_candidate(candidate: CandidateFn, fixture: dict[str, torch.Tensor]) -> torch.Tensor:
    return candidate(
        fixture["b"], fixture["weight"], fixture["gate"], fixture["residual"]
    )


def max_mean_abs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def boundary_byte_model(
    fixture: dict[str, torch.Tensor],
    output_bytes: int,
) -> dict[str, int]:
    b = fixture["b"]
    weight = fixture["weight"]
    gate = fixture["gate"]
    residual = fixture["residual"]
    projected_bytes = output_bytes

    # These are lower bounds on HBM traffic, not profiler replacements.  They
    # count each tensor once per logical boundary and intentionally ignore L2
    # reuse, implicit autocast weight packing, and cuBLAS workspace.
    epilogue_min_bytes = (
        projected_bytes
        + tensor_nbytes(gate)
        + tensor_nbytes(residual)
        + output_bytes
    )
    return {
        "b": tensor_nbytes(b),
        "weight_storage": tensor_nbytes(weight),
        "gate": tensor_nbytes(gate),
        "residual": tensor_nbytes(residual),
        "projected": projected_bytes,
        "output": output_bytes,
        "baseline_boundary_min": (
            tensor_nbytes(b)
            + tensor_nbytes(weight)
            + projected_bytes
            + epilogue_min_bytes
        ),
        "fused_boundary_min": (
            tensor_nbytes(b)
            + tensor_nbytes(weight)
            + tensor_nbytes(gate)
            + tensor_nbytes(residual)
            + output_bytes
        ),
        "standalone_epilogue_min": epilogue_min_bytes,
        "projected_store_reload_removable": 2 * projected_bytes,
    }


def rate_metrics(
    *,
    gemm_flops: int,
    byte_model: dict[str, int],
    baseline_ms: float,
    subranges: dict[str, float],
) -> dict[str, Any]:
    rates: dict[str, Any] = {}
    gemm_ms = subranges.get("output_projection_gemm")
    epilogue_ms = subranges.get("sigmoid_mul_add_epilogue")
    if gemm_ms and gemm_ms > 0:
        rates["gemm_effective_tflops"] = gemm_flops / (gemm_ms * 1e9)
    if epilogue_ms and epilogue_ms > 0:
        rates["epilogue_effective_bandwidth_tb_s"] = (
            byte_model["standalone_epilogue_min"] / (epilogue_ms * 1e9)
        )
        rates["ideal_delete_epilogue_launch"] = {
            "ms": baseline_ms - epilogue_ms,
            "speedup": baseline_ms / max(baseline_ms - epilogue_ms, 1e-9),
        }
    if baseline_ms > 0:
        rates["boundary_effective_tflops"] = gemm_flops / (baseline_ms * 1e9)
        rates["boundary_effective_bandwidth_tb_s_lower_bound"] = (
            byte_model["baseline_boundary_min"] / (baseline_ms * 1e9)
        )
    return rates


def problem_metrics(
    fixture: dict[str, torch.Tensor],
    reference: torch.Tensor,
    baseline_ms: float,
    subranges: dict[str, float],
) -> dict[str, Any]:
    b = fixture["b"]
    weight = fixture["weight"]
    gate = fixture["gate"]
    residual = fixture["residual"]
    samples, tokens, hidden = b.shape
    output_channels = weight.shape[0]
    rows = samples * tokens
    output_elements = rows * output_channels
    output_bytes = output_elements * reference.element_size()
    gemm_flops = 2 * rows * output_channels * hidden
    byte_model = boundary_byte_model(fixture, output_bytes)

    return {
        "shape": {
            "m_rows": rows,
            "n_output": output_channels,
            "k_hidden": hidden,
            "samples": samples,
            "tokens": tokens,
            "gate_batch": gate.shape[0],
        },
        "dtypes": {
            "b": str(b.dtype),
            "weight_storage": str(weight.dtype),
            "gate": str(gate.dtype),
            "residual": str(residual.dtype),
            "output": str(reference.dtype),
        },
        "gemm_flops": gemm_flops,
        "bytes_lower_bound": byte_model,
        **rate_metrics(
            gemm_flops=gemm_flops,
            byte_model=byte_model,
            baseline_ms=baseline_ms,
            subranges=subranges,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--gate-batch", type=int, default=1)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument(
        "--input-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--candidate")
    parser.add_argument("--output-json")
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    candidate = load_candidate(args.candidate) if args.candidate else None
    block = make_block(args)

    with cuda_autocast(args.compute_dtype):
        fixture = make_fixture(block, args)
        reference, subranges = baseline_subranges(**fixture)
        _, baseline_ms = cuda_time(
            lambda: baseline_output(**fixture),
            warmup=args.warmup,
            iters=args.iters,
        )

        candidate_row: dict[str, Any] | None = None
        if candidate is not None:
            candidate_out, candidate_ms = cuda_time(
                lambda: call_candidate(candidate, fixture),
                warmup=args.warmup,
                iters=args.iters,
            )
            candidate_row = {
                "ms": candidate_ms,
                "speedup_vs_baseline": baseline_ms / candidate_ms,
                "parity": max_mean_abs(reference, candidate_out),
            }

    result = {
        "args": vars(args),
        "baseline": {
            "ms": baseline_ms,
            "subranges_ms": subranges,
        },
        "candidate": candidate_row,
        "device": torch.cuda.get_device_name(),
        "problem": problem_metrics(fixture, reference, baseline_ms, subranges),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
