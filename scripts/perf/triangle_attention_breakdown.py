#!/usr/bin/env python3
"""CUDA-event breakdown for one triangle attention module.

Pairformer profiling shows that high-throughput, few-sample inference is now
dominated by triangle attention.  The block-level timer is still too coarse:
it lumps together PyTorch projections, small dtype/mask kernels, and the
cuequivariance attention kernel.  This script separates those pieces so a
custom kernel target is chosen by measured time, not by intuition.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator, TypeVar

import torch

from protenix.model.modules.fused_elementwise_triton import (
    triton_fused_elementwise_available,
    triton_fused_elementwise_enabled,
    triton_triangle_attention_epilogue_enabled,
    triton_sigmoid_mul_heads_first,
    triton_sigmoid_mul_heads_first_swapped_gate,
)
from protenix.model.triangular.layers import (
    cuequivariance_triangular_attn,
    flatten_final_dims,
)
from protenix.model.triangular.qkv_layout_triton import (
    triton_cueq_qkv_layout_available,
    triton_cueq_qkv_layout_enabled,
)
from protenix.model.triangular.triangular import (
    TriangleAttention,
    cueq_bool_mask_enabled,
)
from protenix.model.utils import permute_final_dims

T = TypeVar("T")


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


def timed(
    name: str,
    fn: Callable[[], T],
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
) -> T:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events.append((name, start, end))
    return out


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
        starting=not args.ending,
    ).cuda()
    module.eval()

    # Some OpenFold-style projections are zero-initialized.  Randomizing those
    # matrices makes the screen exercise the same GEMM shapes with nontrivial
    # data while preserving the module topology used by the real model.
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 65537)
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(
        args.batch, args.tokens, args.tokens, args.c_z, device=device, dtype=dtype
    )
    mask = torch.ones(args.batch, args.tokens, args.tokens, device=device, dtype=dtype)
    return x, mask


def manual_cueq_forward(
    module: TriangleAttention,
    x_in: torch.Tensor,
    mask_in: torch.Tensor,
    *,
    ending_norm_before_transpose: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    x = x_in
    mask = mask_in

    if not module.starting and ending_norm_before_transpose:
        # LayerNorm is applied independently over the channel dimension, so it
        # commutes with swapping the two residue axes:
        #     LN(x.transpose(I, J)) == LN(x).transpose(I, J)
        # The default ending-node path normalizes the transposed view.  This
        # probe measures whether normalizing the original contiguous layout
        # first is a good boundary for a production rewrite.
        x = timed("layer_norm_pretranspose", lambda: module.layer_norm(x), events)
        x = timed("input_transpose_after_norm", lambda: x.transpose(-2, -3), events)
        mask = timed("mask_transpose", lambda: mask.transpose(-1, -2), events)
    elif not module.starting:
        x = timed("input_transpose", lambda: x.transpose(-2, -3), events)
        mask = timed("mask_transpose", lambda: mask.transpose(-1, -2), events)
        x = timed("layer_norm", lambda: module.layer_norm(x), events)
    else:
        x = timed("layer_norm", lambda: module.layer_norm(x), events)

    # The cuequivariance kernel consumes a bool mask, but the generic module
    # first builds the additive -inf mask used by the torch/deepspeed paths and
    # then compares it back to zero.  If this is material, it is a good fusion
    # or specialization boundary.
    if cueq_bool_mask_enabled():
        mask_for_attention = timed(
            "mask_bool_direct",
            lambda: (mask == 1)[..., :, None, None, :],
            events,
        )
    else:
        mask_bias = timed(
            "mask_bias_materialize",
            lambda: (module.inf * (mask - 1))[..., :, None, None, :],
            events,
        )

    triangle_bias_linear = timed(
        "triangle_bias_linear",
        lambda: module.linear(x),
        events,
    )
    triangle_bias = timed(
        "triangle_bias_layout",
        lambda: permute_final_dims(triangle_bias_linear, (2, 0, 1)).unsqueeze(-4),
        events,
    )

    # Production CUEQ attention needs physically contiguous [..., I, H, J, D]
    # q/k/v tensors.  Time the same heads-first producer here so the split
    # breakdown accounts for layout traffic at its actual boundary.
    q, k, v = timed(
        "qkv_projection",
        lambda: module.mha._prep_qkv(
            x,
            x,
            apply_scale=False,
            cueq_heads_first=True,
        ),
        events,
    )

    triangle_bias_fp32 = timed(
        "triangle_bias_fp32",
        lambda: triangle_bias.float(),
        events,
    )
    if not cueq_bool_mask_enabled():
        mask_for_attention = timed(
            "mask_bool_from_bias",
            lambda: (mask_bias == 0).bool(),
            events,
        )

    scale = 1.0 / math.sqrt(module.c_hidden)
    heads_first = timed(
        "cueq_attention",
        lambda: cuequivariance_triangular_attn(
            q, k, v, triangle_bias_fp32, mask_for_attention, scale
        ),
        events,
    )
    if isinstance(heads_first, tuple):
        heads_first = heads_first[0]

    if module.mha.linear_g is not None:
        gate = timed("gate_projection", lambda: module.mha.linear_g(x), events)
        # This is the current production fast path.  CUEQ returns heads-first
        # [..., H, Q, D] data; the Triton helper streams sigmoid(gate) * output
        # directly into the flattened [..., Q, H*D] layout consumed by the
        # output projection.  If the helper rejects the shape, the fallback
        # below exposes the separate transpose/sigmoid/multiply/copy costs.
        o = timed(
            "gate_sigmoid_mul_flatten_triton",
            lambda: triton_sigmoid_mul_heads_first(gate, heads_first),
            events,
        )
        if o is None:
            o = timed(
                "attention_transpose",
                lambda: heads_first.transpose(-2, -3),
                events,
            )
            gate = timed(
                "gate_sigmoid_layout",
                lambda: module.mha.sigmoid(gate).view(
                    gate.shape[:-1] + (module.no_heads, -1)
                ),
                events,
            )
            o = timed("gate_multiply", lambda: o * gate, events)
            o = timed("flatten_heads", lambda: flatten_final_dims(o, 2), events)
    else:
        o = timed("attention_transpose", lambda: heads_first.transpose(-2, -3), events)
        o = timed("flatten_heads", lambda: flatten_final_dims(o, 2), events)

    o = timed("output_projection", lambda: module.mha.linear_o(o), events)

    if not module.starting:
        o = timed("output_transpose", lambda: o.transpose(-2, -3), events)

    torch.cuda.synchronize()
    timings = {name: start.elapsed_time(end) for name, start, end in events}
    return o, timings


def manual_cueq_ending_contiguous_forward(
    module: TriangleAttention,
    x_in: torch.Tensor,
    mask_in: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ending-node attention prototype that keeps projections contiguous.

    The production ending path evaluates the starting-node attention on
    ``x.transpose(I, J)``.  That is mathematically clean but makes LayerNorm and
    all projection GEMMs consume a transposed pair layout.  This prototype uses
    the identity ``LN(x.transpose(I, J)) == LN(x).transpose(I, J)`` and computes
    the expensive channel-wise projections on the original contiguous ``x``.
    Small Triton layout helpers then write the swapped CUEQ layouts directly.

    It is deliberately a benchmark path: if this does not move the H100 module
    subtotal, a production rewrite would only add complexity.
    """
    if module.starting:
        raise ValueError("contiguous ending prototype requires ending attention")

    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    x = timed("layer_norm_contiguous", lambda: module.layer_norm(x_in), events)

    if cueq_bool_mask_enabled():
        mask_for_attention = timed(
            "mask_bool_swapped",
            lambda: (mask_in.transpose(-1, -2) == 1)[..., :, None, None, :],
            events,
        )
    else:
        mask_bias = timed(
            "mask_bias_swapped",
            lambda: (module.inf * (mask_in.transpose(-1, -2) - 1))[
                ..., :, None, None, :
            ],
            events,
        )

    triangle_bias_linear = timed(
        "triangle_bias_linear_contiguous",
        lambda: module.linear(x),
        events,
    )
    triangle_bias = timed(
        "triangle_bias_swapped_layout",
        lambda: permute_final_dims(triangle_bias_linear, (2, 1, 0)).unsqueeze(-4),
        events,
    )

    q, k, v = timed(
        "qkv_projection_swapped_layout",
        lambda: module.mha._prep_qkv(
            x,
            x,
            apply_scale=False,
            cueq_heads_first=True,
            cueq_swap_pair_axes=True,
        ),
        events,
    )

    triangle_bias_fp32 = timed(
        "triangle_bias_fp32",
        lambda: triangle_bias.float(),
        events,
    )
    if not cueq_bool_mask_enabled():
        mask_for_attention = timed(
            "mask_bool_from_swapped_bias",
            lambda: (mask_bias == 0).bool(),
            events,
        )

    scale = 1.0 / math.sqrt(module.c_hidden)
    heads_first = timed(
        "cueq_attention",
        lambda: cuequivariance_triangular_attn(
            q, k, v, triangle_bias_fp32, mask_for_attention, scale
        ),
        events,
    )
    if isinstance(heads_first, tuple):
        heads_first = heads_first[0]

    gate = timed("gate_projection_contiguous", lambda: module.mha.linear_g(x), events)
    o = timed(
        "gate_sigmoid_mul_swapped_triton",
        lambda: triton_sigmoid_mul_heads_first_swapped_gate(gate, heads_first),
        events,
    )
    if o is None:
        gate_t = timed(
            "gate_transpose_contiguous",
            lambda: gate.transpose(-2, -3).contiguous(),
            events,
        )
        o = timed(
            "gate_sigmoid_mul_flatten_triton",
            lambda: triton_sigmoid_mul_heads_first(gate_t, heads_first),
            events,
        )
        if o is None:
            o = timed(
                "attention_transpose",
                lambda: heads_first.transpose(-2, -3),
                events,
            )
            gate_view = timed(
                "gate_sigmoid_layout",
                lambda: module.mha.sigmoid(gate_t).view(
                    gate_t.shape[:-1] + (module.no_heads, -1)
                ),
                events,
            )
            o = timed("gate_multiply", lambda: o * gate_view, events)
            o = timed("flatten_heads", lambda: flatten_final_dims(o, 2), events)

    o = timed("output_projection", lambda: module.mha.linear_o(o), events)
    o = timed("output_transpose", lambda: o.transpose(-2, -3), events)

    torch.cuda.synchronize()
    timings = {name: start.elapsed_time(end) for name, start, end in events}
    return o, timings


def module_forward(
    module: TriangleAttention,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return module(x, mask=mask, triangle_attention="cuequivariance")


def layout_probe(
    module: TriangleAttention,
    x_in: torch.Tensor,
    mask_in: torch.Tensor,
) -> dict[str, object]:
    x = x_in
    mask = mask_in
    if not module.starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

    x = module.layer_norm(x)
    if cueq_bool_mask_enabled():
        mask_for_attention = (mask == 1)[..., :, None, None, :]
    else:
        mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
        mask_for_attention = (mask_bias == 0).bool()
    triangle_bias = permute_final_dims(module.linear(x), (2, 0, 1)).unsqueeze(-4)
    # Match TriangleAttention.forward(..., triangle_attention="cuequivariance"):
    # the production path asks _prep_qkv to materialize CUEQ's physical
    # heads-first layout, using the Triton producer when its guards pass.  This
    # keeps the split benchmark honest: qkv_projection then includes the same
    # layout kernel that Nsight Compute sees in the real module, rather than a
    # cheaper metadata transpose that the CUEQ wrapper would later make
    # contiguous out of sight.
    q, k, v = module.mha._prep_qkv(
        x,
        x,
        apply_scale=False,
        cueq_heads_first=True,
    )
    scale = 1.0 / math.sqrt(module.c_hidden)
    heads_first = cuequivariance_triangular_attn(
        q, k, v, triangle_bias.float(), mask_for_attention, scale
    )
    if isinstance(heads_first, tuple):
        heads_first = heads_first[0]
    gate = module.mha.linear_g(x)
    fused = triton_sigmoid_mul_heads_first(gate, heads_first)

    return {
        "triton_enabled": triton_fused_elementwise_enabled(),
        "triangle_attention_epilogue_enabled": (
            triton_triangle_attention_epilogue_enabled()
        ),
        "cueq_qkv_layout_enabled": triton_cueq_qkv_layout_enabled(),
        "cueq_bool_mask_enabled": cueq_bool_mask_enabled(),
        "cueq_qkv_layout_available": triton_cueq_qkv_layout_available(),
        "triton_available": triton_fused_elementwise_available(),
        "helper_returned": fused is not None,
        "gate_shape": list(gate.shape),
        "gate_stride": list(gate.stride()),
        "gate_contiguous": gate.is_contiguous(),
        "gate_dtype": str(gate.dtype),
        "q_shape": list(q.shape),
        "q_stride": list(q.stride()),
        "q_contiguous": q.is_contiguous(),
        "q_dtype": str(q.dtype),
        "heads_first_shape": list(heads_first.shape),
        "heads_first_stride": list(heads_first.stride()),
        "heads_first_contiguous": heads_first.is_contiguous(),
        "heads_first_dtype": str(heads_first.dtype),
        "fused_shape": None if fused is None else list(fused.shape),
        "fused_stride": None if fused is None else list(fused.stride()),
    }


def cuda_time(fn: Callable[[], torch.Tensor], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return float(finite.max().item()) if finite.numel() else float("nan")


def cuda_profiler_capture(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> torch.Tensor:
    """Run one narrow CUDA-profiler capture for Nsight Compute.

    Nsight Compute can be told to ignore kernels until CUDA profiling starts
    (``ncu --profile-from-start off``).  Wrapping only the target module keeps
    reports small and prevents first-call Triton/CUEQ compilation or unrelated
    setup kernels from dominating the launch list.
    """
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(name)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--ending", type=str_bool, default=False)
    dtype_choices = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtype_choices, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    parser.add_argument(
        "--ending-norm-before-transpose",
        type=str_bool,
        default=False,
        help=(
            "Diagnostic only: for ending-node attention, time the mathematically "
            "equivalent path LN(x).transpose(...) instead of LN(x.transpose(...))."
        ),
    )
    parser.add_argument(
        "--ending-contiguous-producer",
        type=str_bool,
        default=False,
        help=(
            "Diagnostic only: for ending-node attention, keep LayerNorm and the "
            "qkv/gate/bias projections on the original contiguous pair tensor, "
            "then write swapped CUEQ layouts with prototype Triton helpers."
        ),
    )
    parser.add_argument(
        "--ncu-capture",
        choices=["none", "module", "manual"],
        default="none",
        help=(
            "Run only a warmed CUDA-profiler capture for Nsight Compute. "
            "'module' captures TriangleAttention.forward; 'manual' captures "
            "the split implementation used by this breakdown script."
        ),
    )
    parser.add_argument(
        "--ncu-iters",
        type=int,
        default=1,
        help="Target calls inside the CUDA-profiler capture range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_breakdown requires CUDA")
    if args.ending_contiguous_producer and not args.ending:
        raise ValueError("--ending-contiguous-producer requires --ending true")

    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)
    module = make_module(args)
    x, mask = make_inputs(args)

    totals: dict[str, float] = defaultdict(float)
    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        if args.ncu_capture != "none":
            capture_fn: Callable[[], torch.Tensor]
            if args.ncu_capture == "module":
                capture_fn = lambda: module_forward(module, x, mask)
            else:
                if args.ending_contiguous_producer:
                    capture_fn = lambda: manual_cueq_ending_contiguous_forward(
                        module,
                        x,
                        mask,
                    )[0]
                else:
                    capture_fn = lambda: manual_cueq_forward(
                        module,
                        x,
                        mask,
                        ending_norm_before_transpose=args.ending_norm_before_transpose,
                    )[0]
            out = cuda_profiler_capture(
                (
                    f"triangle_attention_{args.ncu_capture}_"
                    f"B{args.batch}_N{args.tokens}_"
                    f"{'end' if args.ending else 'start'}"
                ),
                capture_fn,
                warmup=args.warmup,
                iters=args.ncu_iters,
            )
            layout = layout_probe(module, x, mask)
            print(
                json.dumps(
                    {
                        "args": vars(args),
                        "capture_shape": list(out.shape),
                        "layout_probe": layout,
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
            if args.ending_contiguous_producer:
                manual_cueq_ending_contiguous_forward(module, x, mask)
            else:
                manual_cueq_forward(
                    module,
                    x,
                    mask,
                    ending_norm_before_transpose=args.ending_norm_before_transpose,
                )

        reference = module_forward(module, x, mask)
        if args.ending_contiguous_producer:
            split, _ = manual_cueq_ending_contiguous_forward(module, x, mask)
        else:
            split, _ = manual_cueq_forward(
                module,
                x,
                mask,
                ending_norm_before_transpose=args.ending_norm_before_transpose,
            )
        parity = max_abs_error(reference, split)
        layout = layout_probe(module, x, mask)

        for _ in range(args.warmup):
            module_forward(module, x, mask)
        module_forward_ms = cuda_time(lambda: module_forward(module, x, mask), args.iters)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iters):
            if args.ending_contiguous_producer:
                _, timings = manual_cueq_ending_contiguous_forward(module, x, mask)
            else:
                _, timings = manual_cueq_forward(
                    module,
                    x,
                    mask,
                    ending_norm_before_transpose=args.ending_norm_before_transpose,
                )
            for name, elapsed_ms in timings.items():
                totals[name] += elapsed_ms

    mean_ms = {name: total / args.iters for name, total in totals.items()}
    total_ms = sum(mean_ms.values())
    row = {
        "args": vars(args),
        "manual_total_ms": total_ms,
        "module_forward_ms": module_forward_ms,
        "layout_probe": layout,
        "mean_ms": mean_ms,
        "percent": {name: 100.0 * value / total_ms for name, value in mean_ms.items()},
        "parity_max_abs_error": parity,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
