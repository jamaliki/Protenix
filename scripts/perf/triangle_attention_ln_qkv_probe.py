#!/usr/bin/env python3
"""Screen a fused LayerNorm + q/k/v producer for CUEQ triangle attention.

The current triangle-attention path is already fairly good:

    x -> LayerNorm -> one cuBLAS qkv GEMM -> Triton qkv layout -> CUEQ attention

Nsight Compute still shows the producer boundary as material on the wider
Protenix-v2 pairformer shape.  The important question is whether deleting the
materialized normalized activation and qkv-layout pass is worth replacing a
vendor GEMM with a custom kernel.

This script is deliberately benchmark-only.  It compares three producer
boundaries in the same process:

* ``current``: production ``LayerNorm`` + ``Attention._prep_qkv``.
* ``cached``: same math, but with the packed qkv weight cached so a win cannot
  be misattributed to avoiding ``torch.cat``.
* ``fused``: one Triton kernel that normalizes each row, multiplies by the q/k/v
  weights, and writes CUEQ's physical ``[B, I, H, J, D]`` (or swapped ending
  ``[B, J, H, I, D]``) layout directly.

The fused kernel intentionally uses BF16 tensor-core dot products after FP32
row statistics, matching the throughput-oriented inference mode we care about.
If it cannot beat the cached producer here, a production integration would only
add complexity around a worse mainloop.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None

from protenix.model.triangular.qkv_layout_triton import triton_qkv_to_cueq_heads_first
from protenix.model.triangular.triangular import TriangleAttention


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_warps: int


@dataclass(frozen=True)
class ShapeCase:
    name: str
    batch: int
    tokens: int
    c_z: int
    no_heads: int
    c_hidden: int = 32


DEFAULT_CASES = (
    ShapeCase("base_b32_n251", batch=32, tokens=251, c_z=128, no_heads=4),
    ShapeCase("v2_b32_n251", batch=32, tokens=251, c_z=256, no_heads=8),
)


if triton is not None and tl is not None:

    @triton.jit
    def _ln_qkv_to_cueq_kernel(
        x_ptr,
        gamma_ptr,
        beta_ptr,
        weight_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        total_rows: tl.constexpr,
        i_count: tl.constexpr,
        j_count: tl.constexpr,
        c_z: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        swapped: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        row_offsets = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(
            tl.int64
        )
        hidden_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
        component = tl.program_id(2)

        k_offsets = tl.arange(0, block_k)
        row_mask = row_offsets < total_rows
        k_mask = k_offsets < c_z

        # LayerNorm is row-local over the channel axis.  We compute statistics
        # in FP32, then cast the normalized row to BF16 for the tensor-core dot.
        x = tl.load(
            x_ptr + row_offsets[:, None] * c_z + k_offsets[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(x, axis=1) / c_z
        centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) / c_z
        inv_std = tl.rsqrt(variance + eps)

        gamma = tl.load(gamma_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        norm = centered * inv_std[:, None]
        norm = tl.where(k_mask[None, :], norm * gamma[None, :] + beta[None, :], 0.0)

        hidden = head_count * head_dim
        weight_rows = component * hidden + hidden_offsets
        n_mask = hidden_offsets < hidden
        weight = tl.load(
            weight_ptr + weight_rows[:, None] * c_z + k_offsets[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(norm.to(tl.bfloat16), tl.trans(weight))

        m = row_offsets[:, None]
        j = m % j_count
        tmp = m // j_count
        i = tmp % i_count
        batch = tmp // i_count

        n = hidden_offsets[None, :]
        head = n // head_dim
        dim = n - head * head_dim

        if swapped:
            # Ending-node attention is the starting-node problem on
            # x.transpose(I, J).  This store writes the transposed CUEQ layout
            # directly while the producer still reads the original contiguous x.
            out_offsets = (
                (((batch * j_count + j) * head_count + head) * i_count + i)
                * head_dim
                + dim
            )
        else:
            out_offsets = (
                (((batch * i_count + i) * head_count + head) * j_count + j)
                * head_dim
                + dim
            )

        store_mask = row_mask[:, None] & n_mask[None, :]
        tl.store(q_ptr + out_offsets, acc, mask=store_mask & (component == 0))
        tl.store(k_ptr + out_offsets, acc, mask=store_mask & (component == 1))
        tl.store(v_ptr + out_offsets, acc, mask=store_mask & (component == 2))


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
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def parse_kernel_configs(value: str) -> list[KernelConfig]:
    configs: list[KernelConfig] = []
    for item in value.split(","):
        parts = item.lower().split("x")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "kernel configs must look like '16x32x4,16x64x4'"
            )
        configs.append(KernelConfig(*(int(part) for part in parts)))
    return configs


def select_cases(value: str) -> list[ShapeCase]:
    by_name = {case.name: case for case in DEFAULT_CASES}
    if value == "none":
        return []
    if value == "all":
        return list(DEFAULT_CASES)
    cases = []
    for name in value.split(","):
        try:
            cases.append(by_name[name])
        except KeyError as exc:
            raise argparse.ArgumentTypeError(
                f"unknown case {name!r}; choose from {', '.join(by_name)}, all, or none"
            ) from exc
    return cases


def parse_custom_case(value: str) -> ShapeCase:
    """Parse ``name:batch:tokens:c_z:no_heads[:c_hidden]`` for ad hoc screens."""

    parts = value.split(":")
    if len(parts) not in {5, 6}:
        raise argparse.ArgumentTypeError(
            "custom cases must be name:batch:tokens:c_z:no_heads[:c_hidden]"
        )
    name = parts[0]
    if not name:
        raise argparse.ArgumentTypeError("custom case name cannot be empty")
    try:
        batch, tokens, c_z, no_heads = (int(part) for part in parts[1:5])
        c_hidden = int(parts[5]) if len(parts) == 6 else 32
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "custom case dimensions must be integers"
        ) from exc
    if min(batch, tokens, c_z, no_heads, c_hidden) <= 0:
        raise argparse.ArgumentTypeError("custom case dimensions must be positive")
    return ShapeCase(
        name=name,
        batch=batch,
        tokens=tokens,
        c_z=c_z,
        no_heads=no_heads,
        c_hidden=c_hidden,
    )


def make_module(case: ShapeCase, *, ending: bool, seed: int) -> TriangleAttention:
    module = TriangleAttention(
        c_in=case.c_z,
        c_hidden=case.c_hidden,
        no_heads=case.no_heads,
        starting=not ending,
    ).cuda()
    module.eval()

    # The real model has trained weights.  Randomizing only zero-initialized
    # matrices avoids benchmarking degenerate all-zero GEMMs in the synthetic
    # module while preserving the production module topology.
    generator = torch.Generator(device="cuda").manual_seed(seed + 8191)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2 and not torch.any(parameter):
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(case: ShapeCase, dtype_name: str) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    return torch.randn(
        case.batch,
        case.tokens,
        case.tokens,
        case.c_z,
        device="cuda",
        dtype=dtype,
    )


def qkv_weight(module: TriangleAttention, dtype: torch.dtype) -> torch.Tensor:
    weights = [
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
    ]
    return torch.cat([weight.to(dtype=dtype) for weight in weights], dim=0).contiguous()


def current_producer(
    module: TriangleAttention,
    x: torch.Tensor,
    *,
    swapped: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm = module.layer_norm(x)
    return module.mha._prep_qkv(
        x_norm,
        x_norm,
        apply_scale=False,
        cueq_heads_first=True,
        cueq_swap_pair_axes=swapped,
    )


def cached_producer(
    module: TriangleAttention,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    swapped: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm = module.layer_norm(x)
    with torch.amp.autocast("cuda", enabled=False):
        qkv = F.linear(x_norm, weight)
    out = triton_qkv_to_cueq_heads_first(
        qkv,
        module.no_heads,
        swap_pair_axes=swapped,
    )
    if out is None:
        raise RuntimeError("Triton qkv layout helper rejected the benchmark shape")
    return out


def fused_producer(
    module: TriangleAttention,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    swapped: bool,
    config: KernelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for fused producer benchmark")
    if x.dtype != torch.bfloat16:
        raise RuntimeError("fused producer currently targets BF16 inference only")
    if not x.is_contiguous():
        raise RuntimeError("fused producer expects contiguous [B, I, J, C] input")

    batch, i_count, j_count, c_z = x.shape
    head_count = module.no_heads
    head_dim = module.mha.c_hidden
    if swapped:
        out_shape = (batch, j_count, head_count, i_count, head_dim)
    else:
        out_shape = (batch, i_count, head_count, j_count, head_dim)
    q = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    total_rows = batch * i_count * j_count
    hidden = head_count * head_dim
    block_k = 1 << (c_z - 1).bit_length()
    grid = (
        triton.cdiv(total_rows, config.block_m),
        triton.cdiv(hidden, config.block_n),
        3,
    )
    _ln_qkv_to_cueq_kernel[grid](
        x,
        module.layer_norm.weight,
        module.layer_norm.bias,
        weight,
        q,
        k,
        v,
        total_rows,
        i_count,
        j_count,
        c_z,
        head_count,
        head_dim,
        module.layer_norm.eps,
        swapped,
        config.block_m,
        config.block_n,
        block_k,
        num_warps=config.num_warps,
        num_stages=3,
    )
    return q, k, v


def time_cuda(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iters: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
    out = fn()
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


def compare_outputs(
    candidate: Iterable[torch.Tensor],
    reference: Iterable[torch.Tensor],
) -> dict[str, float]:
    max_abs = 0.0
    mean_abs_total = 0.0
    count = 0
    for cand, ref in zip(candidate, reference, strict=True):
        diff = (cand.float() - ref.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs_total += float(diff.sum().item())
        count += diff.numel()
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs_total / count,
    }


def run_case(
    case: ShapeCase,
    *,
    ending: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    module = make_module(case, ending=ending, seed=args.seed)
    x = make_inputs(case, args.input_dtype)
    weight = qkv_weight(module, x.dtype)
    swapped = ending

    with torch.inference_mode(), cuda_autocast(args.compute_dtype):
        current_out, current_ms = time_cuda(
            lambda: current_producer(module, x, swapped=swapped),
            warmup=args.warmup,
            iters=args.iters,
        )
        cached_out, cached_ms = time_cuda(
            lambda: cached_producer(module, x, weight, swapped=swapped),
            warmup=args.warmup,
            iters=args.iters,
        )

        fused_results = []
        for config in args.kernel_configs:
            fused_out, fused_ms = time_cuda(
                lambda config=config: fused_producer(
                    module,
                    x,
                    weight,
                    swapped=swapped,
                    config=config,
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
            fused_results.append(
                {
                    "config": {
                        "block_m": config.block_m,
                        "block_n": config.block_n,
                        "num_warps": config.num_warps,
                    },
                    "fused_ms": fused_ms,
                    "speedup_vs_current": current_ms / fused_ms,
                    "speedup_vs_cached": cached_ms / fused_ms,
                    "parity_vs_cached": compare_outputs(fused_out, cached_out),
                    "parity_vs_current": compare_outputs(fused_out, current_out),
                }
            )

    best = max(fused_results, key=lambda row: row["speedup_vs_cached"])
    return {
        "case": case.__dict__,
        "ending": ending,
        "current_ms": current_ms,
        "cached_ms": cached_ms,
        "cached_speedup_vs_current": current_ms / cached_ms,
        "cached_parity_vs_current": compare_outputs(cached_out, current_out),
        "fused": fused_results,
        "best_fused": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=select_cases, default=list(DEFAULT_CASES))
    parser.add_argument(
        "--custom-case",
        action="append",
        type=parse_custom_case,
        default=[],
        help=(
            "Add an ad hoc shape as name:batch:tokens:c_z:no_heads[:c_hidden]. "
            "Useful for screening mixed-token endpoints without editing the script."
        ),
    )
    parser.add_argument("--include-start", type=str_bool, default=True)
    parser.add_argument("--include-end", type=str_bool, default=True)
    parser.add_argument(
        "--input-dtype",
        choices=["bfloat16"],
        default="bfloat16",
        help="The fused screen is intentionally BF16-only.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16"],
        default="bfloat16",
        help="The screen targets the BF16 inference path that uses tensor cores.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--kernel-configs",
        type=parse_kernel_configs,
        default=parse_kernel_configs("16x32x4,16x64x4,32x32x4"),
        help="Comma-separated block_m x block_n x num_warps Triton configs.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    if args.custom_case:
        args.cases = [*args.cases, *args.custom_case]
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_ln_qkv_probe requires CUDA")
    if triton is None or tl is None:
        raise RuntimeError("triangle_attention_ln_qkv_probe requires Triton")
    if not args.include_start and not args.include_end:
        raise ValueError("at least one of --include-start/--include-end must be true")
    if not args.cases:
        raise ValueError("no shape cases selected")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    results = []
    for case in args.cases:
        if args.include_start:
            results.append(run_case(case, ending=False, args=args))
        if args.include_end:
            results.append(run_case(case, ending=True, args=args))

    payload = {
        "args": {
            **vars(args),
            "cases": [case.__dict__ for case in args.cases],
            "custom_case": [case.__dict__ for case in args.custom_case],
            "kernel_configs": [config.__dict__ for config in args.kernel_configs],
        },
        "results": results,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    print(text)


if __name__ == "__main__":
    main()
