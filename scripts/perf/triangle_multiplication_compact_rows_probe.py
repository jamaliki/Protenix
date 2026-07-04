#!/usr/bin/env python3
"""Screen compact valid-row execution inside v2 triangle multiplication.

The rejected contraction probes showed that a standalone custom contraction is
not enough.  Most triangle-multiplication time is the surrounding row-wise work:
LayerNorm, gated input projection, output LayerNorm, and output projection/gate.

This diagnostic keeps the dense PyTorch/CUEQ-style contraction but compacts the
row-wise producers/consumers to valid `(i, j)` pair rows.  It answers a narrow
question before a native rewrite: is there enough row-wise padding headroom to
justify a full segmented triangle-multiplication kernel, or do gather/scatter
and layout conversions erase the win even in this favorable decomposition?

It is not a production path.  A real implementation would keep the compact
schedule inside native kernels instead of materializing dense `a/b` tensors just
to feed the existing `einsum` contraction.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from collections.abc import Callable

import torch

from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

try:
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from cuequivariance_ops_torch.gated_gemm_torch import (
        fused_sigmoid_gated_dual_gemm,
        fused_sigmoid_gated_dual_gemm_dual_x,
    )
except ImportError:  # pragma: no cover - optional GPU dependency.
    layer_norm_transpose = None
    fused_sigmoid_gated_dual_gemm = None
    fused_sigmoid_gated_dual_gemm_dual_x = None


TensorFn = Callable[[], torch.Tensor]
DTYPE = torch.bfloat16
C_Z = 256
SHORT_LENGTHS = [40, 40, 52, 52, 64, 64, 76, 76, 88, 88, 100, 100, 112, 112, 124, 124]
LONG_LENGTHS = [
    136, 136, 160, 160, 172, 172, 184, 184,
    196, 196, 208, 208, 216, 216, 220, 220,
]


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def lengths_from_args(args: argparse.Namespace) -> list[int]:
    if args.lengths is not None:
        return args.lengths
    return SHORT_LENGTHS if args.preset == "short" else LONG_LENGTHS


def make_modules(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    modules = {
        "outgoing": TriangleMultiplicationOutgoing(C_Z, C_Z).cuda().eval(),
        "incoming": TriangleMultiplicationIncoming(C_Z, C_Z).cuda().eval(),
    }
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1729)
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return modules


def make_padded_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(lengths)
    z = torch.zeros(len(lengths), n_max, n_max, C_Z, device="cuda", dtype=DTYPE)
    mask = torch.zeros(len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length, length, C_Z, device="cuda", dtype=DTYPE, generator=generator
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def make_valid_index(lengths: list[int]) -> torch.Tensor:
    n_max = max(lengths)
    chunks = []
    for batch_index, length in enumerate(lengths):
        rows = torch.arange(length, device="cuda")
        i, j = torch.meshgrid(rows, rows, indexing="ij")
        chunks.append((batch_index * n_max * n_max + i * n_max + j).reshape(-1))
    return torch.cat(chunks)


def valid_from_dense(z: torch.Tensor, valid_index: torch.Tensor) -> torch.Tensor:
    return z.reshape(-1, z.shape[-1]).index_select(0, valid_index)


def scatter_compact(
    compact: torch.Tensor,
    valid_index: torch.Tensor,
    *,
    batch: int,
    tokens: int,
) -> torch.Tensor:
    flat = compact.new_zeros(batch * tokens * tokens, compact.shape[-1])
    flat.index_copy_(0, valid_index, compact)
    return flat.reshape(batch, tokens, tokens, compact.shape[-1])


def run_padded_valid(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    valid_index: torch.Tensor,
) -> torch.Tensor:
    out = module(
        z.clone(),
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )
    return valid_from_dense(out, valid_index)


def _compact_update(
    module: torch.nn.Module,
    z: torch.Tensor,
    valid_index: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    if (
        layer_norm_transpose is None
        or fused_sigmoid_gated_dual_gemm is None
        or fused_sigmoid_gated_dual_gemm_dual_x is None
    ):
        raise RuntimeError("CUEQ torch ops are required")

    batch, tokens, _, c_z = z.shape
    flat_z = z.reshape(-1, c_z)
    z_valid = flat_z.index_select(0, valid_index)

    x_norm = layer_norm_transpose(
        z_valid,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        eps=1e-5,
        layout="nd->nd",
    )
    p_in_weight = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0)
    g_in_weight = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0)
    ab = fused_sigmoid_gated_dual_gemm(
        x_norm,
        g_in_weight,
        p_in_weight,
        mask=None,
        transpose_out=False,
    )
    a_valid, b_valid = torch.chunk(ab, 2, dim=-1)

    a_flat = z.new_zeros(batch * tokens * tokens, c_z)
    b_flat = z.new_zeros(batch * tokens * tokens, c_z)
    a_flat.index_copy_(0, valid_index, a_valid)
    b_flat.index_copy_(0, valid_index, b_valid)
    a = a_flat.reshape(batch, tokens, tokens, c_z).permute(3, 0, 1, 2).contiguous()
    b = b_flat.reshape(batch, tokens, tokens, c_z).permute(3, 0, 1, 2).contiguous()

    if direction == "outgoing":
        update = torch.einsum("dbik,dbjk->dbij", a, b)
    elif direction == "incoming":
        update = torch.einsum("dbki,dbkj->dbij", a, b)
    else:
        raise ValueError("direction must be outgoing or incoming")

    update_valid = (
        update.permute(1, 2, 3, 0)
        .contiguous()
        .reshape(-1, c_z)
        .index_select(0, valid_index)
    )
    update_norm = layer_norm_transpose(
        update_valid,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        eps=1e-5,
        layout="nd->nd",
    )
    out = fused_sigmoid_gated_dual_gemm_dual_x(
        x_norm,
        update_norm,
        module.linear_g.weight,
        module.linear_z.weight,
    )
    return out + z_valid


def run_compact_valid_only(
    module: torch.nn.Module,
    z: torch.Tensor,
    valid_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    return _compact_update(module, z, valid_index, direction=direction)


def run_compact_round_trip(
    module: torch.nn.Module,
    z: torch.Tensor,
    valid_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    compact = _compact_update(module, z, valid_index, direction=direction)
    dense = scatter_compact(compact, valid_index, batch=z.shape[0], tokens=z.shape[1])
    return valid_from_dense(dense, valid_index)


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
    parser.add_argument("--preset", choices=["short", "long"], default="long")
    parser.add_argument("--lengths", type=parse_ints)
    parser.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_multiplication_compact_rows_probe requires CUDA")
    if (
        layer_norm_transpose is None
        or fused_sigmoid_gated_dual_gemm is None
        or fused_sigmoid_gated_dual_gemm_dual_x is None
    ):
        raise RuntimeError("CUEQ torch ops are required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    lengths = lengths_from_args(args)
    module = make_modules(args)[args.direction]
    z, mask = make_padded_inputs(lengths, args)
    valid_index = make_valid_index(lengths)
    valid_pairs = sum(length * length for length in lengths)
    padded_pairs = len(lengths) * max(lengths) ** 2

    variants: dict[str, TensorFn] = {
        "padded_valid": lambda: run_padded_valid(module, z, mask, valid_index),
        "compact_valid_only": lambda: run_compact_valid_only(
            module, z, valid_index, args.direction
        ),
        "compact_round_trip": lambda: run_compact_round_trip(
            module, z, valid_index, args.direction
        ),
    }

    results: dict[str, object] = {}
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        reference, padded_ms = cuda_time(variants["padded_valid"], args.warmup, args.iters)
        results["padded_valid"] = {"ms": padded_ms, "speedup_vs_padded": 1.0}
        for name in ("compact_valid_only", "compact_round_trip"):
            candidate, ms = cuda_time(variants[name], args.warmup, args.iters)
            results[name] = {
                "ms": ms,
                "speedup_vs_padded": padded_ms / ms,
                "valid_vs_padded": compare(reference, candidate),
            }

    row = {
        "args": {**vars(args), "lengths": lengths},
        "shape": {
            "batch": len(lengths),
            "tokens": max(lengths),
            "c_z": C_Z,
            "valid_pairs": valid_pairs,
            "padded_pairs": padded_pairs,
            "valid_pair_efficiency": valid_pairs / padded_pairs,
            "ideal_pair_row_speedup": padded_pairs / valid_pairs,
        },
        "results": results,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device": torch.cuda.get_device_name(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    text = json.dumps(row, indent=2, sort_keys=True)
    if args.output_json:
        from pathlib import Path

        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
