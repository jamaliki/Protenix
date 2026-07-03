#!/usr/bin/env python3
"""Screen a wider CUEQ triangle-attention producer boundary.

The current fast path already keeps CUEQ's attention kernel and fuses the
q/k/v projection into one GEMM.  Around that good kernel we still launch two
more projections from the same normalized pair tensor:

  * a gate projection consumed by the output epilogue
  * a small triangle-bias projection consumed by CUEQ

This probe asks whether those projections should be fused with q/k/v into one
wide cuBLAS GEMM, then unpacked once into CUEQ's physical layouts.  It is
benchmark-only: if the packed producer does not move this isolated boundary, it
is not worth adding production cache/guard machinery.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None

from protenix.model.modules.fused_elementwise_triton import (
    triton_sigmoid_mul_heads_first,
)
from protenix.model.triangular.layers import cuequivariance_triangular_attn
from protenix.model.triangular.triangular import TriangleAttention, cueq_bool_mask_enabled


_BLOCK_SIZE = 256


if triton is not None and tl is not None:

    @triton.jit
    def _packed_projection_unpack_kernel(
        packed_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        gate_ptr,
        bias_ptr,
        total_head_elements: tl.constexpr,
        total_bias_elements: tl.constexpr,
        j_count: tl.constexpr,
        i_count: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        q_offset: tl.constexpr,
        k_offset: tl.constexpr,
        v_offset: tl.constexpr,
        gate_offset: tl.constexpr,
        bias_offset: tl.constexpr,
        packed_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0) * block_size + tl.arange(0, block_size)
        ).to(tl.int64)
        head_mask = offsets < total_head_elements

        d = offsets % head_dim
        tmp = offsets // head_dim
        j = tmp % j_count
        tmp = tmp // j_count
        h = tmp % head_count
        row = tmp // head_count

        hidden = head_count * head_dim
        packed_base = (row * j_count + j) * packed_width + h * head_dim + d
        tl.store(
            q_ptr + offsets,
            tl.load(packed_ptr + packed_base + q_offset, mask=head_mask),
            mask=head_mask,
        )
        tl.store(
            k_ptr + offsets,
            tl.load(packed_ptr + packed_base + k_offset, mask=head_mask),
            mask=head_mask,
        )
        tl.store(
            v_ptr + offsets,
            tl.load(packed_ptr + packed_base + v_offset, mask=head_mask),
            mask=head_mask,
        )
        gate_offsets = row * j_count * hidden + j * hidden + h * head_dim + d
        tl.store(
            gate_ptr + gate_offsets,
            tl.load(packed_ptr + packed_base + gate_offset, mask=head_mask),
            mask=head_mask,
        )

        bias_offsets = offsets
        bias_mask = bias_offsets < total_bias_elements
        bj = bias_offsets % j_count
        tmp = bias_offsets // j_count
        bi = tmp % i_count
        tmp = tmp // i_count
        bh = tmp % head_count
        brow = tmp // head_count
        packed_bias = ((brow * i_count + bi) * j_count + bj) * packed_width
        packed_bias += bias_offset + bh
        tl.store(
            bias_ptr + bias_offsets,
            tl.load(packed_ptr + packed_bias, mask=bias_mask).to(tl.float32),
            mask=bias_mask,
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


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
        starting=not args.ending,
    ).cuda()
    module.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 8191)
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(args.batch, args.tokens, args.tokens, args.c_z, device="cuda", dtype=dtype)
    mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)
    return x, mask


def packed_weight(module: TriangleAttention, dtype: torch.dtype) -> torch.Tensor:
    weights = [
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
        module.mha.linear_g.weight,
        module.linear.weight,
    ]
    if dtype is torch.bfloat16:
        weights = [weight.to(dtype=dtype) for weight in weights]
    return torch.cat(weights, dim=0).contiguous()


def unpack_packed_projection(
    packed: torch.Tensor,
    module: TriangleAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for the packed projection probe")
    head_count = module.no_heads
    head_dim = module.mha.c_hidden
    hidden = head_count * head_dim
    output_shape = packed.shape[:-2] + (head_count, packed.shape[-2], head_dim)
    q = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    gate = torch.empty(packed.shape[:-1] + (hidden,), dtype=packed.dtype, device=packed.device)
    bias = torch.empty(
        packed.shape[:-3] + (1, head_count, packed.shape[-3], packed.shape[-2]),
        dtype=torch.float32,
        device=packed.device,
    )

    rows = math.prod(packed.shape[:-2])
    head_elements = rows * head_count * packed.shape[-2] * head_dim
    bias_elements = math.prod(bias.shape)
    grid = (triton.cdiv(max(head_elements, bias_elements), _BLOCK_SIZE),)
    _packed_projection_unpack_kernel[grid](
        packed,
        q,
        k,
        v,
        gate,
        bias,
        head_elements,
        bias_elements,
        packed.shape[-2],
        packed.shape[-3],
        head_count,
        head_dim,
        0,
        hidden,
        2 * hidden,
        3 * hidden,
        4 * hidden,
        packed.shape[-1],
        _BLOCK_SIZE,
        num_warps=4,
    )
    return q, k, v, gate, bias


def module_forward(module: TriangleAttention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return module(x, mask=mask, triangle_attention="cuequivariance")


def packed_forward(
    module: TriangleAttention,
    x_in: torch.Tensor,
    mask_in: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x = x_in
    mask = mask_in
    if not module.starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    x = module.layer_norm(x)
    with torch.amp.autocast("cuda", enabled=False):
        packed = F.linear(x, weight)
    q, k, v, gate, triangle_bias = unpack_packed_projection(packed, module)
    if cueq_bool_mask_enabled():
        mask_for_attention = (mask == 1)[..., :, None, None, :]
    else:
        mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
        mask_for_attention = (mask_bias == 0).bool()
    scale = 1.0 / math.sqrt(module.c_hidden)
    heads_first = cuequivariance_triangular_attn(q, k, v, triangle_bias, mask_for_attention, scale)
    if isinstance(heads_first, tuple):
        heads_first = heads_first[0]
    out = triton_sigmoid_mul_heads_first(gate, heads_first)
    if out is None:
        out = heads_first.transpose(-2, -3)
        out = out * torch.sigmoid(gate).view(gate.shape[:-1] + (module.no_heads, -1))
        out = out.reshape(gate.shape)
    out = module.mha.linear_o(out)
    if not module.starting:
        out = out.transpose(-2, -3)
    return out


def cuda_time(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    with torch.inference_mode():
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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--ending", type=str_bool, default=False)
    parser.add_argument("--input-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_packed_projection_probe requires CUDA")
    if triton is None or tl is None:
        raise RuntimeError("triangle_attention_packed_projection_probe requires Triton")
    torch.manual_seed(args.seed)
    module = make_module(args)
    x, mask = make_inputs(args)
    dtype = getattr(torch, args.input_dtype)
    weight = packed_weight(module, dtype)

    with torch.inference_mode(), cuda_autocast(args.compute_dtype):
        reference, module_ms = cuda_time(lambda: module_forward(module, x, mask), args.warmup, args.iters)
        candidate, packed_ms = cuda_time(lambda: packed_forward(module, x, mask, weight), args.warmup, args.iters)

    print(
        json.dumps(
            {
                "args": vars(args),
                "module_forward_ms": module_ms,
                "packed_projection_ms": packed_ms,
                "speedup": module_ms / packed_ms,
                "parity_vs_module": compare(candidate, reference),
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
