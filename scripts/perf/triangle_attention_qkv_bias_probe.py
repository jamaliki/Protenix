#!/usr/bin/env python3
"""Screen a narrow CUEQ triangle-attention producer fusion.

The promoted CUEQ triangle-attention path already uses one GEMM for q/k/v and a
Triton copy kernel to write CUEQ's preferred ``[..., I, H, J, D]`` layout.
Profiles still show a separate tiny triangle-bias projection next to that
boundary.  This probe tests the minimal next fusion:

    current:  LayerNorm -> qkv GEMM -> qkv layout
                      \-> bias GEMM -> permute
    candidate: LayerNorm -> [qkv, bias] GEMM -> qkv+bias layout

The gate projection and output projection deliberately stay unchanged.  A
previous, broader qkv+gate+bias packed producer was slower; this script asks
whether the smaller bias-only variant has enough signal to justify production
code.  It is benchmark-only and should be promoted only after a real model gate.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency.
    triton = None
    tl = None

from protenix.model.modules.fused_elementwise_triton import (
    triton_sigmoid_mul_heads_first,
    triton_sigmoid_mul_heads_first_swapped_gate,
)
from protenix.model.triangular.layers import flatten_final_dims
from protenix.model.triangular.layers import cuequivariance_triangular_attn
from protenix.model.triangular.triangular import TriangleAttention, cueq_bool_mask_enabled


_BLOCK_SIZE = 256


if triton is not None and tl is not None:

    @triton.jit
    def _qkv_bias_to_cueq_kernel(
        packed_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        total_head_elements: tl.constexpr,
        total_bias_elements: tl.constexpr,
        i_count: tl.constexpr,
        j_count: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        packed_width: tl.constexpr,
        swapped: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0) * block_size + tl.arange(0, block_size)
        ).to(tl.int64)

        head_mask = offsets < total_head_elements
        d = offsets % head_dim
        tmp = offsets // head_dim
        if swapped:
            i = tmp % i_count
            tmp = tmp // i_count
            h = tmp % head_count
            tmp = tmp // head_count
            j = tmp % j_count
            batch = tmp // j_count
        else:
            j = tmp % j_count
            tmp = tmp // j_count
            h = tmp % head_count
            tmp = tmp // head_count
            i = tmp % i_count
            batch = tmp // i_count

        hidden = head_count * head_dim
        input_base = ((batch * i_count + i) * j_count + j) * packed_width
        input_base += h * head_dim + d
        tl.store(
            q_ptr + offsets,
            tl.load(packed_ptr + input_base, mask=head_mask),
            mask=head_mask,
        )
        tl.store(
            k_ptr + offsets,
            tl.load(packed_ptr + input_base + hidden, mask=head_mask),
            mask=head_mask,
        )
        tl.store(
            v_ptr + offsets,
            tl.load(packed_ptr + input_base + 2 * hidden, mask=head_mask),
            mask=head_mask,
        )

        bias_offsets = offsets
        bias_mask = bias_offsets < total_bias_elements
        if swapped:
            bi = bias_offsets % i_count
            tmp = bias_offsets // i_count
            bj = tmp % j_count
            tmp = tmp // j_count
            bh = tmp % head_count
            bb = tmp // head_count
        else:
            bj = bias_offsets % j_count
            tmp = bias_offsets // j_count
            bi = tmp % i_count
            tmp = tmp // i_count
            bh = tmp % head_count
            bb = tmp // head_count
        packed_bias = ((bb * i_count + bi) * j_count + bj) * packed_width
        packed_bias += 3 * hidden + bh
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
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 8191)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_z,
        device="cuda",
        dtype=dtype,
    )
    mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)
    return x, mask


def qkv_bias_weight(module: TriangleAttention, dtype: torch.dtype) -> torch.Tensor:
    weights = [
        module.mha.linear_q.weight,
        module.mha.linear_k.weight,
        module.mha.linear_v.weight,
        module.linear.weight,
    ]
    if dtype is torch.bfloat16:
        weights = [weight.to(dtype=dtype) for weight in weights]
    return torch.cat(weights, dim=0).contiguous()


def unpack_qkv_bias(
    packed: torch.Tensor,
    module: TriangleAttention,
    *,
    swapped: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for the qkv+bias probe")

    head_count = module.no_heads
    head_dim = module.mha.c_hidden
    if swapped:
        q_shape = packed.shape[:-3] + (
            packed.shape[-2],
            head_count,
            packed.shape[-3],
            head_dim,
        )
        bias_shape = packed.shape[:-3] + (
            1,
            head_count,
            packed.shape[-2],
            packed.shape[-3],
        )
    else:
        q_shape = packed.shape[:-2] + (head_count, packed.shape[-2], head_dim)
        bias_shape = packed.shape[:-3] + (
            1,
            head_count,
            packed.shape[-3],
            packed.shape[-2],
        )

    q = torch.empty(q_shape, dtype=packed.dtype, device=packed.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    bias = torch.empty(bias_shape, dtype=torch.float32, device=packed.device)

    head_elements = math.prod(q_shape)
    bias_elements = math.prod(bias_shape)
    grid = (triton.cdiv(max(head_elements, bias_elements), _BLOCK_SIZE),)
    _qkv_bias_to_cueq_kernel[grid](
        packed,
        q,
        k,
        v,
        bias,
        head_elements,
        bias_elements,
        packed.shape[-3],
        packed.shape[-2],
        head_count,
        head_dim,
        packed.shape[-1],
        swapped,
        _BLOCK_SIZE,
        num_warps=4,
    )
    return q, k, v, bias


def module_forward(module: TriangleAttention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return module(x, mask=mask, triangle_attention="cuequivariance")


def candidate_forward(
    module: TriangleAttention,
    x_in: torch.Tensor,
    mask_in: torch.Tensor,
    weights_by_dtype: dict[torch.dtype, torch.Tensor],
) -> torch.Tensor:
    x = module.layer_norm(x_in)
    weight = weights_by_dtype.get(x.dtype)
    if weight is None:
        weight = qkv_bias_weight(module, x.dtype)
        weights_by_dtype[x.dtype] = weight

    with torch.amp.autocast("cuda", enabled=False):
        packed = F.linear(x, weight)
    q, k, v, triangle_bias = unpack_qkv_bias(
        packed,
        module,
        swapped=not module.starting,
    )

    if module.starting:
        mask_for_attention = (
            (mask_in == 1)[..., :, None, None, :]
            if cueq_bool_mask_enabled()
            else ((module.inf * (mask_in - 1))[..., :, None, None, :] == 0).bool()
        )
    else:
        mask_t = mask_in.transpose(-1, -2)
        mask_for_attention = (
            (mask_t == 1)[..., :, None, None, :]
            if cueq_bool_mask_enabled()
            else ((module.inf * (mask_t - 1))[..., :, None, None, :] == 0).bool()
        )

    heads_first = cuequivariance_triangular_attn(
        q,
        k,
        v,
        triangle_bias,
        mask_for_attention,
        1.0 / math.sqrt(module.c_hidden),
    )
    if isinstance(heads_first, tuple):
        heads_first = heads_first[0]

    if module.starting:
        return module.mha._wrap_up_heads_first(heads_first, x)

    if module.mha.linear_g is not None:
        gate = module.mha.linear_g(x)
        gated = triton_sigmoid_mul_heads_first_swapped_gate(gate, heads_first)
        if gated is None:
            gate = gate.transpose(-2, -3).contiguous()
            gated = triton_sigmoid_mul_heads_first(gate, heads_first)
        if gated is None:
            out = heads_first.transpose(-2, -3)
            gate = module.mha.sigmoid(gate).view(gate.shape[:-1] + (module.no_heads, -1))
            gated = flatten_final_dims(out * gate, 2)
    else:
        gated = flatten_final_dims(heads_first.transpose(-2, -3), 2)
    return module.mha.linear_o(gated).transpose(-2, -3)


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
    parser.add_argument(
        "--input-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_qkv_bias_probe requires CUDA")
    if triton is None or tl is None:
        raise RuntimeError("triangle_attention_qkv_bias_probe requires Triton")

    torch.manual_seed(args.seed)
    module = make_module(args)
    x, mask = make_inputs(args)
    input_dtype = getattr(torch, args.input_dtype)
    weights_by_dtype = {
        input_dtype: qkv_bias_weight(module, input_dtype),
        torch.float32: qkv_bias_weight(module, torch.float32),
    }

    with torch.inference_mode(), cuda_autocast(args.compute_dtype):
        reference, module_ms = cuda_time(
            lambda: module_forward(module, x, mask),
            args.warmup,
            args.iters,
        )
        candidate, candidate_ms = cuda_time(
            lambda: candidate_forward(module, x, mask, weights_by_dtype),
            args.warmup,
            args.iters,
        )

    print(
        json.dumps(
            {
                "args": vars(args),
                "current_ms": module_ms,
                "qkv_bias_ms": candidate_ms,
                "speedup": module_ms / candidate_ms,
                "parity_vs_current": compare(candidate, reference),
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
