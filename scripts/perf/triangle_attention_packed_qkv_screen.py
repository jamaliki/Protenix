#!/usr/bin/env python3
"""Screen a packed-qkv triangle-attention kernel boundary.

The promoted CUEQ path first projects ``x`` to packed ``[..., 3*H*D]`` qkv,
then launches a Triton layout producer that materializes separate contiguous
q/k/v tensors for CUEQ/cuDNN.  NCU showed the CUEQ attention kernel is L2-bound,
so rereading those materialized q/k/v tensors is a credible bottleneck.

This screen adapts Protenix's existing Triton triangle-attention algorithm to
read q/k/v directly from the packed projection and write the heads-first output
expected by the fused epilogue.  It is a boundary test for a future CuTe/CUDA
kernel, not a promoted model path.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from protenix.model.triangular.triangular import TriangleAttention


@contextmanager
def cuda_autocast(dtype_name: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype else nullcontext()
    with ctx:
        yield


@triton.jit
def _packed_qkv_tri_attn_kernel(
    qkv_ptr,
    bias_ptr,
    mask_ptr,
    out_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    input_dtype = qkv_ptr.dtype.element_ty
    block_m = tl.program_id(0)
    h = tl.program_id(1)
    bn = tl.program_id(2)
    b = bn // N
    n = bn - b * N

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    hidden = H * HEAD_DIM
    packed = 3 * hidden

    row_base = ((b * N + n) * S) * packed
    q = tl.load(
        qkv_ptr + row_base + offs_m[:, None] * packed + h * HEAD_DIM + offs_d[None, :],
        mask=offs_m[:, None] < S,
        other=0.0,
    )

    bias_scale = 1.4426950408889634
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    key_offsets = offs_n
    for _ in range(tl.cdiv(S, BLOCK_N)):
        k = tl.load(
            qkv_ptr
            + row_base
            + key_offsets[:, None] * packed
            + hidden
            + h * HEAD_DIM
            + offs_d[None, :],
            mask=key_offsets[:, None] < S,
            other=0.0,
        )
        v = tl.load(
            qkv_ptr
            + row_base
            + key_offsets[:, None] * packed
            + 2 * hidden
            + h * HEAD_DIM
            + offs_d[None, :],
            mask=key_offsets[:, None] < S,
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + ((b * N + n) * S + key_offsets) * H + h,
            mask=key_offsets < S,
            other=0.0,
        ).to(tl.float32)
        key_mask = tl.load(
            mask_ptr + (b * N + n) * S + key_offsets,
            mask=key_offsets < S,
            other=0.0,
        )
        valid = (offs_m[:, None] < S) & (key_offsets[None, :] < S) & (
            key_mask[None, :] > 0.5
        )

        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * QK_SCALE
        qk += bias[None, :] * bias_scale
        qk = tl.where(valid, qk, -1.0e6)
        qk = tl.where(offs_m[:, None] < S, qk, 0.0)

        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_ij = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(input_dtype), v, acc, input_precision=INPUT_PRECISION)
        m_i = m_ij
        l_i = l_ij
        key_offsets += BLOCK_N

    acc = (acc / l_i[:, None]).to(q.dtype)
    out_base = ((b * N + n) * H + h) * S * HEAD_DIM
    tl.store(
        out_ptr + out_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < S,
    )


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
    ).cuda()
    module.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 65537)
            for parameter in module.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(args: argparse.Namespace):
    dtype = getattr(torch, args.input_dtype)
    x = torch.randn(args.batch, args.tokens, args.tokens, args.c_z, device="cuda", dtype=dtype)
    mask = torch.ones(args.batch, args.tokens, args.tokens, device="cuda", dtype=dtype)
    return x, mask


def packed_qkv_projection(module: TriangleAttention, x: torch.Tensor) -> torch.Tensor:
    weights = [module.mha.linear_q.weight, module.mha.linear_k.weight, module.mha.linear_v.weight]
    if x.dtype == torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            return F.linear(x, torch.cat([weight.to(dtype=x.dtype) for weight in weights]))
    return F.linear(x, torch.cat(weights))


def packed_forward(module: TriangleAttention, x: torch.Tensor, mask: torch.Tensor, args):
    x = module.layer_norm(x)
    qkv = packed_qkv_projection(module, x)
    # Match the CUEQ path: the triangle bias is deliberately promoted to FP32
    # before attention even when activations are BF16.
    bias = module.linear(x).float().contiguous()
    batch, rows, cols, _ = x.shape
    out = torch.empty(
        batch,
        rows,
        module.no_heads,
        cols,
        module.c_hidden,
        device=x.device,
        dtype=qkv.dtype,
    )
    grid = (triton.cdiv(cols, args.block_m), module.no_heads, batch * rows)
    _packed_qkv_tri_attn_kernel[grid](
        qkv,
        bias,
        mask,
        out,
        rows,
        cols,
        module.no_heads,
        module.c_hidden,
        1.4426950408889634 / math.sqrt(module.c_hidden),
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        INPUT_PRECISION=args.input_precision,
        num_warps=args.num_warps,
    )
    return module.mha._wrap_up_heads_first(out, x)


def cuda_time(fn, warmup: int, iters: int):
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


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return float(finite.max().item()) if finite.numel() else float("nan")


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return float(finite.mean().item()) if finite.numel() else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument("--input-precision", choices=["tf32", "tf32x3", "ieee"], default="tf32")
    parser.add_argument("--input-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_packed_qkv_screen requires CUDA")
    torch.manual_seed(args.seed)
    module = make_module(args)
    x, mask = make_inputs(args)
    with cuda_autocast(args.compute_dtype):
        cueq_out, cueq_ms = cuda_time(
            lambda: module(x, mask=mask, triangle_attention="cuequivariance"),
            args.warmup,
            args.iters,
        )
        packed_out, packed_ms = cuda_time(
            lambda: packed_forward(module, x, mask, args),
            args.warmup,
            args.iters,
        )
    print(json.dumps({
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "cueq_ms": cueq_ms,
        "packed_qkv_ms": packed_ms,
        "speedup_vs_cueq": cueq_ms / packed_ms,
        "max_abs_error": max_abs_error(cueq_out, packed_out),
        "mean_abs_error": mean_abs_error(cueq_out, packed_out),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
