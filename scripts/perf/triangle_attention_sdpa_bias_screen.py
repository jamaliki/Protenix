#!/usr/bin/env python3
"""Screen SDPA as a triangle-attention replacement.

Triangle attention has a factorized per-key bias.  For a row/head score

    softmax((q @ k.T) * scale + bias_key) @ v

we can append one synthetic Q/K channel and choose ``k_extra = bias_key/scale``.
Then ordinary SDPA sees the same logits without a full [Q, K] bias tensor.  This
is not the final CuTe kernel, but it is a cheap test of the intended fusion
boundary: can a Flash/SDPA-shaped kernel beat CUEQ if it avoids custom bias
handling, and how much overhead comes from materializing the augmented Q/K?
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from protenix.model.triangular.triangular import TriangleAttention
from protenix.model.utils import permute_final_dims


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
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype else nullcontext()
    with ctx:
        yield


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


def cueq_forward(module: TriangleAttention, x: torch.Tensor, mask: torch.Tensor):
    return module(x, mask=mask, triangle_attention="cuequivariance")


def sdpa_bias_forward(
    module: TriangleAttention,
    x: torch.Tensor,
    mask: torch.Tensor,
    padded_dim: int,
    full_mask: bool,
):
    x = module.layer_norm(x)
    batch, rows, cols, _ = x.shape
    heads = module.no_heads
    head_dim = module.c_hidden
    scale = 1.0 / math.sqrt(head_dim)

    triangle_bias = permute_final_dims(module.linear(x), (2, 0, 1))
    triangle_bias = triangle_bias.permute(0, 2, 1, 3).reshape(
        batch * rows,
        heads,
        cols,
    )
    q, k, v = module.mha._prep_qkv(x, x, apply_scale=False, cueq_heads_first=False)
    q = q.reshape(batch * rows, heads, cols, head_dim).contiguous()
    k = k.reshape(batch * rows, heads, cols, head_dim).contiguous()
    v = v.reshape(batch * rows, heads, cols, head_dim).contiguous()

    q_aug = torch.zeros(
        *q.shape[:-1],
        padded_dim,
        dtype=q.dtype,
        device=q.device,
    )
    k_aug = torch.zeros_like(q_aug)
    q_aug[..., :head_dim] = q
    k_aug[..., :head_dim] = k
    q_aug[..., head_dim] = 1.0
    k_aug[..., head_dim] = triangle_bias / scale

    attn_mask = None
    if not full_mask:
        attn_mask = mask.reshape(batch * rows, 1, 1, cols).bool()

    o = F.scaled_dot_product_attention(
        q_aug,
        k_aug,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    o = o.reshape(batch, rows, heads, cols, head_dim)
    return module.mha._wrap_up_heads_first(o, x)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=4)
    parser.add_argument("--padded-dim", type=int, default=40)
    dtypes = ["float32", "bfloat16", "float16"]
    parser.add_argument("--input-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.padded_dim <= args.c_hidden_pair_att:
        raise ValueError("--padded-dim must leave room for the synthetic bias channel")
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_sdpa_bias_screen requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    module = make_module(args)
    x, mask = make_inputs(args)
    full_mask = bool(torch.all(mask == 1).item())
    with cuda_autocast(args.compute_dtype):
        cueq_out, cueq_ms = cuda_time(lambda: cueq_forward(module, x, mask), args.warmup, args.iters)
        sdpa_out, sdpa_ms = cuda_time(
            lambda: sdpa_bias_forward(module, x, mask, args.padded_dim, full_mask),
            args.warmup,
            args.iters,
        )

    print(
        json.dumps(
            {
                "args": vars(args),
                "device": torch.cuda.get_device_name(),
                "cueq_ms": cueq_ms,
                "sdpa_bias_ms": sdpa_ms,
                "speedup_vs_cueq": cueq_ms / sdpa_ms,
                "max_abs_error": max_abs_error(cueq_out, sdpa_out),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
