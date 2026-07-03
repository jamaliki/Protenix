#!/usr/bin/env python3
"""Screen CUEQ's fused diffusion ``attention_pair_bias`` primitive.

The promoted low-sample diffusion path currently does this in Python/PyTorch:

1. project q/k/v from the normalized token activation;
2. project pair ``z`` to per-head attention bias with a 1x1 ``conv2d``;
3. repeat that bias over diffusion samples;
4. run cuDNN/SDPA;
5. apply the attention gate and output projection.

CUEQ 0.10 ships an ``attention_pair_bias`` primitive for this exact algebra.
If it can keep cuDNN-class attention speed while folding the pair-bias/gate/out
boundary, it is a legitimate "meat of the problem" candidate.  If it loses,
the reason is also useful: Protenix's diffusion head dimension is 48, while the
CUEQ docs say the fastest path prefers head dimensions that are multiples of
32.

This is only a hotspot screen.  A win here must still pass the full
mixed-sequence ``N_sample=5``, ``N_step=200`` inference gate before promotion.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import _repo_bootstrap  # noqa: F401

import torch
import torch.nn.functional as F

from protenix.model.modules.transformer import AttentionPairBias


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _token_mask(batch: int, tokens: int, valid_fraction: float) -> torch.Tensor:
    if valid_fraction >= 1.0:
        return torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    min_len = max(1, int(round(tokens * max(0.0, min(1.0, valid_fraction)))))
    lengths = torch.linspace(min_len, tokens, steps=batch, device="cuda").round()
    positions = torch.arange(tokens, device="cuda")
    return positions[None, :] < lengths.to(dtype=torch.long)[:, None]


def _flatten_mask(mask: torch.Tensor, samples: int) -> torch.Tensor:
    return mask[:, None, :].expand(mask.shape[0], samples, mask.shape[1]).reshape(
        mask.shape[0] * samples,
        mask.shape[1],
    )


def _cuda_time(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
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
    return out, start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _record(
    name: str,
    fn: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    try:
        out, mean_ms, peak_mib = _cuda_time(fn, args.warmup, args.iters)
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        return {"name": name, "ok": False, "error": repr(exc)}

    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "peak_allocated_mib": peak_mib,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }
    if reference is not None:
        row["parity_vs_baseline"] = _compare(out, reference)
    return row


def _make_module(args: argparse.Namespace) -> AttentionPairBias:
    module = AttentionPairBias(
        has_s=True,
        create_offset_ln_z=False,
        n_heads=args.heads,
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        cross_attention_mode=False,
    ).cuda()
    module.eval()
    return module


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    dtype = _dtype(args.dtype)
    scale = 1.0 / math.sqrt(args.c_a)
    a = scale * torch.randn(
        args.batch * args.samples,
        args.tokens,
        args.c_a,
        device="cuda",
        dtype=dtype,
    )
    s_record = scale * torch.randn(
        args.batch,
        args.tokens,
        args.c_s,
        device="cuda",
        dtype=dtype,
    )
    s = s_record[:, None].expand(
        args.batch,
        args.samples,
        args.tokens,
        args.c_s,
    ).reshape(args.batch * args.samples, args.tokens, args.c_s)
    z = scale * torch.randn(
        args.batch,
        args.tokens,
        args.tokens,
        args.c_z,
        device="cuda",
        dtype=dtype,
    )
    token_mask_record = _token_mask(args.batch, args.tokens, args.valid_fraction)
    return {
        "a": a.contiguous(),
        "s": s.contiguous(),
        "z": z.contiguous(),
        "z_channel_first": z.permute(0, 3, 1, 2).contiguous(),
        "token_mask_record": token_mask_record,
        "token_mask_flat": _flatten_mask(token_mask_record, args.samples),
    }


def _amp_context(dtype: torch.dtype):
    if dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _cueq_attention_pair_bias(
    module: AttentionPairBias,
    a_norm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    token_mask: torch.Tensor,
    layernorm_z_bias: torch.Tensor | None,
    *,
    return_z_proj: bool,
) -> torch.Tensor:
    from cuequivariance_torch import attention_pair_bias

    result = attention_pair_bias(
        s=a_norm,
        q=q,
        k=k,
        v=v,
        z=z,
        mask=token_mask,
        num_heads=module.n_heads,
        w_proj_z=module.linear_nobias_z.weight,
        w_proj_g=module.attention.linear_g.weight,
        w_proj_o=module.attention.linear_o.weight,
        w_ln_z=module.layernorm_z.weight,
        b_ln_z=layernorm_z_bias,
        inf=1e4,
        eps=module.layernorm_z.eps,
        attn_scale=1.0,
        return_z_proj=return_z_proj,
        is_cached_z_proj=False,
    )
    if isinstance(result, tuple):
        result = result[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--disable-cudnn-sdpa",
        action="store_true",
        help="Disable cuDNN SDPA before calling CUEQ's internal attention wrapper.",
    )
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "baseline_full",
            "baseline_after_qkv",
            "cueq_full",
            "cueq_after_qkv",
        ],
        default="all",
        help=(
            "Run one row in its own process. Useful because native CUEQ "
            "segfaults terminate Python before the full JSON summary is written."
        ),
    )
    return parser.parse_args()


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_attention_pair_bias_cueq_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.disable_cudnn_sdpa and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        # CUEQ's attention_pair_bias wrapper does not catch cuDNN frontend
        # plan-build failures.  Protenix's own SDPA wrapper does, so keep this
        # as an explicit diagnostic rather than silently changing defaults.
        torch.backends.cuda.enable_cudnn_sdp(False)

    dtype = _dtype(args.dtype)
    module = _make_module(args)
    inputs = _make_inputs(args)
    amp = _amp_context(dtype)
    add_key_mask = inputs["token_mask_flat"] is not None

    def normalize_a() -> torch.Tensor:
        with amp:
            return module.layernorm_a(a=inputs["a"], s=inputs["s"])

    a_norm = normalize_a()
    layernorm_z_bias = module.layernorm_z.bias
    if layernorm_z_bias is None:
        # CUEQ 0.10's native pair-bias Triton kernel checks whether LayerNorm
        # has any affine parameter, then unconditionally reads both weight and
        # bias pointers.  Protenix's mathematically equivalent no-offset
        # LayerNorm has weight but no bias, so pass an explicit zero vector for
        # this probe instead of tripping that kernel compilation bug.
        layernorm_z_bias = torch.zeros_like(module.layernorm_z.weight)
    with amp:
        # Protenix scales q before SDPA and calls SDPA with scale=1.0.  The
        # CUEQ primitive gets the same pre-scaled q and attn_scale=1.0 so this
        # remains an implementation comparison, not a hidden math change.
        q, k, v = module.attention._prep_qkv(a_norm, a_norm, apply_scale=True)
    torch.cuda.synchronize()

    def baseline_full() -> torch.Tensor:
        with amp:
            current_a_norm = normalize_a()
            return module.standard_multihead_attention(
                q=current_a_norm,
                kv=current_a_norm,
                z=inputs["z_channel_first"],
                token_mask=inputs["token_mask_flat"],
                enable_efficient_fusion=True,
                z_sample_count=args.samples,
                z_sample_axis=False,
            )

    def baseline_after_qkv() -> torch.Tensor:
        with amp:
            weight = (
                module.linear_nobias_z.weight * module.layernorm_z.weight[None, :]
            )[:, :, None, None]
            bias = F.conv2d(inputs["z_channel_first"], weight)
            bias = (
                bias[:, None]
                .expand(args.batch, args.samples, args.heads, args.tokens, args.tokens)
                .reshape(args.batch * args.samples, args.heads, args.tokens, args.tokens)
            )
            if add_key_mask:
                key_bias = (inputs["token_mask_flat"].to(dtype=bias.dtype) - 1) * 1e4
                bias = bias + key_bias[:, None, None, :]
            o = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                scale=1.0,
            )
            o = o.transpose(-2, -3)
            return module.attention._wrap_up(o, a_norm)

    def cueq_full() -> torch.Tensor:
        with amp:
            current_a_norm = normalize_a()
            current_q, current_k, current_v = module.attention._prep_qkv(
                current_a_norm,
                current_a_norm,
                apply_scale=True,
            )
            return _cueq_attention_pair_bias(
                module,
                current_a_norm,
                current_q,
                current_k,
                current_v,
                inputs["z"],
                inputs["token_mask_record"],
                layernorm_z_bias,
                return_z_proj=False,
            )

    def cueq_after_qkv() -> torch.Tensor:
        with amp:
            return _cueq_attention_pair_bias(
                module,
                a_norm,
                q,
                k,
                v,
                inputs["z"],
                inputs["token_mask_record"],
                layernorm_z_bias,
                return_z_proj=False,
            )

    rows: list[dict[str, Any]] = []
    reference: torch.Tensor | None = None
    if args.case in {"all", "baseline_full", "baseline_after_qkv"}:
        reference, baseline_ms, baseline_peak = _cuda_time(
            baseline_full,
            args.warmup,
            args.iters,
        )
        rows.append(
            {
                "name": "baseline_full_conv2d_sdpa",
                "ok": True,
                "mean_ms": baseline_ms,
                "peak_allocated_mib": baseline_peak,
                "output_shape": list(reference.shape),
                "output_dtype": str(reference.dtype),
            }
        )

    if args.case in {"all", "baseline_after_qkv"}:
        rows.append(
            _record(
                "baseline_after_qkv_projection",
                baseline_after_qkv,
                args,
                reference,
            )
        )
    if args.case in {"all", "cueq_full"}:
        rows.append(_record("cueq_full", cueq_full, args, reference))
    if args.case in {"all", "cueq_after_qkv"}:
        rows.append(_record("cueq_after_qkv_projection", cueq_after_qkv, args, reference))

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "head_dim": args.c_a // args.heads,
        "input_shapes": {name: list(tensor.shape) for name, tensor in inputs.items()},
        "qkv_shapes": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)},
        "rows": rows,
        "timestamp": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
