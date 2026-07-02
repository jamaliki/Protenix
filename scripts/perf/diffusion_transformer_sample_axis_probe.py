#!/usr/bin/env python3
"""Probe the sample-axis boundary inside one diffusion transformer block.

The ragged low-sample path currently flattens ``(record, sample)`` before the
token diffusion transformer:

    a/mask: [B * S, N, C]
    s:      [B, 1, N, C_s] -> repeated to [B * S, N, C_s]
    z bias: [B, H, N, N] -> repeated to [B * S, H, N, N]

Flattening is known-good because Protenix's generic attention helper keeps this
rank-4 shape on the fast BF16 SDPA path.  The cost is repeated bias traffic.

This probe asks the next implementation question at the real block boundary:
would keeping ``a`` as ``[B, S, N, C]`` and broadcasting the sample-invariant
``s`` and bias over ``S`` be faster after including q/k/v projections, gates,
residuals, and the conditioned transition?  It also times the
conservative FP32 rank-5 attention policy that Protenix would use today.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from protenix.model.modules.fused_elementwise_triton import fused_sigmoid_mul
from protenix.model.modules.transformer import DiffusionTransformerBlock


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _make_token_mask(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    min_fraction = max(0.0, min(1.0, args.valid_fraction))
    if min_fraction >= 1.0:
        return torch.ones(args.batch, args.tokens, device=device, dtype=torch.bool)
    min_len = max(1, int(round(args.tokens * min_fraction)))
    lengths = torch.linspace(min_len, args.tokens, steps=args.batch, device=device)
    positions = torch.arange(args.tokens, device=device)
    return positions[None, :] < lengths.round().to(dtype=torch.long)[:, None]


def _flatten_sample_axis(tensor: torch.Tensor, samples: int) -> torch.Tensor:
    if tensor.shape[1] == 1 and samples > 1:
        tensor = tensor.expand(tensor.shape[0], samples, *tensor.shape[2:])
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_mask(mask: torch.Tensor, samples: int) -> torch.Tensor:
    return mask[:, None, :].expand(mask.shape[0], samples, mask.shape[1]).reshape(
        mask.shape[0] * samples, mask.shape[1]
    )


def _project_pair_bias(block: DiffusionTransformerBlock, z_cf: torch.Tensor) -> torch.Tensor:
    apb = block.attention_pair_bias
    weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[
        :, :, None, None
    ]
    return F.conv2d(z_cf, weight)


def _rank5_attn_mask(
    block: DiffusionTransformerBlock, z_cf: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    bias = _project_pair_bias(block, z_cf)[:, None]
    if bool(token_mask.all().item()):
        return bias
    key_bias = (token_mask.to(dtype=bias.dtype) - 1) * 1e4
    return bias + key_bias[:, None, None, None, :]


def _rank5_attention_pair_bias(
    block: DiffusionTransformerBlock,
    a: torch.Tensor,
    s: torch.Tensor,
    z_cf: torch.Tensor,
    token_mask: torch.Tensor,
    fp32_attention: bool,
) -> torch.Tensor:
    """AttentionPairBias.forward with rank-5 q/k/v and broadcasted bias."""
    apb = block.attention_pair_bias
    q_x = apb.layernorm_a(a=a, s=s)
    q, k, v = apb.attention._prep_qkv(q_x=q_x, kv_x=q_x, apply_scale=True)
    attn_mask = _rank5_attn_mask(block, z_cf, token_mask)
    if fp32_attention:
        o = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=attn_mask.float(),
        ).to(dtype=q.dtype)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    o = apb.attention._wrap_up(o.transpose(-2, -3), q_x)
    return fused_sigmoid_mul(apb.linear_a_last(s), o)


def _rank5_block(
    block: DiffusionTransformerBlock,
    a: torch.Tensor,
    s: torch.Tensor,
    z_cf: torch.Tensor,
    token_mask: torch.Tensor,
    fp32_attention: bool,
) -> torch.Tensor:
    attn_out = _rank5_attention_pair_bias(
        block=block,
        a=a,
        s=s,
        z_cf=z_cf,
        token_mask=token_mask,
        fp32_attention=fp32_attention,
    )
    attn_out = attn_out + a
    ff_out = block.conditioned_transition_block(a=attn_out, s=s)
    return ff_out + attn_out


def _flat_block(
    block: DiffusionTransformerBlock,
    a: torch.Tensor,
    s: torch.Tensor,
    z_cf: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    samples = a.shape[1]
    out, _s, _z = block(
        a=_flatten_sample_axis(a, samples),
        s=_flatten_sample_axis(s, samples),
        z=z_cf,
        inplace_safe=False,
        chunk_size=None,
        enable_efficient_fusion=True,
        token_mask=_flatten_mask(token_mask, samples),
        z_sample_count=samples,
    )
    return out.reshape(*a.shape)


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
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    return out, start.elapsed_time(end) / iters, peak_mib


def _compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _record_case(
    name: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor | None,
    args: argparse.Namespace,
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
        row["parity_vs_flat_current"] = _compare(out, reference)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_transformer_sample_axis_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)

    block = DiffusionTransformerBlock(
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        n_heads=args.heads,
        cross_attention_mode=False,
    ).cuda()
    block.eval()

    scale = 1.0 / math.sqrt(args.c_a)
    a = scale * torch.randn(
        args.batch, args.samples, args.tokens, args.c_a, device=device, dtype=dtype
    )
    s = scale * torch.randn(
        args.batch, 1, args.tokens, args.c_s, device=device, dtype=dtype
    )
    z_cf = scale * torch.randn(
        args.batch, args.c_z, args.tokens, args.tokens, device=device, dtype=dtype
    )
    token_mask = _make_token_mask(args, device)

    amp = torch.autocast(device_type="cuda", dtype=dtype)
    cases: list[dict[str, Any]] = []
    with amp:
        flat_out = _flat_block(block, a, s, z_cf, token_mask)
        torch.cuda.synchronize()
        cases.append(
            _record_case(
                "flat_current_repeat_bias",
                lambda: _flat_block(block, a, s, z_cf, token_mask),
                reference=None,
                args=args,
            )
        )
        cases.append(
            _record_case(
                "rank5_broadcast_bias_bf16_attention",
                lambda: _rank5_block(
                    block, a, s, z_cf, token_mask, fp32_attention=False
                ),
                reference=flat_out,
                args=args,
            )
        )
        cases.append(
            _record_case(
                "rank5_broadcast_bias_protenix_current_fp32_policy",
                lambda: _rank5_block(
                    block, a, s, z_cf, token_mask, fp32_attention=True
                ),
                reference=flat_out,
                args=args,
            )
        )

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "shapes": {"a": list(a.shape), "s": list(s.shape), "z_cf": list(z_cf.shape)},
        "token_mask_valid_fraction": float(token_mask.float().mean().item()),
        "cases": cases,
        "timestamp": time.time(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
