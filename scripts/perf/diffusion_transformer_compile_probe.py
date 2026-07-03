#!/usr/bin/env python3
"""Screen `torch.compile` at the diffusion-token transformer boundary.

The post-atom H100 profile no longer has one obvious bad attention kernel in
the diffusion transformer.  Instead, one N20 trace shows thousands of small
launches around linears, SDPA, LayerNorm, sigmoid/gate, residual adds, and copy
traffic.  Before writing a bespoke CuTe kernel, this probe asks a cheaper and
more falsifiable question: can Inductor fuse enough of the existing PyTorch
plumbing to move the full transformer block stack?

The input layout matches the promoted low-sample inference path:

* `a` and `s` are flattened over `record * sample`;
* `z` stays at record grain in channel-first form `[B, C_z, N, N]`;
* each block projects pair bias once per record, then repeats the smaller
  `[B, H, N, N]` bias over sample lanes with `z_sample_count`.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from protenix.model.modules.transformer import DiffusionTransformer


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _token_mask(batch: int, tokens: int, valid_fraction: float) -> torch.Tensor:
    min_fraction = max(0.0, min(1.0, valid_fraction))
    if min_fraction >= 1.0:
        return torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    min_len = max(1, int(round(tokens * min_fraction)))
    lengths = torch.linspace(min_len, tokens, steps=batch, device="cuda").round()
    positions = torch.arange(tokens, device="cuda")
    return positions[None, :] < lengths.to(dtype=torch.long)[:, None]


def _flatten_samples(tensor: torch.Tensor, samples: int) -> torch.Tensor:
    if tensor.shape[1] == 1 and samples > 1:
        tensor = tensor.expand(tensor.shape[0], samples, *tensor.shape[2:])
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _flatten_mask(mask: torch.Tensor, samples: int) -> torch.Tensor:
    return mask[:, None, :].expand(mask.shape[0], samples, mask.shape[1]).reshape(
        mask.shape[0] * samples,
        mask.shape[1],
    )


class FlatDiffusionTransformer(nn.Module):
    """Wrapper with the static kwargs used by batched inference."""

    def __init__(self, model: DiffusionTransformer, samples: int) -> None:
        super().__init__()
        self.model = model
        self.samples = samples

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            a=a,
            s=s,
            z=z,
            inplace_safe=False,
            chunk_size=None,
            enable_efficient_fusion=True,
            token_mask=token_mask,
            z_sample_count=self.samples,
            z_sample_axis=False,
        )


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    scale = 1.0 / math.sqrt(args.c_a)
    dtype = _dtype(args.dtype)
    a = scale * torch.randn(
        args.batch, args.samples, args.tokens, args.c_a, device="cuda", dtype=dtype
    )
    s = scale * torch.randn(
        args.batch, 1, args.tokens, args.c_s, device="cuda", dtype=dtype
    )
    z = scale * torch.randn(
        args.batch, args.c_z, args.tokens, args.tokens, device="cuda", dtype=dtype
    )
    mask = _token_mask(args.batch, args.tokens, args.valid_fraction)
    return {
        "a": _flatten_samples(a, args.samples).contiguous(),
        "s": _flatten_samples(s, args.samples).contiguous(),
        "z": z.contiguous(),
        "token_mask": _flatten_mask(mask, args.samples),
    }


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
        compile_start = time.perf_counter()
        out, mean_ms, peak_mib = _cuda_time(fn, args.warmup, args.iters)
        elapsed_sec = time.perf_counter() - compile_start
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        return {"name": name, "ok": False, "error": repr(exc)}
    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "mean_ms": mean_ms,
        "peak_allocated_mib": peak_mib,
        "elapsed_sec_including_compile": elapsed_sec,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }
    if reference is not None:
        row["parity_vs_eager"] = _compare(out, reference)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=220)
    parser.add_argument("--valid-fraction", type=float, default=0.8)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_transformer_compile_probe requires CUDA")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = _dtype(args.dtype)
    model = DiffusionTransformer(
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
        n_blocks=args.blocks,
        n_heads=args.heads,
    ).cuda().eval()
    wrapper = FlatDiffusionTransformer(model, samples=args.samples).cuda().eval()
    inputs = _make_inputs(args)

    def eager() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            return wrapper(**inputs)

    eager_out, eager_ms, eager_peak = _cuda_time(eager, args.warmup, args.iters)
    rows: list[dict[str, Any]] = [
        {
            "name": "eager",
            "ok": True,
            "mean_ms": eager_ms,
            "peak_allocated_mib": eager_peak,
            "output_shape": list(eager_out.shape),
            "output_dtype": str(eager_out.dtype),
        }
    ]

    compiled_wrapper = torch.compile(
        wrapper,
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=False,
    )

    def compiled() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            return compiled_wrapper(**inputs)

    rows.append(_record("compiled", compiled, args, reference=eager_out))

    result = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "cases": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
