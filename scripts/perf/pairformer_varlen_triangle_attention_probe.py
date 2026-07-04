#!/usr/bin/env python3
"""Screen varlen triangle attention inside a full v2 PairformerBlock.

The standalone triangle-attention probe showed a tempting result: a descriptor
scheduled Triton attention core skips padded token rows and is much faster than
the padded CUEQ/cuDNN core, but the full triangle-attention module only moved a
little once LayerNorm, q/k/v/bias production, gate, and output projection were
charged.

This script answers the promotion question one level higher: if we replace the
starting-node triangle attention inside a real Protenix-v2 PairformerBlock, does
the whole block move enough to justify production wiring?  It also times a
start+ending variant, but that ending path deliberately uses a physical
transpose and is therefore a lower-quality prototype than production's current
contiguous ending-node CUEQ path.  Treat start-only as the fair decision gate;
use start+end only to estimate how much more work a proper ending-varlen
producer would need to recover.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import statistics
from collections.abc import Callable
from contextlib import nullcontext
from typing import Optional

import torch

from protenix.model.modules.pairformer import PairformerBlock

from triangle_attention_varlen_triton_probe import (
    build_descriptors,
    parse_ints,
    parse_tiles,
    triton_varlen_module_forward,
)


TensorPair = tuple[Optional[torch.Tensor], torch.Tensor]
TensorFn = Callable[[], TensorPair]


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def autocast_context(dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name))


def make_block(args: argparse.Namespace) -> PairformerBlock:
    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()

    # Several Protenix projections are intentionally zero-initialized.  That is
    # correct for model starts, but a synthetic perf screen would hide real GEMM
    # and epilogue traffic.  Randomize only all-zero matrices.
    if args.randomize_zero_weights:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 65537)
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    batch = len(lengths)
    n_max = max(lengths)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    s = None
    if args.c_s > 0:
        s = torch.zeros(batch, n_max, args.c_s, device="cuda", dtype=dtype)
    z = torch.zeros(batch, n_max, n_max, args.c_z, device="cuda", dtype=dtype)
    mask = torch.zeros(batch, n_max, n_max, device="cuda", dtype=dtype)

    for index, length in enumerate(lengths):
        if s is not None:
            s[index, :length] = torch.randn(
                length,
                args.c_s,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
        z[index, :length, :length] = torch.randn(
            length,
            length,
            args.c_z,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        mask[index, :length, :length] = 1
    return s, z, mask


def clone_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.clone()


def run_baseline(
    block: PairformerBlock,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
) -> TensorPair:
    return block(
        clone_optional(s),
        z.clone(),
        pair_mask=mask,
        triangle_multiplicative="cuequivariance",
        triangle_attention="cuequivariance",
        inplace_safe=True,
        chunk_size=None,
    )


def finish_block(
    block: PairformerBlock,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
) -> TensorPair:
    z += block.pair_transition(z)
    if block.c_s > 0:
        if s is None:
            raise RuntimeError("block has c_s > 0 but s is None")
        token_mask = torch.diagonal(mask, dim1=-2, dim2=-1)
        s = s + block.attention_pair_bias(a=s, s=None, z=z, token_mask=token_mask)
        s = s + block.single_transition(s)
    return s, z


def run_start_varlen(
    block: PairformerBlock,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths_tensor: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
) -> TensorPair:
    s = clone_optional(s)
    z = z.clone()
    z = block.tri_mul_out(
        z,
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )
    z = block.tri_mul_in(
        z,
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )
    z += triton_varlen_module_forward(
        block.tri_att_start,
        z,
        mask,
        lengths_tensor,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=1.0 / (block.tri_att_start.c_hidden**0.5),
        fused_epilogue=True,
    )
    # Keep the current optimized ending-node path.  This isolates whether the
    # known starting-node varlen boundary is valuable in the real block.
    z += block.tri_att_end._cueq_ending_contiguous_forward(z, mask)
    return finish_block(block, s, z, mask)


def run_start_end_varlen_transpose(
    block: PairformerBlock,
    s: Optional[torch.Tensor],
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths_tensor: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
) -> TensorPair:
    s = clone_optional(s)
    z = z.clone()
    z = block.tri_mul_out(
        z,
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )
    z = block.tri_mul_in(
        z,
        mask=mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="cuequivariance",
    )
    z += triton_varlen_module_forward(
        block.tri_att_start,
        z,
        mask,
        lengths_tensor,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=1.0 / (block.tri_att_start.c_hidden**0.5),
        fused_epilogue=True,
    )
    z_end = z.transpose(-2, -3).contiguous()
    mask_end = mask.transpose(-1, -2)
    z_end += triton_varlen_module_forward(
        block.tri_att_end,
        z_end,
        mask_end,
        lengths_tensor,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=1.0 / (block.tri_att_end.c_hidden**0.5),
        fused_epilogue=True,
    )
    z = z_end.transpose(-2, -3).contiguous()
    return finish_block(block, s, z, mask)


def cuda_time(fn: TensorFn, warmup: int, iters: int) -> tuple[TensorPair, dict[str, float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return out, {
        "ms": statistics.median(samples),
        "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def flatten_valid_tokens(tensor: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    return torch.cat([row[:length] for row, length in zip(tensor, lengths, strict=True)])


def flatten_valid_pairs(tensor: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    return torch.cat(
        [
            row[:length, :length].reshape(-1, row.shape[-1])
            for row, length in zip(tensor, lengths, strict=True)
        ]
    )


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    diff = (candidate.float() - reference.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def compare_outputs(
    reference: TensorPair,
    candidate: TensorPair,
    lengths: list[int],
) -> dict[str, dict[str, float | int]]:
    result = {
        "z": compare(
            flatten_valid_pairs(reference[1], lengths),
            flatten_valid_pairs(candidate[1], lengths),
        )
    }
    if reference[0] is not None and candidate[0] is not None:
        result["s"] = compare(
            flatten_valid_tokens(reference[0], lengths),
            flatten_valid_tokens(candidate[0], lengths),
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints(
            "136,136,148,148,160,160,172,172,184,184,196,196,208,208,220,220"
        ),
    )
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument(
        "--input-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--tiles",
        type=parse_tiles,
        default=parse_tiles("32x32x4,32x64x4,64x32x4,64x64x4"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_varlen_triangle_attention_probe requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    block = make_block(args)
    s, z, mask = make_inputs(args.lengths, args)
    lengths_tensor = torch.tensor(args.lengths, dtype=torch.int32, device=z.device)

    with torch.inference_mode(), autocast_context(args.input_dtype):
        baseline_out, baseline_timing = cuda_time(
            lambda: run_baseline(block, s, z, mask),
            args.warmup,
            args.iters,
        )

        candidates = []
        for block_m, block_n, warps in args.tiles:
            descriptors = build_descriptors(args.lengths, block_m, device=z.device)
            for name, fn in (
                ("start_varlen", run_start_varlen),
                ("start_end_varlen_transpose", run_start_end_varlen_transpose),
            ):
                candidate_out, timing = cuda_time(
                    lambda fn=fn,
                    block_m=block_m,
                    block_n=block_n,
                    warps=warps,
                    descriptors=descriptors: fn(
                        block,
                        s,
                        z,
                        mask,
                        lengths_tensor,
                        descriptors,
                        block_m=block_m,
                        block_n=block_n,
                        warps=warps,
                    ),
                    args.warmup,
                    args.iters,
                )
                candidates.append(
                    {
                        "variant": name,
                        "tile": f"{block_m}x{block_n}x{warps}",
                        "timing": timing,
                        "speedup_vs_baseline": baseline_timing["ms"] / timing["ms"],
                        "parity_vs_baseline_valid": compare_outputs(
                            baseline_out,
                            candidate_out,
                            args.lengths,
                        ),
                        "descriptor_count": int(descriptors[0].numel()),
                    }
                )

    best_by_variant = {}
    for variant in sorted({row["variant"] for row in candidates}):
        rows = [row for row in candidates if row["variant"] == variant]
        best_by_variant[variant] = max(rows, key=lambda row: row["speedup_vs_baseline"])

    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "baseline": baseline_timing,
                "candidates": candidates,
                "best_by_variant": best_by_variant,
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
