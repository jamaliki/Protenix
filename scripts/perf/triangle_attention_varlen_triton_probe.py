#!/usr/bin/env python3
"""Screen a length-aware Triton core for v2 triangle attention.

The v2 ragged-island screen showed useful long-bucket headroom, but Python
islands are not a production strategy: they fragment launches and lose on short
buckets.  This probe asks the next kernel-boundary question:

    If q/k/v and triangle bias are already produced, can a single descriptor-
    scheduled Triton attention core beat the current padded CUEQ/cuDNN core by
    skipping padded rows, query tiles, and key ranges?

This is deliberately benchmark-only.  It does not replace CUEQ in the model and
it does not include q/k/v/bias/gate/output projections in the timed core.  A win
here would justify a production-quality segmented triangle-attention boundary.
A loss means the naive custom mainloop is not good enough even before the
surrounding projection/epilogue integration costs are charged.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None

from protenix.model.triangular.layers import cuequivariance_triangular_attn
from protenix.model.triangular.triangular import TriangleAttention
from protenix.model.utils import permute_final_dims


DTYPE = torch.bfloat16
TensorFn = Callable[[], torch.Tensor]


if triton is not None and tl is not None:

    @triton.jit
    def _varlen_triangle_attention_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        lengths_ptr,
        desc_b_ptr,
        desc_n_ptr,
        desc_q_ptr,
        n_max: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """Forward softmax attention for one valid descriptor tile.

        q/k/v/out use ``[B, N, S, H, D]`` physical layout.  Each descriptor is
        one valid ``(batch, row, query_tile_start)``.  That descriptor list is
        what removes the rectangular-grid waste: no programs are launched for
        padded pair rows or all-padded query tiles.
        """

        desc = tl.program_id(0)
        head = tl.program_id(1)

        batch = tl.load(desc_b_ptr + desc).to(tl.int64)
        pair_row = tl.load(desc_n_ptr + desc).to(tl.int64)
        query_start = tl.load(desc_q_ptr + desc).to(tl.int64)
        valid_len = tl.load(lengths_ptr + batch).to(tl.int64)

        q_offsets = query_start + tl.arange(0, block_m)
        d_offsets = tl.arange(0, head_dim)
        q_mask = q_offsets < valid_len
        d_mask = d_offsets < head_dim

        # q/k/v are contiguous [B, N, S, H, D].  Keep the pointer arithmetic
        # explicit so the kernel can use runtime lengths without constructing
        # many differently shaped tensors.
        q_base = (((batch * n_max + pair_row) * n_max) * head_count + head) * head_dim
        q = tl.load(
            q_ptr
            + q_base
            + q_offsets[:, None] * head_count * head_dim
            + d_offsets[None, :],
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, head_dim), tl.float32)

        log2e: tl.constexpr = 1.4426950408889634
        key_start = tl.full((), 0, tl.int64)
        while key_start < valid_len:
            k_offsets = key_start + tl.arange(0, block_n)
            k_mask = k_offsets < valid_len

            kv_base = (
                (((batch * n_max + pair_row) * n_max) * head_count + head) * head_dim
            )
            k = tl.load(
                k_ptr
                + kv_base
                + k_offsets[:, None] * head_count * head_dim
                + d_offsets[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + kv_base
                + k_offsets[:, None] * head_count * head_dim
                + d_offsets[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            bias_base = (((batch * head_count + head) * n_max) * n_max)
            bias = tl.load(
                bias_ptr
                + bias_base
                + q_offsets[:, None] * n_max
                + k_offsets[None, :],
                mask=q_mask[:, None] & k_mask[None, :],
                other=-1.0e6,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k)) * scale + bias
            scores = scores * log2e
            scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -1.0e6)

            m_ij = tl.maximum(tl.max(scores, axis=1), m_i)
            p = tl.math.exp2(scores - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)

            m_i = m_ij
            l_i = l_ij
            key_start += block_n

        acc = acc / l_i[:, None]
        out = acc.to(q.dtype)

        tl.store(
            out_ptr
            + q_base
            + q_offsets[:, None] * head_count * head_dim
            + d_offsets[None, :],
            out,
            mask=q_mask[:, None] & d_mask[None, :],
        )


def parse_ints(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_tiles(text: str) -> list[tuple[int, int, int]]:
    configs = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.split("x")
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError(
                "tiles must look like '64x32' or '64x32x4'"
            )
        block_m, block_n = int(parts[0]), int(parts[1])
        warps = int(parts[2]) if len(parts) == 3 else 4
        configs.append((block_m, block_n, warps))
    if not configs:
        raise argparse.ArgumentTypeError("at least one tile config is required")
    return configs


def make_module(args: argparse.Namespace) -> TriangleAttention:
    module = TriangleAttention(
        c_in=args.c_z,
        c_hidden=args.c_hidden_pair_att,
        no_heads=args.no_heads_pair,
        starting=True,
    ).cuda()
    module.eval()

    # Synthetic hotspots should exercise the same GEMM and epilogue topology as
    # a real model block.  Randomize only deliberately all-zero matrices so the
    # shapes are realistic without changing code paths.
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 8191)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2 and not torch.any(parameter):
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def make_inputs(
    lengths: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(lengths)
    z = torch.zeros(
        len(lengths),
        n_max,
        n_max,
        args.c_z,
        device="cuda",
        dtype=DTYPE,
    )
    mask = torch.zeros(len(lengths), n_max, n_max, device="cuda", dtype=DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for batch_index, length in enumerate(lengths):
        z[batch_index, :length, :length] = torch.randn(
            length,
            length,
            args.c_z,
            device="cuda",
            dtype=DTYPE,
            generator=generator,
        )
        mask[batch_index, :length, :length] = 1
    return z, mask


def build_descriptors(
    lengths: list[int],
    block_m: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    desc_b = []
    desc_n = []
    desc_q = []
    for batch_index, length in enumerate(lengths):
        for pair_row in range(length):
            for query_start in range(0, length, block_m):
                desc_b.append(batch_index)
                desc_n.append(pair_row)
                desc_q.append(query_start)
    return (
        torch.tensor(desc_b, dtype=torch.int32, device=device),
        torch.tensor(desc_n, dtype=torch.int32, device=device),
        torch.tensor(desc_q, dtype=torch.int32, device=device),
    )


def prepare_attention_tensors(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x = module.layer_norm(z)
    q_hf, k_hf, v_hf = module.mha._prep_qkv(
        x,
        x,
        apply_scale=False,
        cueq_heads_first=True,
    )
    bias = permute_final_dims(module.linear(x), (2, 0, 1)).unsqueeze(-4)
    return {
        "x_norm": x,
        "q_hf": q_hf.contiguous(),
        "k_hf": k_hf.contiguous(),
        "v_hf": v_hf.contiguous(),
        "q_seq": q_hf.transpose(-2, -3).contiguous(),
        "k_seq": k_hf.transpose(-2, -3).contiguous(),
        "v_seq": v_hf.transpose(-2, -3).contiguous(),
        "bias": bias.float().contiguous(),
        "mask_bool": (mask == 1)[..., :, None, None, :],
    }


def prepare_varlen_tensors(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Prepare q/k/v in the layout consumed by the varlen Triton kernel.

    The core-only screen uses CUEQ's heads-first q/k/v so CUEQ and Triton see
    exactly the same producer outputs.  A production varlen path would not do
    that: it needs ``[B, N, S, H, D]`` q/k/v.  Producing that layout directly
    avoids a CUEQ-layout Triton copy followed by another transpose/contiguous
    copy before the custom core.
    """

    x = module.layer_norm(z)
    q, k, v = module.mha._prep_qkv(
        x,
        x,
        apply_scale=False,
        cueq_heads_first=False,
    )
    bias = permute_final_dims(module.linear(x), (2, 0, 1)).unsqueeze(-4)
    return {
        "x_norm": x,
        "q_seq": q.transpose(-2, -3).contiguous(),
        "k_seq": k.transpose(-2, -3).contiguous(),
        "v_seq": v.transpose(-2, -3).contiguous(),
        "bias": bias.float().contiguous(),
    }


def cueq_core(tensors: dict[str, torch.Tensor], scale: float) -> torch.Tensor:
    out = cuequivariance_triangular_attn(
        tensors["q_hf"],
        tensors["k_hf"],
        tensors["v_hf"],
        tensors["bias"],
        tensors["mask_bool"],
        scale,
    )
    if isinstance(out, tuple):
        out = out[0]
    # Return [B, N, S, H, D], the same physical layout as the Triton candidate.
    return out.transpose(-2, -3).contiguous()


def triton_varlen_core(
    tensors: dict[str, torch.Tensor],
    lengths: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
    scale: float,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for the varlen attention probe")

    q = tensors["q_seq"]
    out = torch.empty_like(q)
    desc_b, desc_n, desc_q = descriptors
    grid = (desc_b.numel(), q.shape[-2])
    _varlen_triangle_attention_fwd[grid](
        q,
        tensors["k_seq"],
        tensors["v_seq"],
        tensors["bias"],
        out,
        lengths,
        desc_b,
        desc_n,
        desc_q,
        q.shape[1],
        q.shape[-2],
        q.shape[-1],
        scale,
        block_m,
        block_n,
        num_warps=warps,
        num_stages=3,
    )
    return out


def cueq_module_forward(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return module(z, mask=mask, triangle_attention="cuequivariance")


def triton_varlen_module_forward(
    module: TriangleAttention,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
    scale: float,
) -> torch.Tensor:
    # This intentionally pays the surrounding producer/consumer costs.  The
    # core screen above answers whether the attention mainloop has promise; this
    # boundary asks whether that promise survives the real module plumbing.
    tensors = prepare_varlen_tensors(module, z, mask)
    heads_last = triton_varlen_core(
        tensors,
        lengths,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=scale,
    )
    return module.mha._wrap_up(heads_last, tensors["x_norm"])


def flatten_valid_core(
    tensor: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    chunks = []
    for row, length in zip(tensor, lengths, strict=True):
        chunks.append(row[:length, :length].reshape(-1, row.shape[-2], row.shape[-1]))
    return torch.cat(chunks, dim=0)


def flatten_valid_pair(
    tensor: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    chunks = []
    for row, length in zip(tensor, lengths, strict=True):
        chunks.append(row[:length, :length].reshape(-1, row.shape[-1]))
    return torch.cat(chunks, dim=0)


def compare_valid(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lengths: list[int],
) -> dict[str, float | int]:
    ref = flatten_valid_core(reference, lengths)
    cand = flatten_valid_core(candidate, lengths)
    diff = (cand.float() - ref.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def compare_valid_pair(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lengths: list[int],
) -> dict[str, float | int]:
    ref = flatten_valid_pair(reference, lengths)
    cand = flatten_valid_pair(candidate, lengths)
    diff = (cand.float() - ref.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def cuda_time(fn: TensorFn, warmup: int, iters: int) -> tuple[torch.Tensor, dict[str, float]]:
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


def work_stats(lengths: list[int], block_m: int) -> dict[str, float | int]:
    n_max = max(lengths)
    dense_programs = len(lengths) * n_max * math.ceil(n_max / block_m)
    valid_programs = sum(length * math.ceil(length / block_m) for length in lengths)
    dense_cubic = len(lengths) * n_max * n_max * n_max
    valid_cubic = sum(length * length * length for length in lengths)
    return {
        "dense_programs_per_head": dense_programs,
        "valid_programs_per_head": valid_programs,
        "program_fraction": valid_programs / dense_programs,
        "dense_cubic": dense_cubic,
        "valid_cubic": valid_cubic,
        "cubic_fraction": valid_cubic / dense_cubic,
        "ideal_cubic_speedup": dense_cubic / valid_cubic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints("136,136,148,148,160,160,172,172,184,184,196,196,208,208,220,220"),
    )
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-hidden-pair-att", type=int, default=32)
    parser.add_argument("--no-heads-pair", type=int, default=8)
    parser.add_argument(
        "--tiles",
        type=parse_tiles,
        default=parse_tiles("32x32x4,32x64x4,64x32x4,64x64x4"),
        help="Comma-separated block_m x block_n x warps configs.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--skip-full-boundary",
        action="store_true",
        help="Only time the precomputed q/k/v/bias attention core.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("triangle_attention_varlen_triton_probe requires CUDA")
    if triton is None or tl is None:
        raise RuntimeError("triangle_attention_varlen_triton_probe requires Triton")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    module = make_module(args)
    z, mask = make_inputs(args.lengths, args)
    scale = 1.0 / math.sqrt(args.c_hidden_pair_att)
    lengths_tensor = torch.tensor(args.lengths, dtype=torch.int32, device=z.device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=DTYPE):
        tensors = prepare_attention_tensors(module, z, mask)
        reference, cueq_timing = cuda_time(
            lambda: cueq_core(tensors, scale),
            args.warmup,
            args.iters,
        )
        full_reference = None
        full_cueq_timing = None
        if not args.skip_full_boundary:
            full_reference, full_cueq_timing = cuda_time(
                lambda: cueq_module_forward(module, z, mask),
                args.warmup,
                args.iters,
            )

        candidates = []
        for block_m, block_n, warps in args.tiles:
            descriptors = build_descriptors(
                args.lengths,
                block_m,
                device=z.device,
            )
            candidate, timing = cuda_time(
                lambda descriptors=descriptors, block_m=block_m, block_n=block_n, warps=warps: triton_varlen_core(
                    tensors,
                    lengths_tensor,
                    descriptors,
                    block_m=block_m,
                    block_n=block_n,
                    warps=warps,
                    scale=scale,
                ),
                args.warmup,
                args.iters,
            )
            row = {
                "tile": f"{block_m}x{block_n}x{warps}",
                "timing": timing,
                "speedup_vs_cueq_core": cueq_timing["ms"] / timing["ms"],
                "parity_vs_cueq_valid": compare_valid(
                    reference,
                    candidate,
                    args.lengths,
                ),
                "descriptors": {
                    "count": int(descriptors[0].numel()),
                    **work_stats(args.lengths, block_m),
                },
            }
            if full_reference is not None and full_cueq_timing is not None:
                full_candidate, full_timing = cuda_time(
                    lambda descriptors=descriptors, block_m=block_m, block_n=block_n, warps=warps: triton_varlen_module_forward(
                        module,
                        z,
                        mask,
                        lengths_tensor,
                        descriptors,
                        block_m=block_m,
                        block_n=block_n,
                        warps=warps,
                        scale=scale,
                    ),
                    args.warmup,
                    args.iters,
                )
                row["full_boundary"] = {
                    "timing": full_timing,
                    "speedup_vs_cueq_module": full_cueq_timing["ms"]
                    / full_timing["ms"],
                    "parity_vs_cueq_valid": compare_valid_pair(
                        full_reference,
                        full_candidate,
                        args.lengths,
                    ),
                }
            candidates.append(row)

    best = max(candidates, key=lambda row: row["speedup_vs_cueq_core"])
    full_candidates = [row for row in candidates if "full_boundary" in row]
    best_full = (
        max(
            full_candidates,
            key=lambda row: row["full_boundary"]["speedup_vs_cueq_module"],
        )
        if full_candidates
        else None
    )
    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "fixed_dtype": str(DTYPE),
                "cueq_core": cueq_timing,
                "cueq_module": full_cueq_timing,
                "candidates": candidates,
                "best": best,
                "best_full_boundary": best_full,
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
