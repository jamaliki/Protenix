#!/usr/bin/env python3
"""PairformerBlock screen for a fairer start+end varlen-attention path.

The previous start+end prototype transposed the whole pair tensor for ending
attention.  Production CUEQ avoids that by keeping producers on contiguous
``[B, I, J, C]`` and only swapping the attention layouts.  This benchmark-only
screen gives the varlen path the same chance before any production wiring.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None

from protenix.model.utils import permute_final_dims

from pairformer_varlen_triangle_attention_probe import (
    autocast_context,
    compare_outputs,
    cuda_time,
    finish_block,
    make_block,
    make_inputs,
    parse_ints,
    parse_tiles,
    run_baseline,
    str_bool,
)
from triangle_attention_varlen_triton_probe import (
    build_descriptors,
    triton_varlen_core,
    triton_varlen_module_forward,
)


if triton is not None and tl is not None:

    @triton.jit
    def _sigmoid_mul_heads_last_swapped_gate_fwd(
        gate_ptr,
        heads_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        i_count: tl.constexpr,
        j_count: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Gate logical [B,J,I,H,D] heads with contiguous [B,I,J,H*D] gates."""
        offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(
            tl.int64
        )
        mask = offsets < total_elements

        dim = offsets % head_dim
        tmp = offsets // head_dim
        head = tmp % head_count
        tmp = tmp // head_count
        i = tmp % i_count
        tmp = tmp // i_count
        j = tmp % j_count
        batch = tmp // j_count

        gate_offsets = (((batch * i_count + i) * j_count + j) * head_count + head)
        gate_offsets = gate_offsets * head_dim + dim
        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        heads = tl.load(heads_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 / (1.0 + tl.exp(-gate))) * heads
        tl.store(out_ptr + offsets, out, mask=mask)


def _ending_contiguous_tensors(module, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Prepare ending-node tensors while reading the original contiguous layout."""
    x_norm = module.layer_norm(z)
    q, k, v = module.mha._prep_qkv(
        x_norm,
        x_norm,
        apply_scale=False,
        cueq_heads_first=True,
        cueq_swap_pair_axes=True,
    )
    return {
        "x_norm": x_norm,
        "q_seq": q.transpose(-2, -3).contiguous(),
        "k_seq": k.transpose(-2, -3).contiguous(),
        "v_seq": v.transpose(-2, -3).contiguous(),
        "bias": permute_final_dims(module.linear(x_norm), (2, 1, 0))
        .unsqueeze(-4)
        .float()
        .contiguous(),
    }


def _ending_heads_last_wrap_up(module, heads_last: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    if module.mha.linear_g is None:
        return module.mha.linear_o(heads_last.flatten(-2)).transpose(-2, -3)
    gate = module.mha.linear_g(x_norm).contiguous()
    batch, j_count, i_count, head_count, head_dim = heads_last.shape
    flattened = torch.empty(
        (batch, j_count, i_count, head_count * head_dim),
        device=heads_last.device,
        dtype=heads_last.dtype,
    )
    block_size = 1024
    grid = (triton.cdiv(heads_last.numel(), block_size),)
    _sigmoid_mul_heads_last_swapped_gate_fwd[grid](
        gate,
        heads_last.contiguous(),
        flattened,
        heads_last.numel(),
        i_count,
        j_count,
        head_count,
        head_dim,
        block_size=block_size,
        num_warps=4,
    )
    return module.mha.linear_o(flattened).transpose(-2, -3)


def ending_contiguous_varlen_forward(
    module,
    z: torch.Tensor,
    lengths: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
) -> torch.Tensor:
    tensors = _ending_contiguous_tensors(module, z)
    heads_last = triton_varlen_core(
        tensors,
        lengths,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=1.0 / (module.c_hidden**0.5),
    )
    return _ending_heads_last_wrap_up(module, heads_last, tensors["x_norm"])


def run_start_end_varlen_contiguous(
    block,
    s,
    z: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    descriptors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    block_m: int,
    block_n: int,
    warps: int,
):
    s = None if s is None else s.clone()
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
        lengths,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        scale=1.0 / (block.tri_att_start.c_hidden**0.5),
        fused_epilogue=True,
    )
    z += ending_contiguous_varlen_forward(
        block.tri_att_end,
        z,
        lengths,
        descriptors,
        block_m=block_m,
        block_n=block_n,
        warps=warps,
    )
    return finish_block(block, s, z, mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=parse_ints,
        default=parse_ints("136,136,160,160,172,172,184,184,196,196,208,208,216,216,220,220"),
    )
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument("--input-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--tiles", type=parse_tiles, default=parse_tiles("32x32x4,32x64x4,64x32x4,64x64x4"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if triton is None or tl is None:
        raise RuntimeError("Triton is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    block = make_block(args)
    s, z, mask = make_inputs(args.lengths, args)
    lengths = torch.tensor(args.lengths, dtype=torch.int32, device=z.device)

    with torch.inference_mode(), autocast_context(args.input_dtype):
        baseline_out, baseline_timing = cuda_time(
            lambda: run_baseline(block, s, z, mask),
            args.warmup,
            args.iters,
        )
        candidates = []
        for block_m, block_n, warps in args.tiles:
            descriptors = build_descriptors(args.lengths, block_m, device=z.device)
            out, timing = cuda_time(
                lambda block_m=block_m, block_n=block_n, warps=warps, descriptors=descriptors: run_start_end_varlen_contiguous(
                    block,
                    s,
                    z,
                    mask,
                    lengths,
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
                    "variant": "start_end_varlen_contiguous",
                    "tile": f"{block_m}x{block_n}x{warps}",
                    "timing": timing,
                    "speedup_vs_baseline": baseline_timing["ms"] / timing["ms"],
                    "parity_vs_baseline_valid": compare_outputs(
                        baseline_out,
                        out,
                        args.lengths,
                    ),
                    "descriptor_count": int(descriptors[0].numel()),
                }
            )

    best = max(candidates, key=lambda row: row["speedup_vs_baseline"])
    print(
        json.dumps(
            {
                "args": {**vars(args), "lengths": args.lengths},
                "baseline": baseline_timing,
                "candidates": candidates,
                "best": best,
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
