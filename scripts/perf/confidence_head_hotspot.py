#!/usr/bin/env python3
"""Benchmark the confidence head on representative Protenix inference shapes."""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import time
from typing import Callable

import torch

from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.utils import broadcast_token_to_atom, one_hot


def cuda_time(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], float]:
    with torch.no_grad():
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


def cuda_profile_capture(
    name: str,
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    iters: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    with torch.no_grad():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(name)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            out = fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
    return out


def profile_cuda_ops(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    row_limit: int,
) -> list[dict[str, float | int | str]]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            fn()
        torch.cuda.synchronize()
    rows = []
    for event in profiler.key_averages():
        cuda_us = float(getattr(event, "cuda_time_total", 0.0) or getattr(event, "device_time_total", 0.0) or 0.0)
        if cuda_us > 0:
            rows.append({"key": event.key, "count": event.count, "cuda_time_us": cuda_us})
    rows.sort(key=lambda row: row["cuda_time_us"], reverse=True)
    return rows[:row_limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--atoms", type=int, default=2529)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--c-s-inputs", type=int, default=449)
    parser.add_argument("--max-atoms-per-token", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--profile-target",
        choices=["none", "distance", "pairformer", "logits", "full"],
        default="none",
    )
    parser.add_argument("--profile-row-limit", type=int, default=30)
    parser.add_argument(
        "--ncu-target",
        choices=["none", "distance", "pairformer", "logits", "full"],
        default="none",
    )
    parser.add_argument("--ncu-iters", type=int, default=1)
    parser.add_argument(
        "--inplace-safe",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use the real confidence clone+inplace path before pairformer/full targets.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def make_feature_dict(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    rep_mask = torch.zeros(args.atoms, device=device, dtype=torch.bool)
    rep_mask[: args.tokens] = True
    atom_to_token_idx = torch.arange(args.atoms, device=device) % args.tokens
    atom_to_tokatom_idx = (
        torch.arange(args.atoms, device=device) % args.max_atoms_per_token
    )
    return {
        "distogram_rep_atom_mask": rep_mask,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
    }


def finite_summary(out: torch.Tensor | tuple[torch.Tensor, ...]) -> dict[str, object]:
    tensors = out if isinstance(out, tuple) else (out,)
    return {
        "finite": all(bool(torch.isfinite(t).all().item()) for t in tensors),
        "shapes": [list(t.shape) for t in tensors],
        "dtypes": [str(t.dtype) for t in tensors],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("confidence_head_hotspot requires CUDA")
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    head = ConfidenceHead(
        c_s=args.c_s,
        c_z=args.c_z,
        c_s_inputs=args.c_s_inputs,
        max_atoms_per_token=args.max_atoms_per_token,
    ).cuda()
    head.eval()

    feature_dict = make_feature_dict(args, device)
    s_inputs = torch.randn(args.tokens, args.c_s_inputs, device=device, dtype=dtype)
    s_trunk = torch.randn(args.tokens, args.c_s, device=device, dtype=dtype)
    z_trunk = torch.randn(args.tokens, args.tokens, args.c_z, device=device, dtype=dtype)
    x_pred_coords = torch.randn(args.samples, args.atoms, 3, device=device, dtype=dtype)
    rep_coords = x_pred_coords[:, feature_dict["distogram_rep_atom_mask"], :]

    with torch.no_grad():
        s_norm = head.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))
        z_init = (
            head.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + head.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_base = z_init + z_trunk
        del z_init

    def run_distance() -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False), torch.no_grad():
            coords = rep_coords[0].to(torch.float32)
            distance = torch.cdist(coords, coords)
            z_dist = head.linear_no_bias_d(
                one_hot(
                    x=distance,
                    lower_bins=head.lower_bins,
                    upper_bins=head.upper_bins,
                )
            )
            return z_dist + head.linear_no_bias_d_wo_onehot(distance.unsqueeze(-1))

    z_distance = run_distance()
    z_pair_input = z_base + z_distance

    def run_pairformer() -> tuple[torch.Tensor, torch.Tensor]:
        return head.pairformer_stack(
            s_norm.clone() if args.inplace_safe else s_norm,
            z_pair_input.clone() if args.inplace_safe else z_pair_input,
            pair_mask=None,
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=bool(args.inplace_safe),
            chunk_size=None,
        )

    s_single, z_pair = run_pairformer()
    z_pair = z_pair.to(torch.float32)
    s_single = s_single.to(torch.float32)

    def run_logits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False), torch.no_grad():
            pae = head.linear_no_bias_pae(head.pae_ln(z_pair))
            pde = head.linear_no_bias_pde(head.pde_ln(z_pair + z_pair.transpose(-2, -3)))
            a = broadcast_token_to_atom(
                x_token=s_single,
                atom_to_token_idx=feature_dict["atom_to_token_idx"],
            )
            plddt = torch.einsum(
                "...nc,ncb->...nb",
                head.plddt_ln(a),
                head.plddt_weight[feature_dict["atom_to_tokatom_idx"]],
            )
            resolved = torch.einsum(
                "...nc,ncb->...nb",
                head.resolved_ln(a),
                head.resolved_weight[feature_dict["atom_to_tokatom_idx"]],
            )
            return plddt, pae, pde, resolved

    def run_full() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return head.memory_efficient_forward(
            input_feature_dict=feature_dict,
            s_trunk=s_norm.clone() if args.inplace_safe else s_norm,
            z_pair=z_base.clone() if args.inplace_safe else z_base,
            pair_mask=None,
            x_pred_rep_coords=rep_coords[0],
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=bool(args.inplace_safe),
            chunk_size=None,
        )

    targets = {
        "distance": run_distance,
        "pairformer": run_pairformer,
        "logits": run_logits,
        "full": run_full,
    }
    if args.ncu_target != "none":
        out = cuda_profile_capture(
            args.ncu_target,
            targets[args.ncu_target],
            args.warmup,
            args.ncu_iters,
        )
        print(
            json.dumps(
                {
                    "args": vars(args),
                    "device": torch.cuda.get_device_name(),
                    "target": args.ncu_target,
                    "out": finite_summary(out),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "timestamp": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    rows = {}
    for name, fn in targets.items():
        torch.cuda.reset_peak_memory_stats()
        out, ms = cuda_time(fn, args.warmup, args.iters)
        rows[name] = {
            "ms": ms,
            "out": finite_summary(out),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    profile = (
        profile_cuda_ops(
            targets[args.profile_target], args.warmup, args.profile_row_limit
        )
        if args.profile_target != "none"
        else []
    )

    print(
        json.dumps(
            {
                "args": vars(args),
                "device": torch.cuda.get_device_name(),
                "profile": profile,
                "rows": rows,
                "timestamp": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
