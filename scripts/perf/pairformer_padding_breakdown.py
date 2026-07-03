#!/usr/bin/env python3
"""Diagnose why padded pairformer tokens change valid outputs.

Same-shape batching is now correct for independent sequences, but production
queues often contain nearby rather than identical lengths.  Padding is only
safe if the valid region of a padded run matches a standalone short run.  This
diagnostic replays one PairformerBlock stage by stage and compares the valid
slice after each stage.

Two comparisons matter:

* ``*_valid_region`` compares a short physical tensor with the same values
  embedded in a larger zero-padded tensor.  A nonzero value here means the
  operation is not invariant to physical token length.
* ``*_invalid_region_sensitivity`` compares two padded tensors with identical
  valid values but different invalid/padded values.  A nonzero value here would
  be a real mask leak.

This distinction is important for GPU work: padding can be numerically unsafe
even when the masks are correct, because changing the physical problem size can
select different matmul/attention reduction schedules.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch

from protenix.model.modules.pairformer import PairformerBlock


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
def cuda_autocast(dtype_name: str) -> Iterator[None]:
    if dtype_name == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif dtype_name == "float16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with nullcontext():
            yield


def make_block(args: argparse.Namespace) -> PairformerBlock:
    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()
    if args.randomize_zero_weights:
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 97)
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs_error": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs_error": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def record(
    rows: list[dict[str, object]],
    name: str,
    short_s: torch.Tensor,
    short_z: torch.Tensor,
    zero_s: torch.Tensor,
    zero_z: torch.Tensor,
    noisy_s: torch.Tensor,
    noisy_z: torch.Tensor,
    short_tokens: int,
) -> None:
    rows.append(
        {
            "stage": name,
            "s_valid_region": compare(zero_s[:short_tokens], short_s),
            "z_valid_region": compare(
                zero_z[:short_tokens, :short_tokens],
                short_z,
            ),
            "s_invalid_region_sensitivity": compare(
                noisy_s[:short_tokens],
                zero_s[:short_tokens],
            ),
            "z_invalid_region_sensitivity": compare(
                noisy_z[:short_tokens, :short_tokens],
                zero_z[:short_tokens, :short_tokens],
            ),
        }
    )


def run_stages(
    block: PairformerBlock,
    s: torch.Tensor,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def snapshot(name: str) -> None:
        # Several pairformer paths run with ``inplace_safe=True`` and may reuse
        # the same storage across stages.  Store real snapshots, not tensor
        # aliases, otherwise a later in-place update can make an earlier row
        # look divergent and point us at the wrong masking boundary.
        rows.append((name, s.detach().clone(), z.detach().clone()))

    snapshot("input")

    z = block.tri_mul_out(
        z,
        mask=pair_mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative=args.triangle_multiplicative,
    )
    snapshot("tri_mul_out")

    z = block.tri_mul_in(
        z,
        mask=pair_mask,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative=args.triangle_multiplicative,
    )
    snapshot("tri_mul_in")

    z = z + block.tri_att_start(
        z,
        mask=pair_mask,
        triangle_attention=args.triangle_attention,
        inplace_safe=True,
    )
    snapshot("tri_att_start")

    z = z.transpose(-2, -3).contiguous()
    pair_mask = pair_mask.transpose(-1, -2)
    snapshot("transpose_after_start")

    z = z + block.tri_att_end(
        z,
        mask=pair_mask,
        triangle_attention=args.triangle_attention,
        inplace_safe=True,
    )
    snapshot("tri_att_end")

    z = z.transpose(-2, -3).contiguous()
    pair_mask = pair_mask.transpose(-1, -2)
    snapshot("transpose_after_end")

    z = z + block.pair_transition(z)
    snapshot("pair_transition")

    token_mask = torch.diagonal(pair_mask, dim1=-2, dim2=-1)
    s = s + block.attention_pair_bias(a=s, s=None, z=z, token_mask=token_mask)
    snapshot("token_attention")

    s = s + block.single_transition(s)
    snapshot("single_transition")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-tokens", type=int, default=245)
    parser.add_argument("--full-tokens", type=int, default=600)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--input-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.short_tokens >= args.full_tokens:
        raise ValueError("--short-tokens must be less than --full-tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_padding_breakdown requires CUDA")

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.input_dtype)
    device = torch.device("cuda")
    block = make_block(args)

    short = args.short_tokens
    full = args.full_tokens
    s_short = torch.randn(short, args.c_s, device=device, dtype=dtype)
    z_short = torch.randn(short, short, args.c_z, device=device, dtype=dtype)
    mask_short = torch.ones(short, short, device=device, dtype=dtype)
    s_full = torch.zeros(full, args.c_s, device=device, dtype=dtype)
    z_full = torch.zeros(full, full, args.c_z, device=device, dtype=dtype)
    mask_full = torch.zeros(full, full, device=device, dtype=dtype)
    s_full[:short] = s_short
    z_full[:short, :short] = z_short
    mask_full[:short, :short] = 1

    # Same valid values, same mask, but deliberately dirty padded rows/columns.
    # If these values affect the valid crop, the problem is a mask leak.  If
    # only the zero-padded tensor differs from the short tensor, the problem is
    # physical-length-dependent floating-point scheduling.
    noisy_generator = torch.Generator(device=device).manual_seed(args.seed + 211)
    s_noisy = torch.randn(
        full,
        args.c_s,
        device=device,
        dtype=dtype,
        generator=noisy_generator,
    )
    z_noisy = torch.randn(
        full,
        full,
        args.c_z,
        device=device,
        dtype=dtype,
        generator=noisy_generator,
    )
    s_noisy[:short] = s_short
    z_noisy[:short, :short] = z_short

    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        short_rows = run_stages(block, s_short, z_short, mask_short, args)
        full_rows = run_stages(block, s_full, z_full, mask_full, args)
        noisy_rows = run_stages(block, s_noisy, z_noisy, mask_full, args)
        torch.cuda.synchronize()

    comparisons: list[dict[str, object]] = []
    for (
        (short_name, short_s, short_z),
        (full_name, full_s, full_z),
        (noisy_name, noisy_s, noisy_z),
    ) in zip(
        short_rows,
        full_rows,
        noisy_rows,
    ):
        assert short_name == full_name
        assert short_name == noisy_name
        record(
            comparisons,
            short_name,
            short_s,
            short_z,
            full_s,
            full_z,
            noisy_s,
            noisy_z,
            short,
        )

    print(
        json.dumps(
            {
                "args": vars(args),
                "comparisons": comparisons,
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
