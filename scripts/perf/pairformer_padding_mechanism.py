#!/usr/bin/env python3
"""Isolate the mechanism behind unsafe mixed-length pairformer padding.

The tempting throughput optimization is to pad nearby token lengths into one
larger batch and rely on ``pair_mask``/``token_mask`` to hide the padded rows.
That is only safe if the valid crop of a padded run is numerically equivalent
to a true short run.

This probe separates two very different failure modes:

* ``invalid_region_sensitivity`` changes only the padded values while keeping
  the physical shape fixed.  Nonzero error here means a real mask leak.
* ``valid_region`` compares a true short tensor with the same values embedded
  in a larger masked tensor.  Nonzero error here, with zero invalid-region
  sensitivity, means the masks work but the GPU kernel is not invariant to the
  physical reduction length.  That is expected for floating-point reductions:
  summing 245 terms is not necessarily bitwise identical to summing 384 terms
  where the last 139 terms are mathematically zero.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator

import torch


TensorOp = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a.float() - b.float()).abs()
    finite = diff[torch.isfinite(diff)]
    return {
        "max_abs_error": float(finite.max().item()) if finite.numel() else float("nan"),
        "mean_abs_error": float(finite.mean().item()) if finite.numel() else float("nan"),
        "nan_count": int(torch.isnan(diff).sum().item()),
    }


def valid_crop(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor[:tokens]
    if tensor.ndim == 3 and tensor.shape[-3] == tensor.shape[-2]:
        return tensor[:tokens, :tokens]
    raise ValueError(f"do not know how to crop tensor with shape {tuple(tensor.shape)}")


def make_block(args: argparse.Namespace) -> torch.nn.Module:
    from protenix.model.modules.pairformer import PairformerBlock

    block = PairformerBlock(
        n_heads=args.n_heads,
        c_z=args.c_z,
        c_s=args.c_s,
        dropout=0.0,
        hidden_scale_up=args.hidden_scale_up,
    ).cuda()
    block.eval()
    if args.randomize_zero_weights:
        # Several AF3-style output projections are deliberately zero-initialized.
        # Randomizing only those otherwise-zero matrices makes the diagnostic
        # observe real data movement without perturbing the normal model path.
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 7919)
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_inputs(
    args: argparse.Namespace,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    short = args.short_tokens
    full = args.full_tokens
    dtype = getattr(torch, args.input_dtype)
    device = torch.device("cuda")

    s_short = torch.randn(short, args.c_s, device=device, dtype=dtype)
    z_short = torch.randn(short, short, args.c_z, device=device, dtype=dtype)
    mask_short = torch.ones(short, short, device=device, dtype=dtype)

    s_full = torch.zeros(full, args.c_s, device=device, dtype=dtype)
    z_full = torch.zeros(full, full, args.c_z, device=device, dtype=dtype)
    mask_full = torch.zeros(full, full, device=device, dtype=dtype)
    s_full[:short] = s_short
    z_full[:short, :short] = z_short
    mask_full[:short, :short] = 1

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

    return (
        (s_short, z_short, mask_short),
        (s_full, z_full, mask_full),
        (s_noisy, z_noisy, mask_full),
    )


def screen_operation(
    name: str,
    op: TensorOp,
    short_case: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    full_case: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    noisy_case: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    short_tokens: int,
) -> dict[str, object]:
    short_out = op(*(tensor.clone() for tensor in short_case))
    full_out = op(*(tensor.clone() for tensor in full_case))
    noisy_out = op(*(tensor.clone() for tensor in noisy_case))
    return {
        "operation": name,
        "output_shape_short": list(short_out.shape),
        "output_shape_full": list(full_out.shape),
        "valid_region": compare(valid_crop(full_out, short_tokens), short_out),
        "invalid_region_sensitivity": compare(
            valid_crop(noisy_out, short_tokens),
            valid_crop(full_out, short_tokens),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-tokens", type=int, default=245)
    parser.add_argument("--full-tokens", type=int, default=384)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--input-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.short_tokens >= args.full_tokens:
        raise ValueError("--short-tokens must be less than --full-tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_padding_mechanism requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)
    block = make_block(args)
    short_case, full_case, noisy_case = make_inputs(args)

    def tri_mul_out(s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del s
        return block.tri_mul_out(
            z,
            mask=mask,
            inplace_safe=False,
            _add_with_inplace=False,
            triangle_multiplicative=args.triangle_multiplicative,
        )

    def tri_att_start(s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del s
        return block.tri_att_start(
            z,
            mask=mask,
            triangle_attention=args.triangle_attention,
            inplace_safe=False,
        )

    def pair_transition(s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del s, mask
        return block.pair_transition(z)

    def token_attention(s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_mask = torch.diagonal(mask, dim1=-2, dim2=-1)
        return block.attention_pair_bias(a=s, s=None, z=z, token_mask=token_mask)

    operations: list[tuple[str, TensorOp]] = [
        ("triangle_multiplication_outgoing", tri_mul_out),
        ("triangle_attention_starting_node", tri_att_start),
        ("pair_transition_per_cell", pair_transition),
        ("token_attention_pair_bias", token_attention),
    ]

    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        results = [
            screen_operation(name, op, short_case, full_case, noisy_case, args.short_tokens)
            for name, op in operations
        ]
        torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "args": vars(args),
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
