#!/usr/bin/env python3
"""Pin down where ragged pairformer padding first changes valid outputs.

Each stage compares a true short tensor, the same valid crop inside a larger
zero-padded tensor, and the same padded tensor with random invalid rows/cols.
``valid_region`` catches physical-length-dependent drift; ``invalid_region``
catches real padded-token leakage.  The key trace is triangle multiplication:
its projections are per-cell GEMMs, but its combine matmul reduces across the
token dimension, so mathematically zero padded lanes can still change the GPU
accumulation schedule.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext

import torch

Case = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Cases = tuple[Case, Case, Case]


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
    if dtype_name == "float32":
        with nullcontext():
            yield
        return
    with torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name)):
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
    raise ValueError(f"do not know how to crop {tuple(tensor.shape)}")


def record_stage(
    rows: list[dict[str, object]],
    name: str,
    short: torch.Tensor,
    padded: torch.Tensor,
    noisy: torch.Tensor,
    short_tokens: int,
) -> None:
    rows.append(
        {
            "stage": name,
            "shape_short": list(short.shape),
            "shape_padded": list(padded.shape),
            "valid_region": compare(valid_crop(padded, short_tokens), short),
            "invalid_region_sensitivity": compare(
                valid_crop(noisy, short_tokens),
                valid_crop(padded, short_tokens),
            ),
        }
    )


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
        # AF3 initializes several output projections to zero.  Randomizing only
        # otherwise-zero matrices makes this a diagnostic of real data movement
        # while keeping the normal model initialization untouched.
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 7919)
            for parameter in block.parameters():
                if parameter.ndim >= 2 and not torch.any(parameter):
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
    return block


def make_cases(args: argparse.Namespace) -> Cases:
    short = args.short_tokens
    full = args.full_tokens
    dtype = getattr(torch, args.input_dtype)
    device = torch.device("cuda")

    s_short = torch.randn(short, args.c_s, device=device, dtype=dtype)
    z_short = torch.randn(short, short, args.c_z, device=device, dtype=dtype)
    mask_short = torch.ones(short, short, device=device, dtype=dtype)

    s_padded = torch.zeros(full, args.c_s, device=device, dtype=dtype)
    z_padded = torch.zeros(full, full, args.c_z, device=device, dtype=dtype)
    mask_padded = torch.zeros(full, full, device=device, dtype=dtype)
    s_padded[:short] = s_short
    z_padded[:short, :short] = z_short
    mask_padded[:short, :short] = 1

    generator = torch.Generator(device=device).manual_seed(args.seed + 211)
    s_noisy = torch.randn(
        full, args.c_s, device=device, dtype=dtype, generator=generator
    )
    z_noisy = torch.randn(
        full, full, args.c_z, device=device, dtype=dtype, generator=generator
    )
    s_noisy[:short] = s_short
    z_noisy[:short, :short] = z_short

    return (
        (s_short, z_short, mask_short),
        (s_padded, z_padded, mask_padded),
        (s_noisy, z_noisy, mask_padded),
    )


def triangle_trace(
    module: torch.nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = mask.unsqueeze(-1)
    z_norm = module.layer_norm_in(z)

    a_gate = module.sigmoid(module.linear_a_g(z_norm))
    a_projection = module.linear_a_p(z_norm)
    a_masked = mask * a_gate * a_projection

    b_gate = module.sigmoid(module.linear_b_g(z_norm))
    b_projection = module.linear_b_p(z_norm)
    b_masked = mask * b_gate * b_projection

    # This is the first operation whose mathematical reduction dimension is
    # token length, not channel length.  Padded zeros do not leak, but they do
    # change the physical reduction size and therefore the GPU accumulation
    # schedule selected by matmul/CUEQ.
    combined = module._combine_projections(a_masked, b_masked)
    combined_norm = module.layer_norm_out(combined)
    output_projection = module.linear_z(combined_norm)
    output_gate = module.sigmoid(module.linear_g(z_norm))
    output = output_projection * output_gate

    return {
        "input": z,
        "layer_norm_in": z_norm,
        "a_projection": a_projection,
        "a_masked": a_masked,
        "b_projection": b_projection,
        "b_masked": b_masked,
        "combine_token_reduction": combined,
        "layer_norm_out": combined_norm,
        "output_projection": output_projection,
        "output_gate": output_gate,
        "output": output,
    }


def trace_direction(
    name: str,
    module: torch.nn.Module,
    cases: Cases,
    short_tokens: int,
) -> list[dict[str, object]]:
    short_trace = triangle_trace(module, cases[0][1].clone(), cases[0][2].clone())
    padded_trace = triangle_trace(module, cases[1][1].clone(), cases[1][2].clone())
    noisy_trace = triangle_trace(module, cases[2][1].clone(), cases[2][2].clone())

    rows: list[dict[str, object]] = []
    for stage, short_tensor in short_trace.items():
        record_stage(
            rows,
            f"{name}.{stage}",
            short_tensor,
            padded_trace[stage],
            noisy_trace[stage],
            short_tokens,
        )
    return rows


def screen_module(
    name: str,
    module: torch.nn.Module,
    kernel: str,
    cases: Cases,
    short_tokens: int,
) -> dict[str, object]:
    def run(case: Case) -> torch.Tensor:
        _s, z, mask = case
        return module(
            z.clone(),
            mask=mask.clone(),
            inplace_safe=False,
            _add_with_inplace=False,
            triangle_multiplicative=kernel,
        )

    short = run(cases[0])
    padded = run(cases[1])
    noisy = run(cases[2])
    rows: list[dict[str, object]] = []
    record_stage(rows, f"{name}.module_{kernel}", short, padded, noisy, short_tokens)
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-tokens", type=int, default=245)
    parser.add_argument("--full-tokens", type=int, default=384)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=False)
    dtypes = ["float32", "bfloat16"]
    parser.add_argument("--input-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--compute-dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-zero-weights", type=str_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.short_tokens >= args.full_tokens:
        raise ValueError("--short-tokens must be less than --full-tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("pairformer_padding_deep_dive requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = args.enable_tf32
    torch.manual_seed(args.seed)

    block = make_block(args)
    cases = make_cases(args)
    amp = cuda_autocast(args.compute_dtype)
    with torch.inference_mode(), amp:
        rows = trace_direction(
            "tri_mul_out",
            block.tri_mul_out,
            cases,
            args.short_tokens,
        )
        for kernel in ("torch", "cuequivariance"):
            rows.append(
                screen_module(
                    "tri_mul_out",
                    block.tri_mul_out,
                    kernel,
                    cases,
                    args.short_tokens,
                )
            )
        torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "args": vars(args),
                "device": torch.cuda.get_device_name(),
                "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
