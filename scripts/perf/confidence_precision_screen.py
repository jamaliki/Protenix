#!/usr/bin/env python3
"""Screen lower-precision modes for ConfidenceHead inference matmuls.

This holds the confidence inputs fixed and compares candidate precision modes
against the current production-like path.  It is deliberately smaller than a
full inference benchmark: the goal is to identify which FP32 matmuls are safe
to move onto tensor cores before paying for an end-to-end gate.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.sample_confidence import logits_to_score


MODE_CONFIGS = {
    # Current inference shape: confidence pairformer inherits outer BF16
    # autocast, but the final confidence logits are explicitly forced to FP32.
    "prod": {
        "autocast": "bfloat16",
        "allow_tf32": True,
        "matmul_precision": None,
        "bf16_logits": False,
    },
    # Same as prod, but asks PyTorch to use TF32/tensor-core algorithms for
    # any remaining float32 matmul that is allowed to trade mantissa bits.
    "prod-tf32-high": {
        "autocast": "bfloat16",
        "allow_tf32": True,
        "matmul_precision": "high",
        "bf16_logits": False,
    },
    "prod-tf32-medium": {
        "autocast": "bfloat16",
        "allow_tf32": True,
        "matmul_precision": "medium",
        "bf16_logits": False,
    },
    # Lets only the final confidence logit projections inherit BF16 autocast.
    # ConfidenceHead upcasts the returned logits back to FP32.
    "prod-bf16-logits": {
        "autocast": "bfloat16",
        "allow_tf32": True,
        "matmul_precision": None,
        "bf16_logits": True,
    },
    # Useful as a diagnostic reference, not a proposed production mode.
    "fp32": {
        "autocast": None,
        "allow_tf32": False,
        "matmul_precision": "highest",
        "bf16_logits": False,
    },
}


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in MODE_CONFIGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown precision mode(s): {unknown}")
    return modes


@contextmanager
def precision_context(mode: str):
    cfg = MODE_CONFIGS[mode]
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    old_bf16_logits = os.environ.get("PROTENIX_BF16_CONFIDENCE_LOGITS")

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg["allow_tf32"])
    if cfg["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(str(cfg["matmul_precision"]))
    os.environ["PROTENIX_BF16_CONFIDENCE_LOGITS"] = (
        "1" if cfg["bf16_logits"] else "0"
    )

    autocast_dtype = cfg["autocast"]
    if autocast_dtype == "bfloat16":
        amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    elif autocast_dtype == "float16":
        amp = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp = nullcontext()

    try:
        with amp:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)
        if old_bf16_logits is None:
            os.environ.pop("PROTENIX_BF16_CONFIDENCE_LOGITS", None)
        else:
            os.environ["PROTENIX_BF16_CONFIDENCE_LOGITS"] = old_bf16_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--atoms", type=int, default=2529)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=256)
    parser.add_argument("--c-s-inputs", type=int, default=449)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--max-atoms-per-token", type=int, default=20)
    parser.add_argument("--hidden-scale-up", type=str_bool, default=True)
    parser.add_argument(
        "--input-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help=(
            "Synthetic trunk/input embedding dtype. Keep the default FP32: "
            "AMP then chooses BF16 kernels where production inference allows it."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--load-checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--model-name", default="protenix-v2")
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("prod,prod-tf32-high,prod-bf16-logits"))
    parser.add_argument("--reference-mode", default="prod", choices=sorted(MODE_CONFIGS))
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--inplace-safe", type=str_bool, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--trace-mode", default="prod", choices=["none", *sorted(MODE_CONFIGS)])
    parser.add_argument("--trace-row-limit", type=int, default=40)
    parser.add_argument("--ncu-mode", default="none", choices=["none", *sorted(MODE_CONFIGS)])
    parser.add_argument("--ncu-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def make_feature_dict(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    rep_mask = torch.zeros(args.atoms, device=device, dtype=torch.bool)
    rep_mask[: args.tokens] = True
    atom_to_token_idx = torch.div(
        torch.arange(args.atoms, device=device) * args.tokens,
        args.atoms,
        rounding_mode="floor",
    ).clamp(max=args.tokens - 1)
    atom_to_tokatom_idx = torch.arange(args.atoms, device=device) % args.max_atoms_per_token
    return {
        "distogram_rep_atom_mask": rep_mask,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
    }


def checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.checkpoint is not None:
        return args.checkpoint
    candidate = args.load_checkpoint_dir / f"{args.model_name}.pt"
    return candidate if candidate.exists() else None


def load_confidence_weights(head: ConfidenceHead, args: argparse.Namespace) -> dict[str, Any]:
    path = checkpoint_path(args)
    if path is None:
        return {"loaded": False, "reason": "checkpoint not found"}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    prefix = "confidence_head."
    confidence_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    result = head.load_state_dict(confidence_state, strict=False)
    del checkpoint, state, confidence_state
    return {
        "loaded": True,
        "path": str(path),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def make_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, args.input_dtype)
    return {
        "s_inputs": torch.randn(args.tokens, args.c_s_inputs, device=device, dtype=dtype),
        "s_trunk": torch.randn(args.tokens, args.c_s, device=device, dtype=dtype),
        "z_trunk": torch.randn(args.tokens, args.tokens, args.c_z, device=device, dtype=dtype),
        "x_pred_coords": torch.randn(args.samples, args.atoms, 3, device=device, dtype=torch.float32),
    }


def run_head(
    head: ConfidenceHead,
    feature_dict: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad(), precision_context(mode):
        return head(
            input_feature_dict=feature_dict,
            s_inputs=tensors["s_inputs"],
            s_trunk=tensors["s_trunk"],
            z_trunk=tensors["z_trunk"],
            pair_mask=None,
            x_pred_coords=tensors["x_pred_coords"],
            triangle_multiplicative=args.triangle_multiplicative,
            triangle_attention=args.triangle_attention,
            inplace_safe=args.inplace_safe,
            chunk_size=None,
        )


def cuda_time(fn: Callable[[], tuple[torch.Tensor, ...]], warmup: int, iters: int) -> tuple[tuple[torch.Tensor, ...], float]:
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


def output_meta(out: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    return {
        "dtypes": [str(t.dtype) for t in out],
        "finite": all(bool(torch.isfinite(t).all().item()) for t in out),
        "shapes": [list(t.shape) for t in out],
    }


def tensor_diff(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float()
    diff = cand.float() - ref_f
    ref_rms = torch.sqrt(torch.mean(ref_f * ref_f)).item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    return {
        "max_abs": torch.max(torch.abs(diff)).item(),
        "mean_abs": torch.mean(torch.abs(diff)).item(),
        "rmse": rmse,
        "rel_rmse": rmse / (ref_rms + 1e-12),
    }


def score_diff(name: str, ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float] | None:
    bins = {
        "plddt": (0.0, 1.0, 50, 100.0),
        "pae": (0.0, 32.0, 64, 1.0),
        "pde": (0.0, 32.0, 64, 1.0),
    }.get(name)
    if bins is None:
        return None
    min_bin, max_bin, no_bins, scale = bins
    ref_score = logits_to_score(ref.float(), min_bin, max_bin, no_bins) * scale
    cand_score = logits_to_score(cand.float(), min_bin, max_bin, no_bins) * scale
    return tensor_diff(ref_score, cand_score)


def compare_outputs(ref: tuple[torch.Tensor, ...], cand: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    names = ("plddt", "pae", "pde", "resolved")
    rows = {}
    for name, ref_t, cand_t in zip(names, ref, cand):
        rows[name] = {"logit": tensor_diff(ref_t, cand_t)}
        scores = score_diff(name, ref_t, cand_t)
        if scores is not None:
            rows[name]["score"] = scores
    ref_rank = torch.argsort(logits_to_score(ref[0].float(), 0.0, 1.0, 50).mean(-1), descending=True)
    cand_rank = torch.argsort(logits_to_score(cand[0].float(), 0.0, 1.0, 50).mean(-1), descending=True)
    rows["sample_rank_by_mean_plddt_equal"] = bool(torch.equal(ref_rank, cand_rank))
    return rows


@contextmanager
def trace_matmuls(head: ConfidenceHead):
    records: list[dict[str, Any]] = []
    handles = []

    def linear_hook(name: str, module: torch.nn.Linear):
        def hook(_module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            m = output.numel() // max(1, module.out_features)
            records.append({
                "kind": "linear",
                "name": name,
                "input_dtype": str(inputs[0].dtype),
                "output_dtype": str(output.dtype),
                "input_shape": list(inputs[0].shape),
                "output_shape": list(output.shape),
                "flops": int(2 * m * module.in_features * module.out_features),
            })
        return hook

    for name, module in head.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name, module)))

    old_einsum = torch.einsum

    def traced_einsum(equation, *operands):
        output = old_einsum(equation, *operands)
        flops = 0
        if equation == "...nc,ncb->...nb" and len(operands) == 2:
            k = operands[0].shape[-1]
            n = operands[1].shape[-1]
            m = operands[0].numel() // max(1, k)
            flops = int(2 * m * n * k)
        records.append({
            "kind": "einsum",
            "name": equation,
            "input_dtype": ",".join(str(t.dtype) for t in operands if isinstance(t, torch.Tensor)),
            "output_dtype": str(output.dtype),
            "input_shape": [list(t.shape) for t in operands if isinstance(t, torch.Tensor)],
            "output_shape": list(output.shape),
            "flops": flops,
        })
        return output

    torch.einsum = traced_einsum
    try:
        yield records
    finally:
        torch.einsum = old_einsum
        for handle in handles:
            handle.remove()


def aggregate_trace(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        key = (
            row["kind"],
            row["name"],
            row["input_dtype"],
            row["output_dtype"],
            json.dumps(row["input_shape"]),
            json.dumps(row["output_shape"]),
        )
        item = totals.setdefault(key, {**row, "count": 0, "flops": 0})
        item["count"] += 1
        item["flops"] += row.get("flops", 0)
    rows = list(totals.values())
    rows.sort(key=lambda item: item["flops"], reverse=True)
    return rows[:limit]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("confidence_precision_screen requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    head = ConfidenceHead(
        n_blocks=args.n_blocks,
        c_s=args.c_s,
        c_z=args.c_z,
        c_s_inputs=args.c_s_inputs,
        max_atoms_per_token=args.max_atoms_per_token,
        hidden_scale_up=args.hidden_scale_up,
    ).to(device)
    head.eval()
    load_info = load_confidence_weights(head, args)
    feature_dict = make_feature_dict(args, device)
    tensors = make_inputs(args, device)

    def call(mode: str):
        return run_head(head, feature_dict, tensors, args, mode)

    if args.ncu_mode != "none":
        for _ in range(args.warmup):
            call(args.ncu_mode)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"confidence_precision:{args.ncu_mode}")
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.ncu_iters):
            out = call(args.ncu_mode)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
        print(
            json.dumps(
                {"args": serializable_args(args), "out": output_meta(out), "load": load_info},
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.reference_mode not in args.modes:
        args.modes.insert(0, args.reference_mode)

    rows = {}
    outputs = {}
    for mode in args.modes:
        torch.cuda.reset_peak_memory_stats()
        out, ms = cuda_time(lambda mode=mode: call(mode), args.warmup, args.iters)
        outputs[mode] = tuple(t.detach().cpu() for t in out)
        rows[mode] = {
            "ms": ms,
            "out": output_meta(out),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "torch": {
                "allow_tf32_after_restore": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision_after_restore": torch.get_float32_matmul_precision(),
            },
        }
        del out
        torch.cuda.empty_cache()

    ref = outputs[args.reference_mode]
    comparisons = {
        mode: compare_outputs(ref, out)
        for mode, out in outputs.items()
        if mode != args.reference_mode
    }

    trace = []
    if args.trace_mode != "none":
        with trace_matmuls(head) as records:
            call(args.trace_mode)
            torch.cuda.synchronize()
        trace = aggregate_trace(records, args.trace_row_limit)

    print(json.dumps({
        "args": serializable_args(args),
        "checkpoint": load_info,
        "device": torch.cuda.get_device_name(),
        "modes": MODE_CONFIGS,
        "rows": rows,
        "comparisons_vs_reference": comparisons,
        "trace": trace,
        "timestamp": time.time(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
