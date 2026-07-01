#!/usr/bin/env python3
"""Audit matmul precision across a real Protenix inference call.

This script is intentionally an audit tool, not a throughput benchmark.  Python
hooks add overhead, so the elapsed time reported here should only be used to
confirm that the run completed.  The useful output is the ranked list of matrix
operations, especially modules whose ``precision=torch.float32`` setting forces
autocast off internally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

import torch

from configs.configs_inference import inference_configs
from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.utils.seed import seed_everything
from runner.batch_inference import get_default_runner
from runner.inference import update_inference_configs
from scripts.perf.protenix_inference_benchmark import (
    default_model_params,
    optional_bool,
    optional_int,
    parse_input_indices,
    prediction_sample_count,
    prepare_data,
    str_bool,
)


def git_info(repo: Path) -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    return {
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": run_git("status", "--porcelain"),
    }


def dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def shape_of(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def product(values: list[int] | tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()
    result = tuple(shapes[0])
    for shape in shapes[1:]:
        result = torch.broadcast_shapes(result, tuple(shape))
    return tuple(int(dim) for dim in result)


def matmul_flops(left: torch.Tensor, right: torch.Tensor) -> int:
    """Return a simple multiply-add FLOP estimate for torch.matmul-like ops."""

    if left.ndim == 1 and right.ndim == 1:
        return 2 * int(left.shape[0])
    if left.ndim == 2 and right.ndim == 1:
        return 2 * int(left.shape[0]) * int(left.shape[1])
    if left.ndim == 1 and right.ndim == 2:
        return 2 * int(left.shape[0]) * int(right.shape[1])
    if left.ndim < 2 or right.ndim < 2:
        return 0

    batch = product(broadcast_shape(tuple(left.shape[:-2]), tuple(right.shape[:-2])))
    return 2 * batch * int(left.shape[-2]) * int(left.shape[-1]) * int(right.shape[-1])


def linear_flops(module: torch.nn.Module, output: torch.Tensor) -> int:
    out_features = int(getattr(module, "out_features", 0) or 0)
    in_features = int(getattr(module, "in_features", 0) or 0)
    if out_features <= 0 or in_features <= 0 or output.numel() == 0:
        return 0
    rows = output.numel() // out_features
    return 2 * int(rows) * in_features * out_features


def expand_einsum_labels(
    spec: str,
    ndim: int,
    ellipsis_labels: list[str],
) -> list[str] | None:
    if spec.count("...") > 1:
        return None
    named = spec.replace("...", "")
    if "..." not in spec:
        return list(named) if len(named) == ndim else None
    ellipsis_rank = ndim - len(named)
    if ellipsis_rank < 0 or ellipsis_rank > len(ellipsis_labels):
        return None
    labels = ellipsis_labels[-ellipsis_rank:] if ellipsis_rank else []
    before, after = spec.split("...")
    return list(before) + labels + list(after)


def einsum_flops(
    equation: str,
    operands: tuple[Any, ...],
    output: torch.Tensor,
) -> int:
    """Estimate FLOPs for common explicit einsums.

    For two-operand einsums without repeated labels this generic estimate is
    exactly the usual contraction volume times multiply-add.  For more complex
    equations it is still useful as a materiality screen, but kernel-level
    profiling remains the source of truth.
    """

    if "->" not in equation:
        return 0
    inputs_spec, output_spec = equation.replace(" ", "").split("->", 1)
    input_specs = inputs_spec.split(",")
    tensor_operands = tuple(op for op in operands if isinstance(op, torch.Tensor))
    if len(input_specs) != len(tensor_operands) or not tensor_operands:
        return 0

    max_ellipsis_rank = 0
    for spec, tensor in zip(input_specs, tensor_operands):
        if "..." in spec:
            max_ellipsis_rank = max(max_ellipsis_rank, tensor.ndim - len(spec.replace("...", "")))
    if "..." in output_spec:
        max_ellipsis_rank = max(
            max_ellipsis_rank, output.ndim - len(output_spec.replace("...", ""))
        )
    ellipsis_labels = [f"@{idx}" for idx in range(max_ellipsis_rank)]

    operand_labels = [
        expand_einsum_labels(spec, tensor.ndim, ellipsis_labels)
        for spec, tensor in zip(input_specs, tensor_operands)
    ]
    if any(labels is None for labels in operand_labels):
        return 0
    output_labels = expand_einsum_labels(output_spec, output.ndim, ellipsis_labels)
    if output_labels is None:
        return 0

    sizes: dict[str, int] = {}
    for labels, tensor in zip(operand_labels, tensor_operands):
        assert labels is not None
        for label, dim in zip(labels, tensor.shape):
            sizes[label] = max(sizes.get(label, 1), int(dim))
    for label, dim in zip(output_labels, output.shape):
        sizes[label] = max(sizes.get(label, 1), int(dim))

    labels_in_contraction = sorted(
        {label for labels in operand_labels for label in (labels or [])}
        | set(output_labels)
    )
    if not labels_in_contraction:
        return 0
    contraction_volume = product([sizes[label] for label in labels_in_contraction])
    return 2 * contraction_volume


def module_family(name: str) -> str:
    parts = name.split(".")
    if not parts or not parts[0]:
        return "unknown"
    return parts[0]


def user_location(repo: Path) -> str:
    """Find the first non-torch frame that called a traced tensor operation."""

    frame = sys._getframe(2)
    while frame is not None:
        filename = Path(frame.f_code.co_filename)
        filename_str = str(filename)
        if (
            "site-packages/torch/" not in filename_str
            and filename.name != Path(__file__).name
        ):
            try:
                display = filename.resolve().relative_to(repo)
            except Exception:
                display = filename
            return f"{display}:{frame.f_lineno}:{frame.f_code.co_name}"
        frame = frame.f_back
    return "unknown"


class MatmulAudit(AbstractContextManager["MatmulAudit"]):
    """Aggregate matmul-like operations while one forward pass runs."""

    def __init__(
        self,
        model: torch.nn.Module,
        repo: Path,
        trace_nonmodule: bool,
    ) -> None:
        self.model = model
        self.repo = repo
        self.trace_nonmodule = trace_nonmodule
        self.rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.family_rows: dict[tuple[str, str, bool], dict[str, Any]] = {}
        self.handles: list[Any] = []
        self.old_ops: dict[str, Callable[..., Any]] = {}

    def add(self, row: dict[str, Any]) -> None:
        key = (
            row.get("kind"),
            row.get("name"),
            row.get("source"),
            tuple(row.get("input_shape") or []),
            tuple(row.get("right_shape") or []),
            tuple(row.get("output_shape") or []),
            row.get("input_dtype"),
            row.get("right_dtype"),
            row.get("output_dtype"),
            row.get("compute_dtype"),
            row.get("forced_precision"),
        )
        if key not in self.rows:
            self.rows[key] = {
                **row,
                "count": 0,
                "estimated_flops": 0,
            }
        entry = self.rows[key]
        entry["count"] += 1
        entry["estimated_flops"] += int(row.get("estimated_flops", 0) or 0)

        family_key = (
            str(row.get("family") or "unknown"),
            str(row.get("compute_dtype") or "unknown"),
            bool(row.get("forced_precision")),
        )
        if family_key not in self.family_rows:
            self.family_rows[family_key] = {
                "family": family_key[0],
                "compute_dtype": family_key[1],
                "forced_precision": family_key[2],
                "count": 0,
                "estimated_flops": 0,
            }
        family = self.family_rows[family_key]
        family["count"] += 1
        family["estimated_flops"] += int(row.get("estimated_flops", 0) or 0)

    def __enter__(self) -> "MatmulAudit":
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.handles.append(
                    module.register_forward_hook(self._linear_hook(name))
                )
        if self.trace_nonmodule:
            self._patch_torch_ops()
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for name, old_op in self.old_ops.items():
            setattr(torch, name, old_op)
        self.old_ops = {}

    def _linear_hook(self, name: str) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            if not isinstance(output, torch.Tensor):
                return

            input_tensor = inputs[0]
            precision = getattr(module, "precision", None)
            forced_precision = precision is not None
            if forced_precision:
                compute_dtype = dtype_name(precision)
            elif output.dtype in (torch.bfloat16, torch.float16):
                compute_dtype = dtype_name(output.dtype)
            else:
                compute_dtype = dtype_name(input_tensor.dtype)

            self.add(
                {
                    "kind": "linear_module",
                    "name": name,
                    "class": type(module).__name__,
                    "family": module_family(name),
                    "source": None,
                    "input_shape": shape_of(input_tensor),
                    "right_shape": [int(getattr(module, "out_features", 0)), int(getattr(module, "in_features", 0))],
                    "output_shape": shape_of(output),
                    "input_dtype": dtype_name(input_tensor.dtype),
                    "right_dtype": dtype_name(getattr(module, "weight").dtype),
                    "output_dtype": dtype_name(output.dtype),
                    "compute_dtype": compute_dtype,
                    "forced_precision": forced_precision,
                    "module_precision": dtype_name(precision),
                    "estimated_flops": linear_flops(module, output),
                }
            )

        return hook

    def _patch_torch_ops(self) -> None:
        self.old_ops = {
            "matmul": torch.matmul,
            "bmm": torch.bmm,
            "mm": torch.mm,
            "einsum": torch.einsum,
        }

        def traced_matmul(left: torch.Tensor, right: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            output = self.old_ops["matmul"](left, right, *args, **kwargs)
            self._add_binary_op("torch.matmul", left, right, output, matmul_flops(left, right))
            return output

        def traced_bmm(left: torch.Tensor, right: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            output = self.old_ops["bmm"](left, right, *args, **kwargs)
            flops = 0
            if left.ndim == 3 and right.ndim == 3:
                flops = 2 * int(left.shape[0]) * int(left.shape[1]) * int(left.shape[2]) * int(right.shape[2])
            self._add_binary_op("torch.bmm", left, right, output, flops)
            return output

        def traced_mm(left: torch.Tensor, right: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            output = self.old_ops["mm"](left, right, *args, **kwargs)
            self._add_binary_op("torch.mm", left, right, output, matmul_flops(left, right))
            return output

        def traced_einsum(equation: Any, *operands: Any, **kwargs: Any) -> torch.Tensor:
            output = self.old_ops["einsum"](equation, *operands, **kwargs)
            tensors = tuple(op for op in operands if isinstance(op, torch.Tensor))
            first = tensors[0] if tensors else None
            second = tensors[1] if len(tensors) > 1 else None
            flops = (
                einsum_flops(equation, operands, output)
                if isinstance(equation, str) and isinstance(output, torch.Tensor)
                else 0
            )
            self.add(
                {
                    "kind": "torch.einsum",
                    "name": str(equation),
                    "class": None,
                    "family": "nonmodule",
                    "source": user_location(self.repo),
                    "input_shape": shape_of(first),
                    "right_shape": shape_of(second),
                    "output_shape": shape_of(output),
                    "input_dtype": dtype_name(first.dtype) if first is not None else None,
                    "right_dtype": dtype_name(second.dtype) if second is not None else None,
                    "output_dtype": dtype_name(output.dtype),
                    "compute_dtype": dtype_name(output.dtype),
                    "forced_precision": False,
                    "module_precision": None,
                    "estimated_flops": flops,
                }
            )
            return output

        torch.matmul = traced_matmul
        torch.bmm = traced_bmm
        torch.mm = traced_mm
        torch.einsum = traced_einsum

    def _add_binary_op(
        self,
        kind: str,
        left: torch.Tensor,
        right: torch.Tensor,
        output: torch.Tensor,
        flops: int,
    ) -> None:
        self.add(
            {
                "kind": kind,
                "name": kind,
                "class": None,
                "family": "nonmodule",
                "source": user_location(self.repo),
                "input_shape": shape_of(left),
                "right_shape": shape_of(right),
                "output_shape": shape_of(output),
                "input_dtype": dtype_name(left.dtype),
                "right_dtype": dtype_name(right.dtype),
                "output_dtype": dtype_name(output.dtype),
                "compute_dtype": dtype_name(output.dtype),
                "forced_precision": False,
                "module_precision": None,
                "estimated_flops": flops,
            }
        )

    def top_rows(self, limit: int) -> list[dict[str, Any]]:
        rows = sorted(
            self.rows.values(),
            key=lambda row: (row["estimated_flops"], row["count"]),
            reverse=True,
        )
        return rows[:limit]

    def fp32_rows(self, limit: int) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.rows.values()
            if row.get("compute_dtype") == "float32"
            or row.get("output_dtype") == "float32"
            or row.get("module_precision") == "float32"
        ]
        rows.sort(key=lambda row: (row["estimated_flops"], row["count"]), reverse=True)
        return rows[:limit]

    def forced_fp32_linear_rows(self, limit: int) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.rows.values()
            if row.get("kind") == "linear_module"
            and row.get("forced_precision")
            and row.get("module_precision") == "float32"
        ]
        rows.sort(key=lambda row: (row["estimated_flops"], row["count"]), reverse=True)
        return rows[:limit]

    def family_summary(self) -> list[dict[str, Any]]:
        rows = list(self.family_rows.values())
        rows.sort(key=lambda row: (row["estimated_flops"], row["count"]), reverse=True)
        return rows


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--input-indices", default="0")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=32)
    parser.add_argument("--use-default-params", type=str_bool, default=False)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use-msa", type=str_bool, default=True)
    parser.add_argument("--use-template", type=str_bool, default=False)
    parser.add_argument("--use-rna-msa", type=str_bool, default=False)
    parser.add_argument("--use-seeds-in-json", type=str_bool, default=False)
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--enable-cache", type=str_bool, default=True)
    parser.add_argument("--enable-fusion", type=str_bool, default=True)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument(
        "--skip-amp-confidence-head",
        type=optional_bool,
        default=None,
        help="auto/none keeps runner default; false lets confidence head run under AMP.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=optional_int, default=None)
    parser.add_argument("--sample-diffusion-chunk-size", type=optional_int, default=None)
    parser.add_argument("--deterministic", type=str_bool, default=False)
    parser.add_argument("--trace-nonmodule", type=str_bool, default=True)
    parser.add_argument("--row-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(args.dump_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json).resolve() if args.output_json else out_dir / "precision_audit.json"

    if args.use_default_params:
        args.n_cycle, args.n_step = default_model_params(args.model_name)

    inference_configs["dump_dir"] = str(out_dir / "predictions")
    inference_configs["input_json_path"] = args.input_json
    inference_configs["num_workers"] = args.num_workers

    load_start = time.perf_counter()
    runner = get_default_runner(
        seeds=[args.seed],
        n_cycle=args.n_cycle,
        n_step=args.n_step,
        n_sample=args.n_sample,
        dtype=args.dtype,
        model_name=args.model_name,
        use_msa=args.use_msa,
        trimul_kernel=args.triangle_multiplicative,
        triatt_kernel=args.triangle_attention,
        enable_cache=args.enable_cache,
        enable_fusion=args.enable_fusion,
        enable_tf32=args.enable_tf32,
        use_template=args.use_template,
        use_rna_msa=args.use_rna_msa,
        use_seeds_in_json=args.use_seeds_in_json,
    )
    load_sec = time.perf_counter() - load_start
    runner.configs.input_json_path = args.input_json
    runner.configs.num_workers = args.num_workers
    runner.configs.infer_setting.chunk_size = args.chunk_size
    runner.configs.infer_setting.sample_diffusion_chunk_size = (
        args.sample_diffusion_chunk_size
    )

    dataset = InferenceDataset(runner.configs)
    selected_indices = parse_input_indices(args.input_indices)
    if selected_indices is not None:
        n_available = len(dataset.inputs)
        if any(index < 0 or index >= n_available for index in selected_indices):
            raise ValueError(
                f"--input-indices {selected_indices} out of range for {n_available} inputs"
            )
        dataset.inputs = [dataset.inputs[index] for index in selected_indices]
    if not dataset.inputs:
        raise ValueError("no inputs selected")

    raw_name = dataset.inputs[0]["name"]

    def build_data(suffix: str) -> tuple[dict[str, Any], dict[str, float]]:
        sample_name = f"{raw_name}_precision_audit_seed{args.seed}_{suffix}"
        run_data, _atom_array, run_feature_times = prepare_data(dataset, 0, sample_name)
        return run_data, run_feature_times

    data, feature_times = build_data("shape")
    n_token = int(data["N_token"].item())
    n_atom = int(data["N_atom"].item())
    n_msa = int(data["N_msa"].item())
    runner.update_model_configs(update_inference_configs(runner.configs, n_token))
    if args.skip_amp_confidence_head is not None:
        runner.configs.skip_amp.confidence_head = args.skip_amp_confidence_head
        runner.model.configs.skip_amp.confidence_head = args.skip_amp_confidence_head

    for warmup_idx in range(args.warmup_runs):
        seed_everything(seed=args.seed + 90000001 + warmup_idx, deterministic=args.deterministic)
        warmup_data, _ = build_data(f"warmup{warmup_idx}")
        # Protenix deletes large input features during inference once caches are
        # built.  Re-featurizing each pass keeps warmup from changing the traced
        # input and avoids auditing an impossible post-mutation state.
        with torch.inference_mode():
            _ = runner.predict(warmup_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    data, feature_times = build_data("trace")
    seed_everything(seed=args.seed, deterministic=args.deterministic)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    forward_start = time.perf_counter()
    with torch.inference_mode(), MatmulAudit(
        runner.model,
        repo=repo,
        trace_nonmodule=args.trace_nonmodule,
    ) as audit:
        prediction = runner.predict(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    forward_sec = time.perf_counter() - forward_start

    coordinate = prediction.get("coordinate")
    finite_coordinate = (
        bool(torch.isfinite(coordinate).all().item())
        if isinstance(coordinate, torch.Tensor)
        else None
    )
    result = {
        "meta": {
            "repo": git_info(repo),
            "args": vars(args),
            "load_sec": load_sec,
            "forward_sec_with_hooks": forward_sec,
            "feature_breakdown_sec": feature_times,
            "n_token": n_token,
            "n_atom": n_atom,
            "n_msa": n_msa,
            "skip_amp_confidence_head": bool(
                runner.model.configs.skip_amp.confidence_head
            ),
            "prediction_samples": prediction_sample_count(prediction, args.n_sample),
            "coordinate_all_finite": finite_coordinate,
            "cuda_memory": cuda_memory(),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            "env": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith("PROTENIX_")
                or key
                in {
                    "CUDA_VISIBLE_DEVICES",
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NODELIST",
                    "PYTORCH_CUDA_ALLOC_CONF",
                }
            },
            "notes": [
                "Python hooks make forward_sec_with_hooks unsuitable for throughput claims.",
                "linear_module.compute_dtype is module.precision when precision is forced; otherwise it is inferred from the output/input dtype.",
                "estimated_flops are multiply-add estimates used for materiality ranking, not hardware counters.",
            ],
        },
        "family_summary": audit.family_summary(),
        "top_matmul_like_ops": audit.top_rows(args.row_limit),
        "top_fp32_compute_ops": audit.fp32_rows(args.row_limit),
        "top_forced_fp32_linear_modules": audit.forced_fp32_linear_rows(args.row_limit),
    }

    output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["meta"], indent=2, sort_keys=True))
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
