#!/usr/bin/env python3
"""Probe full Protenix inference with a leading batch of same-shape inputs."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from configs.configs_inference import inference_configs
from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.utils.seed import seed_everything
from runner.batch_inference import get_default_runner
from runner.inference import update_inference_configs


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def optional_int(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"none", "null", "-1"}:
        return None
    return int(value)


def clone_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: clone_tree(value) for key, value in obj.items()}
    return copy.deepcopy(obj)


def stack_tree(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack([item.clone() for item in items], dim=0)
    if isinstance(first, dict):
        return {key: stack_tree([item[key] for item in items]) for key in first}
    return copy.deepcopy(first)


def finite_summary(prediction: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("coordinate", "summary_confidence"):
        value = prediction.get(key)
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "all_finite": bool(torch.isfinite(value).all().item()),
            }
        elif isinstance(value, list):
            summary[key] = {"list_len": len(value)}
    return summary


def timing_summary(runner: Any) -> dict[str, float]:
    log_dict = getattr(runner, "last_log_dict", {})
    if not isinstance(log_dict, dict):
        return {}
    timings = log_dict.get("time", {})
    if not isinstance(timings, dict):
        return {}
    return {
        key: float(value)
        for key, value in timings.items()
        if isinstance(value, (int, float))
    }


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def timed_predict(
    runner: Any, data: dict[str, Any]
) -> tuple[dict[str, Any], float, dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = runner.predict(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return prediction, time.perf_counter() - start, timing_summary(runner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default="examples/example.json")
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use-msa", type=str_bool, default=True)
    parser.add_argument("--use-template", type=str_bool, default=False)
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--enable-cache", type=str_bool, default=True)
    parser.add_argument("--enable-fusion", type=str_bool, default=True)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--chunk-size", type=optional_int, default=None)
    parser.add_argument("--sample-diffusion-chunk-size", type=optional_int, default=None)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dump-dir", default="/tmp/protenix_same_shape_batch_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.dump_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_configs["dump_dir"] = str(out_dir / "predictions")
    inference_configs["input_json_path"] = args.input_json
    inference_configs["num_workers"] = 0

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
    )
    runner.configs.infer_setting.chunk_size = args.chunk_size
    runner.configs.infer_setting.sample_diffusion_chunk_size = (
        args.sample_diffusion_chunk_size
    )
    dataset = InferenceDataset(runner.configs)
    data, _, feature_times = dataset.process_one(dataset.inputs[args.input_index])
    n_token = int(data["N_token"].item())
    runner.update_model_configs(update_inference_configs(runner.configs, n_token))
    base_feature = data["input_feature_dict"]

    def make_single() -> dict[str, Any]:
        return {"input_feature_dict": clone_tree(base_feature)}

    def make_batch() -> dict[str, Any]:
        return {
            "input_feature_dict": stack_tree(
                [base_feature for _ in range(args.batch_size)]
            )
        }

    for index in range(args.warmup):
        seed_everything(args.seed + 9000 + index, deterministic=False)
        timed_predict(runner, make_single())

    torch.cuda.reset_peak_memory_stats()
    loop_predictions = []
    loop_start = time.perf_counter()
    loop_secs = []
    loop_model_timings = []
    for index in range(args.batch_size):
        seed_everything(args.seed + index, deterministic=False)
        prediction, seconds, model_timing = timed_predict(runner, make_single())
        loop_predictions.append(finite_summary(prediction))
        loop_secs.append(seconds)
        loop_model_timings.append(model_timing)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    loop_wall = time.perf_counter() - loop_start
    loop_memory = cuda_memory()

    torch.cuda.reset_peak_memory_stats()
    seed_everything(args.seed, deterministic=False)
    batch_prediction, batch_sec, batch_model_timing = timed_predict(
        runner, make_batch()
    )
    batch_memory = cuda_memory()

    row = {
        "args": vars(args),
        "input_name": dataset.inputs[args.input_index]["name"],
        "shape": {
            "n_token": n_token,
            "n_atom": int(data["N_atom"].item()),
            "n_msa": int(data["N_msa"].item()),
        },
        "feature_times": feature_times,
        "loop": {
            "wall_sec": loop_wall,
            "item_secs": loop_secs,
            "model_timings": loop_model_timings,
            "sequences_per_sec": args.batch_size / loop_wall,
            "predictions": loop_predictions,
            **loop_memory,
        },
        "batched": {
            "wall_sec": batch_sec,
            "model_timing": batch_model_timing,
            "sequences_per_sec": args.batch_size / batch_sec,
            "prediction": finite_summary(batch_prediction),
            **batch_memory,
        },
        "speedup": loop_wall / batch_sec,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
