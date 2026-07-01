#!/usr/bin/env python3
"""Benchmark Protenix inference throughput for design-style sampling workloads."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from configs.configs_inference import inference_configs
from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.utils.distributed import DIST_WRAPPER
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


def parse_seeds(seed_arg: str, count: int | None) -> list[int]:
    if count is not None:
        base = int(seed_arg.split(",")[0]) if seed_arg else 101
        return [base + i for i in range(count)]
    return [int(seed.strip()) for seed in seed_arg.split(",") if seed.strip()]


def parse_input_indices(indices_arg: str) -> list[int] | None:
    if not indices_arg.strip():
        return None
    return [int(index.strip()) for index in indices_arg.split(",") if index.strip()]


def default_model_params(model_name: str) -> tuple[int, int]:
    if model_name in {
        "protenix_base_default_v0.5.0",
        "protenix_base_constraint_v0.5.0",
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        "protenix-v2",
    }:
        return 10, 200
    if model_name in {
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_tiny_default_v0.5.0",
    }:
        return 4, 5
    raise ValueError(f"unknown Protenix model for default params: {model_name}")


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


def prediction_sample_count(prediction: dict[str, Any], fallback: int) -> int:
    summary = prediction.get("summary_confidence")
    if isinstance(summary, list):
        return len(summary)
    coordinate = prediction.get("coordinate")
    if isinstance(coordinate, torch.Tensor) and coordinate.ndim >= 2:
        return int(coordinate.shape[-3])
    return fallback


def make_work_items(
    n_inputs: int,
    seeds: list[int],
    repeat: int,
    min_items_per_rank: int,
    world_size: int,
) -> list[tuple[int, int, int]]:
    base_items = [
        (repeat_idx, input_idx, seed)
        for repeat_idx in range(repeat)
        for input_idx in range(n_inputs)
        for seed in seeds
    ]
    if not base_items:
        raise ValueError("no benchmark work items were generated")

    items = list(base_items)
    target = max(1, world_size) * max(1, min_items_per_rank)
    extra_repeat = repeat
    while len(items) < target:
        for _, input_idx, seed in base_items:
            items.append((extra_repeat, input_idx, seed + 1000003 * extra_repeat))
            if len(items) >= target:
                break
        extra_repeat += 1
    return items


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def prepare_data(
    dataset: InferenceDataset,
    input_idx: int,
    sample_name: str,
) -> tuple[dict[str, Any], Any, dict[str, float]]:
    data, atom_array, feature_times = dataset.process_one(dataset.inputs[input_idx])
    data["sample_name"] = sample_name
    data["sample_index"] = input_idx
    return data, atom_array, feature_times


def run_one(
    *,
    runner: Any,
    dataset: InferenceDataset,
    input_idx: int,
    seed: int,
    repeat_idx: int,
    args: argparse.Namespace,
    phase: str,
    dump: bool,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    rank = DIST_WRAPPER.rank
    raw_name = dataset.inputs[input_idx]["name"]
    sample_name = f"{raw_name}_r{rank}_rep{repeat_idx}_seed{seed}_{phase}"

    row: dict[str, Any] = {
        "phase": phase,
        "rank": rank,
        "local_rank": DIST_WRAPPER.local_rank,
        "input_index": input_idx,
        "input_name": raw_name,
        "seed": seed,
        "repeat": repeat_idx,
        "sample_name": sample_name,
        "ok": False,
        "wall_start": time.time(),
    }

    feature_start = time.perf_counter()
    data, atom_array, feature_times = prepare_data(dataset, input_idx, sample_name)
    row["feature_total_sec"] = time.perf_counter() - feature_start
    row["feature_breakdown_sec"] = feature_times
    row["n_token"] = int(data["N_token"].item())
    row["n_atom"] = int(data["N_atom"].item())
    row["n_msa"] = int(data["N_msa"].item())

    runner.update_model_configs(update_inference_configs(runner.configs, row["n_token"]))
    seed_everything(seed=seed, deterministic=args.deterministic)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    forward_start = time.perf_counter()
    if profile_path is None:
        prediction = runner.predict(data)
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            prediction = runner.predict(data)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(profile_path))
        table_path = profile_path.with_suffix(".top_cuda.txt")
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        table_path.write_text(
            profiler.key_averages().table(sort_by=sort_key, row_limit=40)
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    row["forward_sec"] = time.perf_counter() - forward_start
    row["model_log"] = getattr(runner, "last_log_dict", {})
    row.update(cuda_memory())

    n_prediction_samples = prediction_sample_count(prediction, args.n_sample)
    row["prediction_samples"] = n_prediction_samples
    row["forward_samples_per_sec"] = n_prediction_samples / row["forward_sec"]
    coordinate = prediction.get("coordinate")
    if isinstance(coordinate, torch.Tensor):
        row["coordinate_all_finite"] = bool(torch.isfinite(coordinate).all().item())

    dump_start = time.perf_counter()
    if dump:
        runner.dumper.dump(
            dataset_name="",
            pdb_id=sample_name,
            seed=seed,
            pred_dict=prediction,
            atom_array=atom_array,
            entity_poly_type={
                key: value
                for key, value in data["entity_poly_type"].items()
                if value != "non-polymer"
            },
        )
    row["dump_sec"] = time.perf_counter() - dump_start
    row["total_sec"] = time.time() - row["wall_start"]
    row["wall_end"] = time.time()
    row["end_to_end_samples_per_sec"] = n_prediction_samples / row["total_sec"]
    row["ok"] = True

    del prediction
    if args.empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def aggregate_metrics(out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("metrics_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    timed = [row for row in rows if row.get("phase") == "timed" and row.get("ok")]
    if not timed:
        return {"timed_items": 0, "rows": len(rows)}

    wall_start = min(row["wall_start"] for row in timed)
    wall_end = max(row["wall_end"] for row in timed)
    total_wall_sec = wall_end - wall_start
    total_samples = sum(row["prediction_samples"] for row in timed)
    total_forward_sec = sum(row["forward_sec"] for row in timed)
    total_item_sec = sum(row["total_sec"] for row in timed)

    return {
        "timed_items": len(timed),
        "ranks": sorted({row["rank"] for row in timed}),
        "total_samples": total_samples,
        "total_wall_sec": total_wall_sec,
        "aggregate_samples_per_sec": total_samples / total_wall_sec,
        "sum_forward_sec": total_forward_sec,
        "sum_item_sec": total_item_sec,
        "mean_forward_samples_per_sec": sum(
            row["forward_samples_per_sec"] for row in timed
        )
        / len(timed),
        "mean_end_to_end_samples_per_sec": sum(
            row["end_to_end_samples_per_sec"] for row in timed
        )
        / len(timed),
        "max_peak_allocated_mib": max(row["peak_allocated_mib"] for row in timed),
        "max_peak_reserved_mib": max(row["peak_reserved_mib"] for row in timed),
        "mean_forward_sec": total_forward_sec / len(timed),
        "mean_feature_sec": sum(row["feature_total_sec"] for row in timed)
        / len(timed),
        "mean_dump_sec": sum(row["dump_sec"] for row in timed) / len(timed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--input-indices",
        default="",
        help="Comma-separated input indices from --input-json to benchmark; default is all.",
    )
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-items-per-rank", type=int, default=1)
    parser.add_argument("--warmup-items", type=int, default=0)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--n-sample", type=int, default=5)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=optional_int, default=None)
    parser.add_argument("--sample-diffusion-chunk-size", type=optional_int, default=5)
    parser.add_argument("--deterministic", type=str_bool, default=False)
    parser.add_argument("--dump", type=str_bool, default=True)
    parser.add_argument("--empty-cache", type=str_bool, default=True)
    parser.add_argument("--torch-profile", type=str_bool, default=False)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    out_dir = Path(args.dump_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_rank{DIST_WRAPPER.rank}.jsonl"
    metrics_path.unlink(missing_ok=True)

    if args.use_default_params:
        args.n_cycle, args.n_step = default_model_params(args.model_name)

    seeds = parse_seeds(args.seeds, args.num_seeds)
    repo = Path(__file__).resolve().parents[2]

    inference_configs["dump_dir"] = str(out_dir / "predictions")
    inference_configs["input_json_path"] = args.input_json
    inference_configs["num_workers"] = args.num_workers

    load_start = time.perf_counter()
    runner = get_default_runner(
        seeds=seeds,
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
                f"--input-indices {selected_indices} out of range for "
                f"{n_available} inputs"
            )
        dataset.inputs = [dataset.inputs[index] for index in selected_indices]
    work_items = make_work_items(
        n_inputs=len(dataset.inputs),
        seeds=seeds,
        repeat=args.repeat,
        min_items_per_rank=args.min_items_per_rank,
        world_size=DIST_WRAPPER.world_size,
    )
    assigned_items = work_items[DIST_WRAPPER.rank :: DIST_WRAPPER.world_size]

    meta = {
        "repo": git_info(repo),
        "rank": DIST_WRAPPER.rank,
        "world_size": DIST_WRAPPER.world_size,
        "local_rank": DIST_WRAPPER.local_rank,
        "load_sec": load_sec,
        "args": vars(args),
        "input_count": len(dataset.inputs),
        "assigned_items": len(assigned_items),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "env": {
            key: os.environ.get(key)
            for key in (
                "PROTENIX_ROOT_DIR",
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
            )
        },
    }
    (out_dir / f"meta_rank{DIST_WRAPPER.rank}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    profile_done = False
    warmup_source = assigned_items or work_items[:1]
    for idx, (repeat_idx, input_idx, seed) in enumerate(warmup_source[: args.warmup_items]):
        row = run_one(
            runner=runner,
            dataset=dataset,
            input_idx=input_idx,
            seed=seed + 90000001 + idx,
            repeat_idx=repeat_idx,
            args=args,
            phase="warmup",
            dump=False,
        )
        write_jsonl(metrics_path, row)

    for repeat_idx, input_idx, seed in assigned_items:
        profile_path = None
        if args.torch_profile and not profile_done and DIST_WRAPPER.rank == 0:
            profile_path = out_dir / "profiles" / "rank0_forward_trace.json"
            profile_done = True
        row = run_one(
            runner=runner,
            dataset=dataset,
            input_idx=input_idx,
            seed=seed,
            repeat_idx=repeat_idx,
            args=args,
            phase="timed",
            dump=args.dump,
            profile_path=profile_path,
        )
        write_jsonl(metrics_path, row)
        logging.info(
            "rank=%s input=%s seed=%s samples=%s forward=%.3fs total=%.3fs",
            DIST_WRAPPER.rank,
            row["input_name"],
            seed,
            row["prediction_samples"],
            row["forward_sec"],
            row["total_sec"],
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if DIST_WRAPPER.rank == 0:
        summary = aggregate_metrics(out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
