#!/usr/bin/env python3
"""Measure same-shape Protenix inference batches sharing one GPU.

The production design workload is throughput-oriented: many independent
sequences, usually with ``N_sample`` equal to 1 or a small integer.  A single
large leading batch is one way to fill an H100, but it is not the only one.  If
the hot kernels are still too small or have synchronization gaps, two or more
moderate leading-batch workers can sometimes improve aggregate per-GPU
throughput.

This script is deliberately a profiler harness, not a serving implementation.
Each process loads its own model, warms up, then waits on a barrier so the
reported aggregate window covers only the overlapped ``runner.predict()`` calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import torch

from configs.configs_inference import inference_configs
from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.utils.seed import seed_everything
from runner.batch_inference import get_default_runner
from runner.inference import update_inference_configs
from scripts.perf.same_shape_batch_probe import (
    clone_tree,
    cuda_memory,
    finite_summary,
    optional_int,
    stack_tree,
    str_bool,
    timed_predict,
)


def worker_main(
    worker: int,
    args: argparse.Namespace,
    barrier: mp.Barrier,
    queue: mp.Queue,
) -> None:
    try:
        worker_dir = Path(args.dump_dir).resolve() / f"worker_{worker}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(worker_dir / "triton_cache")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(worker_dir / "torchinductor_cache")
        inference_configs["dump_dir"] = str(worker_dir / "predictions")
        inference_configs["input_json_path"] = args.input_json
        inference_configs["num_workers"] = 0

        runner = get_default_runner(
            seeds=[args.seed + worker],
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
        batch = {
            "input_feature_dict": stack_tree(
                [base_feature for _ in range(args.batch_size)]
            )
        }

        for index in range(args.warmup):
            seed_everything(args.seed + 9000 + worker * 100 + index, deterministic=False)
            timed_predict(runner, {"input_feature_dict": clone_tree(base_feature)})

        barrier.wait(timeout=args.barrier_timeout_sec)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        seed_everything(args.seed + worker, deterministic=False)
        wall_start = time.time()
        perf_start = time.perf_counter()
        prediction = runner.predict(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        perf_end = time.perf_counter()
        wall_end = time.time()

        queue.put(
            {
                "ok": True,
                "worker": worker,
                "batch_size": args.batch_size,
                "wall_start": wall_start,
                "wall_end": wall_end,
                "predict_sec": perf_end - perf_start,
                "model_timing": copy.deepcopy(getattr(runner, "last_log_dict", {})),
                "prediction": finite_summary(prediction),
                "shape": {
                    "n_token": n_token,
                    "n_atom": int(data["N_atom"].item()),
                    "n_msa": int(data["N_msa"].item()),
                },
                "feature_times": feature_times,
                **cuda_memory(),
            }
        )
    except BaseException as exc:  # propagate child failures into the JSON result
        queue.put(
            {
                "ok": False,
                "worker": worker,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default="examples/example.json")
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=200)
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
    parser.add_argument("--dump-dir", default="/tmp/protenix_same_shape_concurrency")
    parser.add_argument("--barrier-timeout-sec", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("same_shape_batch_concurrency_probe requires CUDA")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.workers)
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker_main, args=(worker, args, barrier, queue))
        for worker in range(args.workers)
    ]
    for proc in procs:
        proc.start()

    rows = [queue.get() for _ in procs]
    for proc in procs:
        proc.join()

    ok_rows = [row for row in rows if row.get("ok")]
    summary: dict[str, Any] = {
        "args": vars(args),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "total_sequences": args.workers * args.batch_size,
        "rows": sorted(rows, key=lambda row: row.get("worker", -1)),
    }
    if ok_rows:
        wall_sec = max(row["wall_end"] for row in ok_rows) - min(
            row["wall_start"] for row in ok_rows
        )
        summary.update(
            {
                "ok_workers": len(ok_rows),
                "aggregate_wall_sec": wall_sec,
                "aggregate_sequences_per_sec": len(ok_rows)
                * args.batch_size
                / wall_sec,
                "mean_worker_predict_sec": sum(row["predict_sec"] for row in ok_rows)
                / len(ok_rows),
                "max_worker_predict_sec": max(row["predict_sec"] for row in ok_rows),
                "max_peak_allocated_mib": max(
                    row["peak_allocated_mib"] for row in ok_rows
                ),
                "max_peak_reserved_mib": max(row["peak_reserved_mib"] for row in ok_rows),
            }
        )
    else:
        summary["ok_workers"] = 0

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
