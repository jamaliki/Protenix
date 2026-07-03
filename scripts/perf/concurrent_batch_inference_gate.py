#!/usr/bin/env python3
"""Measure same-GPU concurrency for batched campaign inference.

The optimized mixed-sequence path still launches many small H100 kernels.  If
one B16 campaign does not fill the GPU, running two warm workers on the same GPU
can improve per-GPU throughput without changing model math.  This script keeps
the comparison fair by loading persistent workers, running a short warmup, then
synchronizing the full timed campaign.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import datetime as dt
import json
import logging
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from queue import Empty
from typing import Any

import torch

from configs.configs_inference import inference_configs
from runner.batch_inference import get_default_runner
from runner.campaign_inputs import resolve_inference_jsons, write_campaign_json
from runner.inference import infer_predict


TIMING_RE = re.compile(
    r"Batch model timing total .*?, (?P<items>\d+) input\(s\), "
    r"predict (?P<predict>[0-9.]+)s"
)
START_RE = re.compile(r"TIMED_START worker=(?P<worker>\d+) repeat=(?P<repeat>\d+)")
DONE_RE = re.compile(r"TIMED_DONE worker=(?P<worker>\d+) repeat=(?P<repeat>\d+)")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s,%(msecs)-3d %(levelname)-8s "
            "[%(filename)s:%(lineno)d] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(path, mode="w", encoding="utf-8")],
        force=True,
    )


def set_steps(runner: Any, n_step: int) -> None:
    runner.configs.sample_diffusion.N_step = n_step
    runner.model.configs.sample_diffusion.N_step = n_step


def set_output_dir(runner: Any, path: Path) -> None:
    """Move Protenix's stateful dumper before each measured pass.

    ``InferenceRunner`` captures ``dump_dir`` inside both ``runner`` and its
    ``DataDumper`` at construction time.  Reusing that directory after warmup
    lets Protenix skip or overwrite already-produced outputs, which measures
    filesystem bookkeeping instead of model work.  Retargeting all three pieces
    keeps warmup and timed repeats independent while preserving one loaded
    model per worker.
    """
    path.mkdir(parents=True, exist_ok=True)
    runner.configs.dump_dir = str(path)
    runner.dump_dir = str(path)
    runner.error_dir = str(path / "ERR")
    Path(runner.error_dir).mkdir(parents=True, exist_ok=True)
    runner.init_dumper(
        need_atom_confidence=runner.configs.need_atom_confidence,
        sorted_by_ranking_score=runner.configs.sorted_by_ranking_score,
    )


def make_runner(args: argparse.Namespace, worker_dir: Path) -> Any:
    input_jsons = resolve_inference_jsons(args.input_json)
    campaign_json, _cleanup = write_campaign_json(
        input_jsons, str(worker_dir), sort_by_token_length=True
    )
    inference_configs["dump_dir"] = str(worker_dir / "predictions")
    inference_configs["input_json_path"] = campaign_json
    inference_configs["num_workers"] = 0
    runner = get_default_runner(
        seeds=[args.seed],
        n_cycle=args.n_cycle,
        n_step=args.n_step,
        n_sample=args.n_sample,
        dtype=args.dtype,
        model_name=args.model_name,
        use_msa=False,
        trimul_kernel=args.triangle_multiplicative,
        triatt_kernel=args.triangle_attention,
        enable_cache=True,
        enable_fusion=True,
        enable_tf32=True,
        use_template=False,
        use_rna_msa=False,
        inference_batch_size=args.batch_size,
        inference_batch_mode=args.batch_mode,
        inference_token_bucket_size=0,
    )
    runner.configs.input_json_path = campaign_json
    runner.configs.num_workers = 0
    return runner


def worker_main(
    worker: int,
    args_dict: dict[str, Any],
    barrier: mp.Barrier,
    queue: mp.Queue,
) -> None:
    args = argparse.Namespace(**args_dict)
    worker_dir = Path(args.run_dir) / f"worker_{worker}"
    log_path = worker_dir / "worker.log"
    configure_logging(log_path)
    try:
        logging.info("WORKER_INIT worker=%s pid=%s", worker, os.getpid())
        runner = make_runner(args, worker_dir)
        set_steps(runner, args.warmup_step)
        set_output_dir(runner, worker_dir / "warmup_predictions")
        runner.configs.seeds = [args.seed + 10_000 + worker]
        logging.info("WARMUP_START worker=%s", worker)
        infer_predict(runner, runner.configs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logging.info("WARMUP_DONE worker=%s", worker)

        timed_results = []
        for repeat in range(args.repeats):
            runner.configs.seeds = [args.seed + 100_000 * repeat + worker]
            set_steps(runner, args.n_step)
            set_output_dir(runner, worker_dir / f"timed_predictions_r{repeat}")
            barrier.wait(timeout=args.barrier_timeout_sec)
            start_epoch = time.time()
            start_perf = time.perf_counter()
            logging.info("TIMED_START worker=%s repeat=%s", worker, repeat)
            infer_predict(runner, runner.configs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_perf
            end_epoch = time.time()
            logging.info(
                "TIMED_DONE worker=%s repeat=%s elapsed_sec=%.6f",
                worker,
                repeat,
                elapsed,
            )
            timed_results.append(
                {
                    "repeat": repeat,
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "elapsed_sec": elapsed,
                }
            )

        queue.put(
            {
                "worker": worker,
                "ok": True,
                "log_path": str(log_path),
                "timed": timed_results,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in Slurm failures.
        logging.exception("WORKER_FAILED worker=%s", worker)
        queue.put(
            {
                "worker": worker,
                "ok": False,
                "log_path": str(log_path),
                "error": repr(exc),
            }
        )


def parse_worker_log(path: Path) -> dict[str, Any]:
    active = False
    predict_sec = 0.0
    items = 0
    batches = 0
    for line in path.read_text(errors="replace").splitlines():
        if START_RE.search(line):
            active = True
            continue
        if DONE_RE.search(line):
            active = False
            continue
        if not active:
            continue
        match = TIMING_RE.search(line)
        if match:
            batches += 1
            items += int(match.group("items"))
            predict_sec += float(match.group("predict"))
    return {"timed_batches": batches, "timed_items": items, "predict_sec": predict_sec}


def collect_results(queue: mp.Queue, expected: int) -> list[dict[str, Any]]:
    results = []
    deadline = time.time() + 24 * 3600
    while len(results) < expected and time.time() < deadline:
        try:
            results.append(queue.get(timeout=10))
        except Empty:
            continue
    return sorted(results, key=lambda row: row["worker"])


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.workers)
    queue: mp.Queue = ctx.Queue()
    args_dict = vars(args)
    processes = [
        ctx.Process(target=worker_main, args=(worker, args_dict, barrier, queue))
        for worker in range(args.workers)
    ]
    for process in processes:
        process.start()
    results = collect_results(queue, args.workers)
    for process in processes:
        process.join()

    if len(results) != args.workers or any(not row.get("ok") for row in results):
        return {"ok": False, "workers": results}

    parsed = []
    for row in results:
        log_summary = parse_worker_log(Path(row["log_path"]))
        parsed.append({**row, **log_summary})

    start = min(timed["start_epoch"] for row in parsed for timed in row["timed"])
    end = max(timed["end_epoch"] for row in parsed for timed in row["timed"])
    wall_sec = end - start
    parsed_records = sum(row["timed_items"] for row in parsed)
    configured_records = args.workers * args.repeats * args.records_per_campaign
    if parsed_records != configured_records:
        return {
            "ok": False,
            "args": vars(args),
            "workers": parsed,
            "timed_records": parsed_records,
            "configured_records": configured_records,
            "error": "timed model logs did not cover the expected records",
        }
    generated = parsed_records * args.n_sample
    predict_sec = sum(row["predict_sec"] for row in parsed)
    return {
        "ok": True,
        "args": vars(args),
        "workers": parsed,
        "start_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
        "end_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
        "wall_sec": wall_sec,
        "timed_records": parsed_records,
        "configured_records": configured_records,
        "generated_samples": generated,
        "wall_samples_per_sec": generated / wall_sec,
        "sum_predict_sec": predict_sec,
        "predict_samples_per_sec": generated / predict_sec if predict_sec else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--records-per-campaign", type=int, default=32)
    parser.add_argument(
        "--barrier-timeout-sec",
        type=float,
        default=600.0,
        help="Fail rather than hang forever if a peer dies before a timed repeat.",
    )
    parser.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--warmup-step", type=int, default=2)
    parser.add_argument("--n-sample", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-mode", default="auto")
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.repeats < 1:
        raise ValueError("--workers and --repeats must be positive")
    result = run_gate(args)
    out_path = Path(args.run_dir) / "summary.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
