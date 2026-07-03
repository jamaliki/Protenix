#!/usr/bin/env python3
"""Profile the actual batched campaign inference path.

The one-input benchmark is useful for old single-record latency work, but the
throughput branch now optimizes many independent records per GPU.  This helper
uses the same runner path as ``runner/batch_inference.py`` and only adds a
PyTorch Chrome trace around ``infer_predict``.  Use short diffusion step counts
for launch attribution, then gate any candidate with the full ``N_step=200``
batch runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import _repo_bootstrap  # noqa: F401

import torch

from configs.configs_inference import inference_configs
from runner.batch_inference import get_default_runner
from runner.campaign_inputs import resolve_inference_jsons, write_campaign_json
from runner.inference import infer_predict


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


def parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use-msa", type=str_bool, default=False)
    parser.add_argument("--use-template", type=str_bool, default=False)
    parser.add_argument("--use-rna-msa", type=str_bool, default=False)
    parser.add_argument("--triangle-attention", default="cuequivariance")
    parser.add_argument("--triangle-multiplicative", default="cuequivariance")
    parser.add_argument("--enable-cache", type=str_bool, default=True)
    parser.add_argument("--enable-fusion", type=str_bool, default=True)
    parser.add_argument("--enable-tf32", type=str_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-mode", default="auto")
    parser.add_argument("--token-bucket-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=optional_int, default=None)
    parser.add_argument("--sample-diffusion-chunk-size", type=optional_int, default=5)
    parser.add_argument(
        "--profile-pairformer-scopes",
        type=str_bool,
        default=True,
        help=(
            "Enable optional pairformer record_function ranges so the exported "
            "trace can attribute kernels to triangle attention, transitions, "
            "and token attention. Disable only when measuring profiler overhead."
        ),
    )
    parser.add_argument(
        "--sort-by-token-length",
        type=str_bool,
        default=True,
        help=(
            "Match runner/batch_inference.py campaign batching by writing a "
            "temporary length-sorted campaign JSON before profiling."
        ),
    )
    parser.add_argument("--profile-memory", type=str_bool, default=True)
    parser.add_argument("--top-rows", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    dump_dir = Path(args.dump_dir).resolve()
    profile_dir = dump_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    if args.profile_pairformer_scopes:
        os.environ["PROTENIX_PROFILE_PAIRFORMER_SCOPES"] = "1"
    input_jsons = resolve_inference_jsons(args.input_json)
    profile_input_json, _cleanup_json = write_campaign_json(
        input_jsons,
        str(dump_dir),
        sort_by_token_length=args.sort_by_token_length,
    )

    # get_default_runner reads the module-level inference config when it builds
    # the ConfigDict, so set the perf-run paths before constructing the runner.
    inference_configs["dump_dir"] = str(dump_dir / "predictions")
    inference_configs["input_json_path"] = profile_input_json
    inference_configs["num_workers"] = args.num_workers

    runner = get_default_runner(
        seeds=parse_seeds(args.seeds),
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
        inference_batch_size=args.batch_size,
        inference_batch_mode=args.batch_mode,
        inference_token_bucket_size=args.token_bucket_size,
    )
    configs = runner.configs
    configs.input_json_path = profile_input_json
    configs.num_workers = args.num_workers
    configs.infer_setting.chunk_size = args.chunk_size
    configs.infer_setting.sample_diffusion_chunk_size = (
        args.sample_diffusion_chunk_size
    )

    trace_path = profile_dir / "rank0_batched_inference_trace.json"
    top_path = profile_dir / "rank0_batched_inference_top_cuda.txt"
    meta_path = profile_dir / "profile_meta.json"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    start = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("protenix/batched_inference_campaign"):
            infer_predict(runner, configs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start

    profiler.export_chrome_trace(str(trace_path))
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    top_path.write_text(profiler.key_averages().table(sort_by=sort_key, row_limit=args.top_rows))
    meta_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "elapsed_sec": elapsed_sec,
                "profile_input_json": profile_input_json,
                "trace": str(trace_path),
                "top_cuda": str(top_path),
                "torch": {
                    "version": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "device": (
                        torch.cuda.get_device_name()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"elapsed_sec": elapsed_sec, "trace": str(trace_path)}, indent=2))


if __name__ == "__main__":
    main()
