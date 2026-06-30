# Protenix inference throughput campaign

Status: per-GPU optimization in progress

## Target workload

- Hardware: Sandpit Tokyo H100 80 GB GPUs; all headline speedups are per GPU.
  Multi-GPU runs are scaling sanity checks only.
- Primary metric: generated structure samples/sec after model load. End-to-end samples/sec including feature prep and dumping is tracked as a guardrail.
- Workload assumption: protein-design use means many independent samples/seeds for already prepared inputs. Online MSA/template search is not the GPU bottleneck and is excluded from the first GPU baseline unless stated otherwise.
- Default baseline: `protenix_base_default_v1.0.0`, bf16, 10 cycles, 200 diffusion steps, 5 samples, MSA enabled, cuequivariance triangle attention/multiplicative kernels, diffusion cache/fusion/TF32 enabled.
- Guardrails: output count, finite predictions, peak allocated/reserved GPU memory, no correctness-affecting dtype/kernel changes without an explicit opt-in and follow-up validation.

## Baseline checklist

```text
repo:
  branch: codex/protenix-throughput
  upstream: bytedance/Protenix main
  fork: jamaliki/Protenix
workload:
  input: examples/example.json unless a larger design panel is selected
  model/config: protenix_base_default_v1.0.0 defaults
  precision: bf16
  compile/checkpoint flags: default Protenix inference kernels, no torch.compile
hardware:
  cluster: Sandpit Tokyo
  GPU count: 1-GPU baseline/profile/gates
metric window:
  warmup: recorded separately
  timed: all rows with phase=timed in metrics_rank*.jsonl
outputs:
  harness: scripts/perf/protenix_inference_benchmark.py
  slurm: scripts/perf/tokyo_protenix_benchmark.sbatch
  trace reducer: scripts/perf/trace_aggregate.py
```

## Running report

### Round 00 - harness and cluster baseline

Status: complete

Hypothesis:
- Before optimizing kernels, measure whether throughput is limited by model forward, feature preparation, dumping, single-GPU underutilization, or failure to distribute independent design samples across GPUs.

Change:
- Added a benchmark harness that records per-rank feature time, model forward time, dump time, peak memory, and aggregate samples/sec.
- Added a Tokyo Slurm launcher with explicit CPU and memory requests.

Experiment:
- Baseline: upstream default inference settings on one H100.
- Candidate screen: larger `N_sample` values with default diffusion chunking.
- Profile: PyTorch/CUPTI Chrome trace on one default forward.

Results:
| run | samples/sec | forward samples/sec | peak allocated MiB | notes |
| --- | ---: | ---: | ---: | --- |
| default `N_sample=5`, `N_step=200`, 1xH100 | 0.528 | 0.544 | 2414 | warmed, dump disabled, 7r6r |
| `N_sample=20`, default diffusion chunk 5, 1xH100 | 0.728 | 0.735 | 2861 | still chunks denoising in groups of 5 |
| `N_sample=40`, default diffusion chunk 5, 1xH100 | 0.757 | 0.761 | 3707 | diminishing return |
| `N_sample=80`, default diffusion chunk 5, 1xH100 | 0.773 | 0.775 | 5487 | best screened value so far, only 1.47x per GPU |

Interpretation:
- Warmed feature preparation is small (`~0.26s`) and dumping is negligible when disabled.
- The default forward is dominated by the model, especially the 200-step diffusion loop.
- The first `N_sample` sweep did not unchunk diffusion: Protenix defaults
  `infer_setting.sample_diffusion_chunk_size=5`, so `N_sample=80` runs 16
  sequential diffusion chunks and mainly amortizes pairformer/confidence overhead.
- Raw trace for one default profiled forward:
  - `6,120,748` total trace events.
  - `474,531` CUDA kernel events.
  - `474,531` kernel launch API calls, with `~1.97s` CPU time in launch APIs.
  - PyTorch table reports `~3.51s` self CUDA kernel time for the profiled forward.
  - Largest kernel families by total CUDA time are many small cublas GEMMs,
    memory-efficient attention, copy/cast elementwise kernels, fast LayerNorm,
    CUTLASS/cuDNN attention kernels, and cuequivariance fused kernels.

Top launch families from `trace_aggregate.py`:

| family | calls | total CUDA/runtime time |
| --- | ---: | ---: |
| cublas sm90 GEMM 64x128 | 40,623 | 415 ms |
| PyTorch mem-efficient attention | 4,800 | 306 ms |
| cublas sm90 GEMM 128x64 | 24,000 | 201 ms |
| direct copy/cast elementwise | 24,919 | 362 ms |
| fast LayerNorm forward | 33,926 | 213 ms |
| cublas sm90 GEMM 128x128 | 10,105 | 149 ms |
| CUTLASS GEMM 128x64 | 11,403 | 138 ms |
| CUTLASS GEMM 256x64 | 4,800 | 126 ms |
| cuDNN SDPA | 1,200 | 124 ms |
| vectorized add/mul/sigmoid elementwise | >95,000 | >288 ms |
| cuequivariance fused gated dual GEMM | 2,160 | 93 ms kernel / 358 ms runtime region |
| cuequivariance triangle attention | 1,240 | 324 ms runtime region |

Decision:
- Do not claim multi-GPU scaling as an optimization. The 8-GPU sanity check
  reached 5.58 samples/sec aggregate, but that is not a valid per-GPU speedup.
- First per-GPU target is to remove default diffusion chunking for small/medium
  inputs on H100, then test CUDA graph capture or other launch-amortization
  strategies. Whole-block "mega kernels" are too risky until narrower fusion and
  launch scheduling have been measured.

Next:
- Expose `sample_diffusion_chunk_size` in the harness and run `N_sample=80`
  with `sample_diffusion_chunk_size=None` or `80` on one H100.
- If that improves per-GPU throughput, add an H100-oriented auto-chunk policy.
- Profile the best per-GPU setting with a smaller trace window, then evaluate
  CUDA graph capture for the fixed-shape denoising loop.
