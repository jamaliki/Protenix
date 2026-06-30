# Protenix inference throughput campaign

Status: setup in progress

## Target workload

- Hardware: Sandpit Tokyo H100 80 GB GPUs; start with 1 GPU for hotspot screens, then gate on 8 GPUs for throughput.
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
  GPU count: 1-GPU screen, 8-GPU gate
metric window:
  warmup: recorded separately
  timed: all rows with phase=timed in metrics_rank*.jsonl
outputs:
  harness: scripts/perf/protenix_inference_benchmark.py
  slurm: scripts/perf/tokyo_protenix_benchmark.sbatch
```

## Running report

### Round 00 - harness and cluster baseline

Status: planned

Hypothesis:
- Before optimizing kernels, measure whether throughput is limited by model forward, feature preparation, dumping, single-GPU underutilization, or failure to distribute independent design samples across GPUs.

Change:
- Added a benchmark harness that records per-rank feature time, model forward time, dump time, peak memory, and aggregate samples/sec.
- Added a Tokyo Slurm launcher with explicit CPU and memory requests.

Experiment:
- Baseline: upstream default inference settings.
- Candidate: none.
- Hardware: first 1xH100 smoke, then 1xH100 default profile, then 8xH100 throughput gate.
- Command/job IDs: pending.

Results:
| shape | baseline | candidate | delta | memory | notes |
| --- | ---: | ---: | ---: | ---: | --- |

Interpretation:
- Pending cluster runs.

Decision:
- Pending.

Next:
- Install/verify the cluster environment, run a short smoke with reduced diffusion steps, then run the default baseline and torch profiler.
