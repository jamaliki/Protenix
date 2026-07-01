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

## Current promoted one-H100 throughput stack

Use the following for the current best high-throughput 7r6r-style inference
path:

```bash
# Max-throughput point screened so far for the 7r6r-style benchmark.
# Use N_SAMPLE=1280 instead when halving memory is worth a ~1.5% throughput hit.
export N_SAMPLE=2560
export SAMPLE_DIFFUSION_CHUNK_SIZE=none
export CHUNK_SIZE=none
export PROTENIX_GPU_RANDOM_AUGMENT=1
export PROTENIX_BF16_DIFFUSION_CORE=1
export PROTENIX_ATTENTION_FORCE_FP32=0
export PROTENIX_TRITON_LOCAL_ATTN=1
export PROTENIX_TRITON_LOCAL_ATTN_NUM_WARPS=1
export PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16=1
export PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS=1
export PROTENIX_TRITON_FUSED_ELEMENTWISE=1
export PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL=1
export PROTENIX_BF16_ATOM_ATTENTION=1
export PROTENIX_SUMMARY_SAMPLE_CHUNK_SIZE=32
export PROTENIX_STREAM_CONFIDENCE_SUMMARY=1
export PROTENIX_CONFIDENCE_LOGIT_CHUNK_SIZE=32
export PROTENIX_BROADCAST_DIFFUSION_S=1
```

Measured one-H100 warmed throughput at commit `8142521` is `9.178`
end-to-end samples/sec and `9.187` forward samples/sec at `N_sample=2560`,
`N_step=200`, true unchunked diffusion on the 7r6r benchmark input, reserving
`36.3 GiB`. This is `17.38x` over the original per-GPU default aggregate
baseline of `0.528` samples/sec. The lower-memory `N_sample=1280` point gives
`9.043` end-to-end samples/sec and `9.060` forward samples/sec while reserving
`19.1 GiB`, so it remains the better debug/default setting when memory
headroom matters. Larger unchunked batches are not currently safe:
`N_sample=3840` and `N_sample=5120` both fail during the atom encoder warmup in
a BF16 cuBLAS GEMM followed by an illegal-memory-access report.

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

### Round 01 - unchunk diffusion for high-throughput inference

Status: promoted as an opt-in throughput setting

Hypothesis:
- For protein-design throughput, large `N_sample` batches should run through the
  diffusion denoiser as one chunk on H100 instead of Protenix's default chunks
  of 5 samples.

Results:
| run | aggregate samples/sec | forward samples/sec | peak allocated MiB | speedup vs default |
| --- | ---: | ---: | ---: | ---: |
| default `N_sample=5`, chunk 5 | 0.528 | 0.544 | 2414 | 1.00x |
| `N_sample=80`, chunk `None` | 2.436 | 2.457 | 5487 | 4.61x |

Decision:
- Keep the harness support for `sample_diffusion_chunk_size=None`. This is the
  largest clean per-GPU win so far and does not change model math.

### Round 02 - diffusion hotspot timing

Status: complete

Experiment:
- Added CUDA-event timing inside `DiffusionModule`, exported through the
  inference harness as `model_log.time.*`.
- Representative screen: `N_step=20`, `N_sample=80`, unchunked, 7r6r, one H100.

Results:
| component | time at N20/N80 |
| --- | ---: |
| pairformer | ~2.6 s |
| diffusion total | ~2.0-2.8 s depending dtype policy |
| confidence | ~1.3 s |
| diffusion atom encoder | ~0.7-1.1 s |
| diffusion transformer | ~0.7-1.1 s |
| diffusion atom decoder | ~0.56-0.95 s |
| conditioning/input/output rescale | negligible |

Interpretation:
- At high `N_sample`, launch overhead is not the dominant remaining issue.
  The diffusion loop is dominated by atom local attention plus the token
  diffusion transformer.
- Whole-model mega-kernels are not the right next unit. The right unit is
  local atom attention and its bias/projection path, plus dtype/backend policy
  for token attention.

### Round 03 - rejected broad fusions and CUDA graphs

Status: rejected

Results:
| candidate | result | decision |
| --- | ---: | --- |
| Attention q/k/v/g projection concat fusion | full N200/N80 forward ~33.04 s | rejected; slower than unchunked baseline |
| Transition a1/a2 projection concat fusion | full N200/N80 forward ~32.89 s | rejected; slower than unchunked baseline |
| CUDA graph replay for denoiser | full N200/N80 forward ~36.35 s | rejected; slower, likely memory/copy and graph-management overhead |
| BF16 atom attention | finite but atom encoder/decoder slower | rejected |
| Atom-conditioning cache | N20/N80 forward 7.043 -> 7.002 s | rejected alone; too small for persistent-cache complexity |

Decision:
- Do not carry these experimental paths in the branch. They were removed after
  measurement.

### Round 04 - GPU augmentation and diffusion transformer BF16

Status: partially promoted behind explicit environment flags

Hypothesis:
- The SciPy/CPU random rotation path inside every diffusion step is avoidable
  for inference.
- The token diffusion transformer can use BF16 arithmetic on H100, while atom
  local attention should stay FP32 unless a dedicated kernel replaces it.

Change:
- `PROTENIX_GPU_RANDOM_AUGMENT=1`: generate random rotations/translations on GPU
  for inference calls with no mask and no gradients.
- `PROTENIX_BF16_DIFFUSION_CORE=1`: run the token diffusion transformer under
  BF16 autocast and cast back to FP32 before the atom decoder.
- `PROTENIX_ATTENTION_FORCE_FP32=0`: allow full token attention to stay BF16,
  but the implementation now forces local atom-attention shapes back to FP32.
- Added shape-specific SDPA fallback for cuDNN "no valid execution plan"
  failures; only the failing signature is routed to non-cuDNN SDPA backends.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | peak allocated MiB | speedup vs default |
| --- | ---: | ---: | ---: | ---: | ---: |
| default `N_sample=5`, chunk 5 | 0.528 | 0.544 | 9.198 for 5 samples | 2414 | 1.00x |
| `N_sample=80`, unchunked | 2.436 | 2.457 | 32.55 | 5487 | 4.61x |
| GPU aug + BF16 transformer, `N_sample=80` | 3.247 | 3.282 | 24.37 | 5865 | 6.15x |
| GPU aug + split attention policy, `N_sample=160` | 3.397 | 3.417 | 46.83 | 9425 | 6.43x |

Guardrails:
- Timed `N20/N80` split-policy smoke: finite coordinates, 6.873 s forward,
  diffusion 2.041 s, pairformer 2.581 s, confidence 1.344 s.
- Full `N200/N160` split-policy gate: finite coordinates, 47.09 s end-to-end
  for the timed item, 9.4 GiB peak allocated.

Rejected dtype/backend variants:
| candidate | result | decision |
| --- | ---: | --- |
| unguarded BF16 SDPA everywhere | crashes for `N20/N80` with cuDNN no-plan | rejected |
| global sticky non-cuDNN SDPA fallback | `N200/N160` 3.25 samples/sec | rejected; slows valid token-attention shapes |
| always-FP32 q/k/v attention inside BF16 core | `N200/N160` 3.01 samples/sec | rejected; robust but too slow |
| BF16/fallback for atom local attention | atom encoder+decoder N20/N80 2.05 s vs 1.29 s split policy | rejected |

Decision:
- Promote the robust split policy as the current branch candidate, but it is
  still short of the 10x target. Current best robust per-GPU speedup is ~6.4x.
- The earlier unsafe `N160` screen reached 3.655 samples/sec, but it is not
  acceptable as a default because smaller `N20/N80` shapes can crash in cuDNN
  SDPA.

Next:
- The remaining meat is atom local attention. A credible next attempt is a
  dedicated Triton/CUDA local attention kernel for the fixed `n_queries=32`,
  `n_keys=128`, `n_heads=4`, `c_atom=128` atom transformer path, including pair
  bias. A whole-block mega-kernel is too broad; the right fusion boundary is
  local q/k/v projection + bias application + SDPA for the 32x128 windows.

### Round 09 - Triton atom local attention and true unchunked attention

Status: promoted as an opt-in throughput path

Hypothesis:
- The local atom attention path spends material time in dense trunk
  rearrangement, FP32 upcasts, pair-bias addition, and SDPA dispatch for fixed
  `32x128` windows. A narrow Triton kernel can compute the pair-biased local
  attention directly over the original `[... , H, N_atom, D]` q/k/v layout and
  broadcasted pair bias.

Change:
- Added `PROTENIX_TRITON_LOCAL_ATTN=1`, a guarded H100-oriented Triton forward
  kernel for local atom attention with `n_queries=32`, `n_keys=128`,
  `head_dim=32`, FP32/BF16 inputs, FP32 output, and broadcasted pair bias.
- Added an isolated hotspot benchmark:
  `scripts/perf/local_attention_hotspot.py`.
- Fixed the benchmark harness so `--chunk-size none` actually sets
  `runner.configs.infer_setting.chunk_size = None`. Previously the harness left
  the default `chunk_size=256` in place, which meant diffusion atom attention
  was still chunked even in "unchunked" runs.

Hotspot screen:
| shape | reference | Triton | speedup | parity |
| --- | ---: | ---: | ---: | --- |
| BF16 q/k/v, `N_sample=80`, 7r6r atom count | 4.48 ms | 0.89 ms | 5.0x | max abs error `5.3e-6`, no NaNs |
| BF16 q/k/v, `N_sample=160`, 7r6r atom count | 8.80 ms | 1.75 ms | 5.0x | max abs error `7.2e-6`, no NaNs |

Diagnostics:
- Initial model gates showed no atom-time movement because the real diffusion
  path was still using `attn_chunk_size=256`; the Triton fast path is correctly
  disabled for chunked local attention.
- After fixing `chunk_size=None`, the next blocker was broadcasted diffusion
  pair bias (`[1, H, n_trunks, 32, 128]` bias for `[N_sample, H, N_atom, D]`
  q/k/v). The kernel now supports this by using the expanded bias view without
  materializing a copy.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | peak allocated MiB | speedup vs default |
| --- | ---: | ---: | ---: | ---: | ---: |
| default `N_sample=5`, chunk 5 | 0.528 | 0.544 | 9.198 for 5 samples | 2414 | 1.00x |
| previous robust split policy, `N_sample=160` | 3.397 | 3.417 | 46.83 | 9425 | 6.43x |
| true `chunk_size=None`, no Triton, `N_sample=160` | 4.501 | 4.535 | 35.28 | 9425 | 8.52x |
| true `chunk_size=None` + Triton local attention, `N_sample=160` | 4.996 | 5.038 | 31.76 | 9425 | 9.46x |
| true `chunk_size=None` + Triton local attention, `N_sample=320` | 5.347 | 5.371 | 59.58 | 16545 | 10.13x |

Representative component movement at `N_step=200`, `N_sample=160`:
| component | no Triton | Triton local attention | delta |
| --- | ---: | ---: | ---: |
| diffusion total | 28.12 s | 24.44 s | -13.1% |
| atom encoder | 7.68 s | 5.84 s | -24.0% |
| atom decoder | 7.17 s | 5.34 s | -25.6% |
| diffusion transformer | 12.88 s | 12.86 s | unchanged |
| pairformer | 2.61 s | 2.67 s | unchanged/noise |
| confidence | 2.77 s | 2.81 s | unchanged/noise |

Decision:
- Promote the harness chunk-size fix and the guarded Triton atom-local
  attention path. The branch now has a measured `10.13x` per-GPU throughput
  improvement over the original default on the 7r6r benchmark.
- Do not stop here. After atom attention, the remaining large measured
  component is `diffusion_transformer_sec`, which is dominated by full
  token/trunk attention and transition work. The next credible fusion boundary
  is pair-biased full-token attention, especially fusing pair-bias projection
  from `z` into score computation rather than materializing a separate
  `[H, N_token, N_token]` bias tensor.

Next:
- Build an isolated benchmark for `AttentionPairBias.standard_multihead_attention`
  at the real 7r6r token count and sample counts.
- Compare the current `layernorm_z -> linear_z -> SDPA -> gate -> output`
  path with a full-token Triton attention prototype that reads/project pair
  bias on the fly.
- Promote only if it moves real `diffusion_transformer`, pairformer, or
  confidence timings in the full inference gate.

### Round 10 - trunk block launch diagnosis and gated elementwise fusion

Status: promoted behind an explicit environment flag

Hypothesis:
- The next large measured component after atom local attention is the
  diffusion token transformer. A fused pair-bias/attention "mega-kernel" would
  only be worthwhile if pair-bias materialization is material in the real
  layout. If the real bottleneck is instead many elementwise gates around
  cuDNN SDPA and GEMMs, narrower forward-only Triton elementwise fusion should
  move the trunk without replacing high-quality tensor-core kernels.

Diagnostics:
- Real model shape is 24 trunk blocks, 16 heads, `c_token=768`, `c_s=384`,
  `c_z=128`, head dim 48, token count 245 for 7r6r.
- `z_pair` is expanded with a singleton sample dimension and broadcast across
  `N_sample`; the efficient path receives `z` as `[1, 128, 245, 245]`, not
  `[N_sample, 128, 245, 245]`.
- Faithful trunk hotspot at `N_sample=160`, BF16 token attention:
  - full block: `2.74 ms`
  - attention-pair-bias half: `1.34 ms`
  - conditioned transition: `1.12 ms`
  - pair-bias projection: only `0.03 ms`
- Therefore fusing pair-bias projection into SDPA is not the next meat for this
  benchmark; it would target well under 2% of trunk block time.

Top CUDA launch families in one full trunk block (`N_sample=160`, `z_samples=1`):
| family | launches | CUDA time |
| --- | ---: | ---: |
| BF16 tensor-core GEMM kernels (`nvjet_sm90_tst_..._TNT`) | 9 | 0.721 ms |
| cuDNN flash SDPA | 1 | 0.432 ms |
| BF16 vectorized multiply kernels | 6 | 0.427 ms |
| BF16 vectorized add kernels | 4 | 0.244 ms |
| BF16 tensor-core GEMM kernels with bias | 5 | 0.241 ms |
| BF16 sigmoid kernels | 5 | 0.210 ms |
| fast LayerNorm kernels | 4 | 0.155 ms |
| BF16 SiLU kernel | 1 | 0.082 ms |

Runtime launch profile for the same block:
- 60 CUDA kernel launches total.
- `cudaLaunchKernel`: 44 calls, `0.270 ms` runtime API time.
- `cuLaunchKernelEx`: 15 calls, `0.074 ms` runtime API time.
- `cudaMemsetAsync`: 15 calls, `0.069 ms` runtime API time.

Change:
- Added `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`, a guarded inference-only Triton
  path for repeated contiguous elementwise gates:
  - `sigmoid(x) * y + z` in `AdaptiveLayerNorm`
  - `sigmoid(x) * y` in attention/output gating
  - `silu(x) * y` in `ConditionedTransitionBlock`
- The path falls back automatically when Triton is unavailable, gradients are
  enabled, tensors are non-CUDA, dtypes differ, shapes differ, or tensors are
  non-contiguous.

Rejected screen:
- `torch.compile` on a single trunk block gave only `2.735 -> 2.664 ms`
  (`1.027x`) and hit graph breaks on the custom fast-LayerNorm extension.
  This is too small and brittle to promote.
- A stride-aware attention-gate Triton variant for the non-contiguous SDPA
  output did not earn its complexity: the fused hotspot stayed effectively
  unchanged (`2.395 -> 2.395 ms` at N160 and `4.512 -> 4.503 ms` at N320).
  It was reverted.

Hotspot screen:
| shape | baseline block | fused block | speedup | notes |
| --- | ---: | ---: | ---: | --- |
| `N_sample=160`, `z_samples=1` | 2.728 ms | 2.395 ms | 1.14x | transition 1.118 -> 0.902 ms |
| `N_sample=320`, `z_samples=1` | 5.154 ms | 4.512 ms | 1.14x | transition 2.119 -> 1.735 ms |

Numerical check:
- Same-weight trunk block comparison at `N_sample=160`:
  - finite outputs for both paths
  - max abs diff `0.03125`
  - mean abs diff `8.5e-5`
  - sampled p99 abs diff `0.00390625`
  - high max relative differences are only near zero-valued references.

Representative full inference gates:
| run | aggregate samples/sec | forward samples/sec | forward sec | diffusion transformer | peak allocated MiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N160, fused off | 5.016 | 5.056 | 31.64 | 12.83 s | 9425 | baseline |
| N160, fused on | 5.370 | 5.418 | 29.53 | 11.15 s | 9425 | +7.1% aggregate |
| N320, fused off | 5.352 | 5.376 | 59.52 | 24.77 s | 16545 | baseline |
| N320, fused on | 5.707 | 5.734 | 55.81 | 21.56 s | 16545 | +6.6% aggregate |

Decision:
- Promote the guarded elementwise fusion flag. It moves the measured
  diffusion-transformer component in the real workload and raises the best
  per-GPU 7r6r throughput from `5.347` to `5.707` aggregate samples/sec.
- The campaign is now `10.81x` over the original default aggregate baseline
  (`5.707 / 0.528`) on one H100, without using extra GPUs.

Next:
- The remaining trunk time is dominated by tensor-core GEMMs and cuDNN SDPA.
  Replacing these with a broad hand-written "mega-kernel" is unlikely to win
  unless it preserves tensor-core efficiency and removes enough memory traffic.
- The next credible kernel target is a more ambitious fused transition epilogue
  that combines GEMM outputs with SiLU/mul/output gating without replacing the
  GEMMs themselves, or a true GEMM-epilogue integration rather than another
  standalone elementwise launch.

### Round 11 - local attention TF32 precision tuning

Status: promoted as the default precision for the guarded Triton local-attention path

Hypothesis:
- After Round 10, the largest owned kernel in the current profile is the
  Triton atom local-attention kernel. It was using `input_precision="tf32x3"`
  for FP32 dot products, which is more accurate but slower than standard TF32.
  Since the inference path already enables TF32 globally, using standard TF32
  inside the opt-in Triton local-attention fast path should improve throughput
  with acceptable inference-level numerical movement.

Current profile:
- Warmed N20/N80 profile with accepted flags showed `_local_attention_kernel`
  as a top owned kernel family: 123 launches, 211 ms total, about 1.72 ms per
  call.
- Other large families are mostly outside this branch's easy control:
  cuequivariance pairformer kernels, cuDNN SDPA, cublas GEMMs, fast LayerNorm,
  and copy/cast traffic.

Change:
- Added `PROTENIX_TRITON_LOCAL_ATTN_INPUT_PRECISION={tf32,tf32x3,ieee}`.
- Added `PROTENIX_TRITON_LOCAL_ATTN_NUM_WARPS={1,2,4,8}` for screening.
- Changed the guarded Triton local-attention default from `tf32x3` to `tf32`.
  Users can still set `PROTENIX_TRITON_LOCAL_ATTN_INPUT_PRECISION=tf32x3` for
  the previous conservative precision.

Hotspot screen (`N_sample=80`, outer batch 1, 7r6r atom count):
| dtype | precision | warps | candidate ms | max abs error vs PyTorch ref | notes |
| --- | --- | ---: | ---: | ---: | --- |
| FP32 | tf32x3 | 4 | 1.784 | `1.2e-5` | previous default |
| FP32 | tf32x3 | 8 | 2.579 | `1.2e-5` | rejected |
| FP32 | tf32 | 4 | 0.942 | `1.39e-2` | promoted |
| FP32 | tf32 | 8 | 1.501 | `1.39e-2` | rejected |
| BF16 | tf32x3 | 4 | 0.887 | `5.3e-6` | previous BF16 behavior |
| BF16 | tf32 | 4 | 0.645 | `2.27e-3` | faster, same promoted default |

Representative full inference gates:
| run | aggregate samples/sec | forward samples/sec | forward sec | atom encoder | atom decoder | peak allocated MiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N160, local precision tf32x3 | 5.361 | 5.408 | 29.58 | 5.56 s | 5.12 s | 9425 | baseline |
| N160, local precision tf32 | 5.779 | 5.833 | 27.43 | 4.55 s | 4.10 s | 9425 | +7.8% aggregate |
| N320, local precision tf32x3 | 5.757 | 5.784 | 55.33 | 10.88 s | 10.18 s | 16545 | baseline |
| N320, local precision tf32 | 6.227 | 6.258 | 51.13 | 8.86 s | 8.08 s | 16545 | +8.2% aggregate |

Guardrails:
- Full inference coordinates were finite in both N160 and N320 gates.
- Peak allocated memory was unchanged.
- The hotspot numerical movement is larger than `tf32x3` but is consistent
  with standard TF32 inference and remains behind the Triton local-attention
  opt-in path.

Decision:
- Promote standard TF32 as the default dot precision for
  `PROTENIX_TRITON_LOCAL_ATTN=1`.
- The best measured one-H100 7r6r throughput is now `6.227` aggregate
  samples/sec at `N_sample=320`, `N_step=200`, or `11.79x` over the original
  default aggregate baseline (`6.227 / 0.528`).

Next:
- Re-profile the new best path if continuing. The atom local-attention kernel
  is much smaller now; remaining material costs are likely tensor-core GEMMs,
  cuequivariance pairformer/confidence kernels, trunk SDPA, and host-visible
  copy/cast traffic.

### Round 12 - rejected generic Transition activation fusion

Status: rejected and reverted

Hypothesis:
- The generic `Transition` module still performs separate SiLU and multiply
  launches in inference. Reusing the existing Triton `silu(x) * y` helper behind
  `PROTENIX_TRITON_FUSED_TRANSITION=1` might remove a small amount of launch and
  memory traffic outside the diffusion conditioned-transition block.

Experiment:
- Baseline: latest promoted Round 11 path, `N_sample=160`, `N_step=200`,
  `PROTENIX_TRITON_FUSED_TRANSITION=0`.
- Candidate: same flags plus `PROTENIX_TRITON_FUSED_TRANSITION=1`.
- Hardware: one Sandpit Tokyo H100.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round12_transition_fusion_gate_n160_20260630_212917`.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | diffusion transformer | peak allocated MiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N160, generic transition fusion off | 5.709 | 5.762 | 27.77 | 11.17 s | 9425 | baseline |
| N160, generic transition fusion on | 5.575 | 5.627 | 28.43 | 11.17 s | 9425 | -2.3% aggregate |

Interpretation:
- The candidate did not move the diffusion-transformer timer and slightly
  worsened full forward throughput. The slowdown is likely overhead/noise in
  non-dominant transition uses rather than movement of the real bottleneck.

Decision:
- Reject and revert the generic `Transition` feature flag and code. Do not carry
  a default-off path without a representative workload win.

Next:
- Re-profile the current promoted Round 11 baseline before writing another
  fusion. The next candidate should target a launch family that remains material
  after TF32 local attention, not a generic helper that only looks plausible from
  code structure.

### Round 13 - rejected Triton full-token attention replacement

Status: rejected and reverted

Hypothesis:
- The diffusion trunk full-token attention has the stable shape
  `[N_sample, 16 heads, 245 tokens, head_dim 48]` with broadcast pair bias. A
  Triton flash-style forward kernel might extend the atom-attention approach to
  trunk attention and remove cuDNN SDPA dispatch/materialization overhead.

Change:
- Added an opt-in `PROTENIX_TRITON_FULL_ATTN=1` prototype for full-token
  attention with broadcast `[1|N_sample, 1|heads, N, N]` pair bias.
- The first compile attempt exposed a Triton dot dtype mismatch in the value
  matmul (`fp32` softmax probabilities times `bf16` values); that was fixed by
  casting values to `fp32`, matching the atom local-attention kernel pattern.

Hotspot screen:
- Hardware: one Sandpit Tokyo H100.
- Command shape: `scripts/perf/trunk_attention_hotspot.py --samples 160
  --z-samples 1 --tokens 245 --dtype bfloat16 --warmup 10 --iters 40
  --efficient-fusion 1`.
- Shared env: `PROTENIX_ATTENTION_FORCE_FP32=0`,
  `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`.

Results:
| run | full block | attention-pair-bias half | attention qkv/sdpa/gate/out | transition | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| current cuDNN SDPA path | 2.411 ms | 1.203 ms | 0.993 ms | 0.946 ms | finite |
| `PROTENIX_TRITON_FULL_ATTN=1` | 3.063 ms | 1.423 ms | 1.125 ms | 1.158 ms | finite, slower |

Interpretation:
- This is the wrong fusion boundary. cuDNN SDPA is already efficient for the
  245-token trunk shape, and a broad replacement loses more attention time than
  it can recover from launch/bias overhead.
- Fusing pair-bias projection into such a kernel would not rescue it for this
  benchmark: the measured pair-bias projection is only about `0.033 ms/block`.

Decision:
- Reject and revert. Do not carry a full-attention replacement path unless a
  future profile shows cuDNN SDPA itself has become the dominant, inefficient
  kernel or a kernel design preserves cuDNN-level tensor-core efficiency.

Next:
- Focus on remaining material launch families that do not replace strong
  vendor kernels: copy/cast/layout traffic, confidence/pairformer overhead, or
  true GEMM epilogues where the tensor-core GEMM remains the compute engine.

### Round 14 - current-best full trace diagnosis

Status: complete

Experiment:
- Full PyTorch/CUPTI trace of the current promoted path after the Round 13
  cleanup, commit `392146b`, one Sandpit Tokyo H100.
- Shape: `N_sample=160`, `N_step=200`, true unchunked diffusion,
  `PROTENIX_TRITON_LOCAL_ATTN=1`,
  `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`, standard-TF32 local attention.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round13_profile_current_best_srun_20260630_220939`.
- Note: CUPTI inflated the profiled forward wall time to `468.7 s`; throughput
  from this run is invalid. Kernel launch families and CUDA totals are the
  useful signal.

Top kernel families:
| family | calls | CUDA time |
| --- | ---: | ---: |
| BF16 tensor-core GEMM (`nvjet_sm90_tst_..._TNT`) | 43,200 | 9.06 s |
| cuDNN flash SDPA | 4,800 | 5.81 s |
| Triton atom local attention | 1,203 | 5.55 s |
| cublas TF32 GEMM | 21,005 | 3.34 s |
| BF16 tensor-core GEMM with bias | 24,000 | 2.94 s |
| FP32 elementwise multiply | 14,361 | 2.14 s |
| fast LayerNorm FP32 | 14,306 | 2.14 s |
| fast LayerNorm BF16 | 23,960 | 2.02 s |
| Triton SiLU-mul | 6,003 | 1.55 s |
| FP32 elementwise add | 5,362 | 1.35 s |
| Triton sigmoid-mul | 10,803 | 1.21 s |
| Triton sigmoid-mul-add | 9,600 | 1.15 s |

Runtime launch/copy families:
| family | calls | runtime/API time |
| --- | ---: | ---: |
| `cudaLaunchKernel` | 383,312 | 11.83 s |
| `cuLaunchKernelEx` | 123,106 | 10.30 s |
| `cudaMemsetAsync` | 108,389 | 10.11 s |
| `cudaMemcpyAsync` | 53,231 | 0.38 s |

Interpretation:
- The remaining cost is no longer one obvious unfused PyTorch op. It is a mix of
  vendor tensor-core GEMMs, cuDNN SDPA, custom local attention, fast LayerNorm,
  and residual elementwise/copy traffic.
- Replacing cuDNN SDPA with a broad Triton attention kernel was already shown to
  lose. Future wins need to preserve strong vendor kernels and remove adjacent
  traffic, or improve the already-owned local-attention kernel.

Decision:
- Use this as the current bottleneck map. Do not optimize from code aesthetics
  alone.

### Round 15 - rejected diffusion transformer pair-z cache

Status: rejected and reverted

Hypothesis:
- The diffusion trunk receives the same pair embedding `z_pair` on every
  denoising step. Precomputing the efficient-fusion transformer layout
  (`normalize -> permute -> contiguous`) once per sampled structure might remove
  repeated dtype/layout work without replacing cuDNN SDPA or tensor-core GEMMs.

Change:
- Added an opt-in `PROTENIX_CACHE_DIFFUSION_TRANSFORMER_Z=1` path that prepared
  the transformer `z` tensor once after the shared diffusion pair cache and
  reused it inside `DiffusionModule.f_forward`.

Experiment:
- Paired one-H100 gate on `gpu-8`, commit `f5ba324`, `N_sample=160`,
  `N_step=200`, true unchunked diffusion, accepted Round 11 flags, one warmup
  item and one timed 7r6r item per variant.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round14_cache_transformer_z_gate_20260630_223013`.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | diffusion transformer | peak allocated MiB | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache off | 5.795 | 5.851 | 27.346 | 11.137 s | 9425 | baseline |
| cache on | 5.742 | 5.797 | 27.599 | 11.136 s | 9440 | -0.9% aggregate |

Interpretation:
- The target timer did not move: `diffusion_transformer_sec` was effectively
  identical. The extra cached tensor slightly increased peak memory and full
  forward time.
- The repeated pair-z layout work is not currently a material bottleneck in the
  real workload, despite looking plausible from code inspection and trace
  counts.

Decision:
- Reject and revert. Do not carry this default-off cache path.

Next:
- Use the full trace as the map: remaining material work is mostly tensor-core
  GEMMs, cuDNN SDPA, the already-custom atom local attention, fast LayerNorm,
  and residual elementwise/copy traffic. Future candidates need to move one of
  those measured families directly.

### Round 16 - BF16 atom encoder/decoder with Triton local attention

Status: promoted as an opt-in throughput flag

Hypothesis:
- Earlier broad BF16 atom attention was slower before the custom local-attention
  path existed. After promoting the Triton atom local-attention kernel, running
  the atom encoder/decoder under BF16 autocast should reduce adjacent linear,
  normalization, and elementwise traffic while preserving the already-guarded
  local-attention kernel behavior.

Change:
- Added `PROTENIX_BF16_ATOM_ATTENTION=1` to wrap the diffusion atom
  encoder/decoder calls in CUDA BF16 autocast. The flag is off by default and
  is only promoted as part of the explicit throughput stack above.

Experiment:
- Paired one-H100 gates on `gpu-8`, commit `28e65fd`, true unchunked diffusion,
  one warmup 7r6r item and one timed 7r6r item per variant.
- Shared flags: `PROTENIX_GPU_RANDOM_AUGMENT=1`,
  `PROTENIX_BF16_DIFFUSION_CORE=1`, `PROTENIX_ATTENTION_FORCE_FP32=0`,
  `PROTENIX_TRITON_LOCAL_ATTN=1`,
  `PROTENIX_TRITON_LOCAL_ATTN_NUM_WARPS=4`,
  `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`, `PROTENIX_TIMING_DIFFUSION=1`.
- N160 run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round15_bf16_atom_attention_gate_20260630_224053`.
- N320 run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round15_bf16_atom_attention_n320_gate_20260630_224507`.

Results:
| shape | BF16 atom flag | aggregate samples/sec | forward samples/sec | forward sec | atom encoder | atom decoder | transformer | peak allocated MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N160 | 0 | 5.821 | 5.881 | 27.208 | 4.538 s | 4.078 s | 11.105 s | 9425 | yes |
| N160 | 1 | 6.123 | 6.186 | 25.867 | 4.237 s | 3.048 s | 11.100 s | 9429 | yes |
| N320 | 0 | 6.266 | 6.299 | 50.803 | 8.860 s | 8.022 s | 21.444 s | 16545 | yes |
| N320 | 1 | 6.642 | 6.679 | 47.914 | 8.258 s | 5.965 s | 21.436 s | 16549 | yes |

Interpretation:
- The win is real and localized. The diffusion transformer time is unchanged,
  while atom decoder time falls substantially and atom encoder time improves
  modestly.
- Peak memory is effectively unchanged. The candidate remains finite on both
  gate shapes.
- New best per-GPU throughput is `6.642` aggregate samples/sec, `12.58x` over
  the original default per-GPU aggregate baseline.

Decision:
- Promote `PROTENIX_BF16_ATOM_ATTENTION=1` into the explicit high-throughput
  stack.

Next:
- Re-profile the new best stack. The likely next legitimate candidates are
  narrower than whole-model mega-kernels: improve the owned local-attention
  kernel and its projection/bias boundary, remove residual FP32 LayerNorm and
  elementwise traffic where numerically safe, or find GEMM epilogue fusions that
  preserve vendor tensor-core kernels.

### Round 17 - current-best profile after BF16 atom attention

Status: complete

Experiment:
- Full PyTorch/CUPTI trace of the current promoted stack after Round 16,
  commit `f3f5405`, one Sandpit Tokyo H100.
- Shape: `N_sample=160`, `N_step=200`, true unchunked diffusion, full promoted
  stack including `PROTENIX_BF16_ATOM_ATTENTION=1`.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round17_profile_bf16_atom_best_20260630_225215`.
- Note: CUPTI inflated the profiled forward wall time to `464.4 s`; throughput
  from this run is invalid. Kernel rankings and launch counts are the useful
  signal.

Top kernel families:
| family | calls | CUDA time |
| --- | ---: | ---: |
| BF16 tensor-core GEMM (`nvjet_sm90_tst_..._TNT`) | 43,200 | 3.49 s |
| cuDNN flash SDPA | 4,800 | 2.09 s |
| Triton atom local attention | 1,203 | 1.53 s |
| BF16 tensor-core GEMM with bias | 24,000 | 1.16 s |
| fast LayerNorm BF16 | 25,560 | 0.94 s |
| Triton SiLU-mul | 6,003 | 0.82 s |
| Triton sigmoid-mul-add | 9,600 | 0.79 s |
| BF16 elementwise add | 14,602 | 0.72 s |
| BF16 copy/cast kernel | 61,546 | 0.71 s |
| fast LayerNorm FP32 | 12,706 | 0.63 s |

Runtime launch/copy families:
| family | calls | runtime/API time |
| --- | ---: | ---: |
| `cudaLaunchKernel` | 404,061 | 4.31 s |
| `cuLaunchKernelEx` | 142,506 | 3.18 s |
| `cudaMemsetAsync` | 102,789 | 1.84 s |
| `cudaMemcpyAsync` | 53,231 | 0.34 s |

Interpretation:
- BF16 atom attention substantially reduced the local-attention side of the
  profile. The remaining cost is now spread across tensor-core GEMMs, cuDNN
  SDPA, local attention, LayerNorm, and residual elementwise/copy traffic.
- The trace still showed `aten::conv2d`/BF16 GEMM work for trunk pair-bias
  projection. Because the diffusion pair embedding and block weights are fixed
  across all denoising steps, a per-block pair-bias cache was a plausible next
  candidate.

Decision:
- Use this as the post-Round-16 bottleneck map. Do not replace cuDNN SDPA with
  a broad Triton attention kernel; that was already slower in Round 13.

### Round 18 - rejected diffusion transformer pair-bias cache

Status: rejected and reverted

Hypothesis:
- The previous pair-z layout cache only removed normalization/layout work. A
  more ambitious cache would precompute the 24 block-specific projected trunk
  pair biases (`[1, n_heads, N_token, N_token]`) once per model forward and
  reuse them across all 200 denoising steps, removing repeated `conv2d`/GEMM
  launches without replacing cuDNN SDPA.

Change:
- Added an opt-in `PROTENIX_CACHE_DIFFUSION_TRANSFORMER_PAIR_BIAS=1` path that
  threaded precomputed pair biases through `DiffusionTransformerBlock` and
  `AttentionPairBias`.
- First implementation keyed the cache after `DiffusionConditioning.forward`.
  In the real inference path `inplace_safe=True` clones the shared pair tensor,
  so the cache missed every denoising step. A follow-up keyed the cache on the
  original shared `pair_z` argument before that clone.

Experiment:
- Broken-cache gate: one-H100 N160 run on `gpu-8`, commit `650bd90`,
  run directory
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round18_pair_bias_cache_gate_n160_20260630_231026`.
  This run was also thermally anomalous (`gpu-8` reported 77-84 C before
  execution) and both variants were much slower than the current baseline, so
  it was not used for the decision.
- Fixed-cache gate: one-H100 N160 run on `gpu-14`, commit `6c8d755`,
  true unchunked diffusion, current promoted stack, one warmup 7r6r item and
  one timed 7r6r item per variant.
- Fixed-cache run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round19_pair_bias_cache_fixed_gate_n160_20260630_231659`.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | diffusion transformer | peak allocated MiB | peak reserved MiB | finite coords | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pair-bias cache off | 6.038 | 6.099 | 26.236 | 11.190 s | 9429 | 14198 | yes | baseline |
| pair-bias cache on | 6.072 | 6.137 | 26.072 | 11.071 s | 9429 | 14278 | yes | +0.6% aggregate |

Interpretation:
- The fixed cache did engage and moved the intended transformer timer by about
  `0.12 s`, but the representative workload win was below the promotion
  threshold and smaller than normal single-run cluster noise.
- The implementation adds cross-module plumbing and a persistent tensor cache
  for only a sub-1% N160 gain, with a small reserved-memory increase. At N320
  the same absolute saved work would be a smaller relative gain because the
  projected pair bias is broadcast over samples.

Decision:
- Reject and revert both pair-bias cache commits. Do not carry this default-off
  complexity.

Next:
- The remaining credible targets need to move larger measured families: reduce
  the number of linear/GEMM launches without losing tensor-core efficiency, or
  target LayerNorm/elementwise/copy traffic inside the diffusion transformer and
  atom encoder/decoder where a full-block replacement is not required.

### Round 19 - rejected fused self-QKV projection

Status: rejected and reverted

Hypothesis:
- The current-best trace is dominated by many linear/GEMM launches. For
  standard self-attention where `q_x` and `kv_x` are the same tensor, computing
  Q/K/V with one concatenated `F.linear` could reduce launches and improve GEMM
  shape efficiency while leaving cuDNN SDPA unchanged.

Change:
- Added a guarded inference-only `PROTENIX_FUSED_SELF_QKV=1` path in
  `Attention._prep_qkv`. It only activated for CUDA self-attention with matching
  q/k/v input dimensions and no k/v bias. Concatenated weights/biases were
  cached by parameter pointer/version.

Hotspot screen:
- Hardware: one Sandpit Tokyo H100 (`gpu-14`), commit `5e9ecba`.
- Shape: diffusion trunk attention block,
  `scripts/perf/trunk_attention_hotspot.py --samples 160 --z-samples 1
  --tokens 245 --dtype bfloat16 --warmup 10 --iters 60
  --efficient-fusion 1`.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round20_fused_self_qkv_hotspot_20260630_232415`.

Results:
| run | full block | attention qkv/sdpa/gate/out | pair bias projection | transition | peak allocated MiB | finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fused self-QKV off | 2.424 ms | 1.002 ms | 0.033 ms | 0.915 ms | 1086 | yes |
| fused self-QKV on | 2.426 ms | 1.029 ms | 0.033 ms | 0.911 ms | 1156 | yes |

Interpretation:
- The candidate directly worsened the targeted attention subrange and increased
  memory. A larger single GEMM did not beat the existing separate projection
  path for this H100 shape.

Decision:
- Reject after hotspot; do not spend a full inference gate. Revert the
  implementation.

Next:
- Do not pursue naive projection concatenation. Future GEMM-side work needs a
  more precise boundary, such as preserving vendor GEMMs while fusing epilogues,
  or a shape-specific kernel that demonstrates a hotspot win before touching
  the full inference path.

### Round 20 - rejected BF16 confidence head

Status: rejected and reverted

Hypothesis:
- At `N_sample=320`, confidence scoring is a material part of forward time.
  The base-model inference config explicitly disables AMP for the confidence
  head. Allowing the confidence head to run under the outer BF16 autocast might
  reduce tensor-core and copy/cast cost without skipping confidence outputs.

Change:
- Added opt-in `PROTENIX_BF16_CONFIDENCE=1`, overriding
  `configs.skip_amp.confidence_head` only for `run_confidence_head`.

Experiment:
- Paired one-H100 N160 gate on `gpu-14`, commit `e649b11`, current promoted
  stack, one warmup 7r6r item and one timed 7r6r item per variant.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round21_bf16_confidence_gate_n160_20260630_232642`.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | confidence | diffusion | peak allocated MiB | finite coords | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BF16 confidence off | 6.081 | 6.144 | 26.041 | 2.721 s | 18.926 s | 9429 | yes | baseline |
| BF16 confidence on | 5.961 | 6.020 | 26.577 | 3.067 s | 18.952 s | 9405 | yes | -2.0% aggregate |

Interpretation:
- The candidate made the confidence head slower despite slightly lower peak
  allocated memory. This is likely a case where the confidence kernels or
  numerics benefit from the existing FP32 policy.

Decision:
- Reject and revert. Keep the base confidence head AMP-disabled.

Next:
- Treat confidence as a semantic/output-level tradeoff, not a dtype free win.
  Skipping or deferring confidence may be useful for design-only coordinate
  generation, but it changes the inference product and should not be counted as
  a default model throughput optimization without an explicit user decision.

### Round 21 - N_sample saturation and the N1280 memory cliff

Status: complete

Hypothesis:
- The memory headroom at `N_sample=320` might allow substantially larger sample
  batches, further amortizing pairformer/feature overhead and increasing
  throughput.

Experiment:
- One-H100 saturation sweep on commit `fef90a4`, current promoted stack after
  Round 20, true unchunked diffusion, one warmup 7r6r item and one timed 7r6r
  item per shape.
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round22_nsample_saturation_20260630_233243`.

Results:
| N_sample | aggregate samples/sec | forward samples/sec | forward sec | confidence | diffusion | transformer | peak allocated MiB | peak reserved MiB | notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 320 | 6.515 | 6.551 | 48.850 | 5.682 s | 36.737 s | 21.714 s | 16549 | 26242 | finite |
| 640 | 6.813 | 6.832 | 93.675 | 11.030 s | 72.958 s | 43.437 s | 30788 | 50032 | finite |
| 960 | 6.863 | 6.876 | 139.608 | 16.659 s | 109.145 s | 65.146 s | 45027 | 73820 | finite; best throughput but little memory margin |
| 1280 | failed | failed | - | - | - | - | - | - | OOM in confidence `pde_preds = torch.stack(...)` |

Interpretation:
- Larger batches help only weakly once `N_sample >= 640`. The fixed pairformer
  and feature-prep costs are already amortized; diffusion and confidence scale
  almost linearly with sample count.
- `N_sample=1280` does not fail in the diffusion coordinate generator. It fails
  when the full-output confidence path materializes large FP32 PAE/PDE logits:
  `N_sample x N_token x N_token x 64` for each pair head. The failed allocation
  was `18.32 GiB` with the process already using `65.71 GiB`.
- The practical high-throughput setting is therefore not "as large as memory
  permits". It is `N_sample=640` as a safe default, or `N_sample=960` when
  accepting a narrow H100 80 GB memory margin for a small extra throughput gain.

Decision:
- Update the recommended stack to `N_sample=640` safe, with `N_sample=960` as
  the current best measured single-run throughput.
- Do not promote `N_sample=1280` without changing the confidence-output memory
  behavior.

### Round 22 - Nsight Compute roofline screen

Status: complete

Hypothesis:
- If the remaining kernels are near peak tensor compute, then further large
  "mega-kernel" rewrites are unlikely to help. If they are memory/locality or
  latency limited, the next work should reduce traffic, launches, or improve
  tiling rather than simply increase batch size.

Setup:
- Installed Nsight Compute in a separate micromamba environment:
  `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/nsight-compute/bin/ncu`.
- Version: `2026.1.1.x` from `cuda-nsight-compute=12.4.1`.
- Profiler commands were launched inside Slurm GPU allocations on compute
  nodes. The login node does not need to have Nsight Compute installed and
  should not be treated as a source of GPU-kernel truth.
- Reports:
  - Local attention:
    `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round23_ncu_roofline_local_attention_20260630_235252/local_attention.ncu-rep`
  - Trunk SDPA/GEMM:
    `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round24_ncu_roofline_trunk_kernels_20260630_235337/`
  - LayerNorm retry:
    `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round25b_ncu_layernorm_roofline_20260630_235720/bf16_layernorm.ncu-rep`

Results:
| kernel | representative shape | duration | SM throughput | memory throughput | DRAM/L2/L1 signal | interpretation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Triton atom local attention | `N_sample=640`, 2529 atoms, BF16, 32x128 local windows | 5.10 ms | 32.2% | 91.5% | DRAM 21.7%, L1/TEX 91.5%, L2 47.5% | on-chip/L1 traffic limited, not tensor-compute limited |
| cuDNN trunk flash SDPA | 640 x 245 tokens x 16 heads x head_dim 48 | 1.66 ms | 51.5% | 37.8% | DRAM 17.2%, L2 37.8% | not peak compute or HBM; likely latency/scheduler/shape efficiency |
| BF16 tensor-core GEMM with bias | trunk GEMM family, `nvjet_sm90_tst_..._bias_TNN` | 163 us | 59.3% | 62.5% | DRAM 62.5%, L2 75.5%, 2.09 TB/s | compute/memory balanced; reduce both work and traffic |
| BF16 fast LayerNorm | `LayerNormForwardV2<BFloat16, float4>` | 194 us | 54.4% | 73.5% | 2.46 TB/s, L2 73.0%, L1/TEX 29.0% | memory-bandwidth heavy; fusion/recompute can help more than more batching |

Interpretation:
- We are not hitting maximum tensor compute on the major remaining kernels.
  The strongest owned kernel, atom local attention, is already near the L1/TEX
  roof and only at about one third SM throughput. Bigger sample batches do not
  fix that; better tiling/reuse or less bias/value traffic is required.
- cuDNN SDPA is not saturating compute either, but the previous full-token
  Triton replacement lost to cuDNN. The path forward is not a broad SDPA
  replacement unless it preserves vendor-level tensor-core efficiency.
- LayerNorm and elementwise/copy families remain plausible fusion targets
  because their roofline position is memory/traffic dominated.

Decision:
- Use the roofline as a constraint on further "mega-kernel" work:
  - local atom attention: optimize memory locality/reuse first;
  - trunk attention: avoid replacing cuDNN SDPA wholesale;
  - GEMM-heavy paths: only pursue epilogue-style fusions that preserve strong
    tensor-core GEMMs;
  - LayerNorm/elementwise: narrow fusion remains credible.

### Round 23 - opt-in confidence pair-logit CPU offload

Status: capacity mode kept; rejected as a throughput setting

Hypothesis:
- `N_sample=1280` fails because confidence materializes full PAE/PDE logits on
  GPU. Offloading those pair logits to CPU as they are produced should make the
  4x larger batch fit, proving whether batch size alone can improve throughput.

Change:
- Added `PROTENIX_CONFIDENCE_CPU_OFFLOAD=1`, extending Protenix's existing
  large-token PAE/PDE CPU-offload behavior to high sample-count inference.
- Added optional
  `PROTENIX_CONFIDENCE_CPU_OFFLOAD_THRESHOLD_GIB=<float>` for size-triggered
  offload.
- Avoided `torch.cuda.empty_cache()` per sample for high-sample offload; the old
  cache purge remains only for the original `N_token > 2000` path or when
  `PROTENIX_CONFIDENCE_CPU_OFFLOAD_EMPTY_CACHE=1` is set.

Experiments:
- First N1280 offload run on `gpu-8` was cancelled after warmup because the
  node was hot (`81-84 C`) and the initial implementation purged the CUDA
  allocator cache per sample. Warmup throughput was only `2.98` samples/sec.
- Corrected run on cool `gpu-14`, commit `c927824`, one warmup item and one
  timed item, run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round27_confidence_offload_n1280_noempty_20260701_001110`.

Results:
| run | aggregate samples/sec | forward samples/sec | forward sec | confidence | diffusion | transformer | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N960, normal confidence | 6.863 | 6.876 | 139.608 | 16.659 s | 109.145 s | 65.146 s | 45027 | 73820 | yes |
| N1280, confidence CPU offload | 6.472 | 6.480 | 197.536 | 30.864 s | 145.531 s | 87.314 s | 18073 | 21598 | yes |

Interpretation:
- The 4x-from-N320 batch can be made to fit; peak GPU allocation falls to
  `18.1 GiB` because the giant PAE/PDE logits are no longer stacked on GPU.
- It is not a throughput improvement. Diffusion scales almost linearly with the
  extra samples and confidence becomes slower from CPU/GPU pair-logit traffic.
- This confirms that higher batches do not help because the workload has little
  fixed overhead left to amortize. The remaining bottlenecks are per-sample
  diffusion transformer, atom attention, confidence pairformer, LayerNorm, and
  elementwise/memory traffic.

Decision:
- Keep the offload path as an explicit capacity mode for users who need very
  large `N_sample` runs to fit.
- Do not include `PROTENIX_CONFIDENCE_CPU_OFFLOAD=1` or `N_sample=1280` in the
  promoted throughput stack.

Next:
- For real throughput, do not spend more time increasing `N_sample` alone.
  Work on the linear-scaling kernels: local-attention memory traffic,
  LayerNorm/elementwise fusion, and confidence-summary/logit streaming that
  computes only needed design scores without full pair-logit materialization
  when that semantic tradeoff is acceptable.

### Round 24 - BF16 local-attention warp retune

Status: promoted

Hypothesis:
- The Nsight Compute local-attention report showed the owned Triton atom
  local-attention kernel is L1/TEX/on-chip-memory limited, not tensor-compute
  limited. For the fixed 32 x 128 local window, the previous four-warp launch
  shape likely creates unnecessary scheduler/on-chip pressure. Fewer warps
  should reduce that pressure without changing memory footprint or semantics.

Change:
- Changed the default Triton local-attention launch shape to one warp for BF16
  and two warps for FP32.
- Kept `PROTENIX_TRITON_LOCAL_ATTN_NUM_WARPS={1,2,4,8}` as an explicit
  override for future profiling or non-H100 shapes.

Isolated screen:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round28_local_attention_parameter_screen_20260701_002259`.
- Workload: one-H100 local-attention benchmark, 7r6r atom count, BF16/FP32,
  `N_sample={320,640,960}`, `input_precision={tf32,tf32x3}`,
  `num_warps={1,2,4,8}`.

Results:
| dtype | N_sample | best setting | best ms | previous four-warp ms | delta |
| --- | ---: | --- | ---: | ---: | ---: |
| BF16 | 320 | `tf32`, 1 warp | 2.160 | 2.501 | -13.6% |
| BF16 | 640 | `tf32`, 1 warp | 4.268 | 4.991 | -14.5% |
| BF16 | 960 | `tf32`, 1 warp | 6.382 | 7.469 | -14.5% |
| FP32 | 320 | `tf32`, 2 warps | 2.654 | 3.673 | -27.7% |
| FP32 | 640 | `tf32`, 2 warps | 5.281 | 7.327 | -27.9% |
| FP32 | 960 | `tf32`, 2 warps | 7.906 | 10.987 | -28.0% |

Representative full inference gates:
- Shared workload: `N_sample=640`, BF16 diffusion core, BF16 atom attention,
  true unchunked diffusion, one warmup 7r6r item and one timed 7r6r item per
  variant, dump disabled.
- Round 29c run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round29c_local_attention_warp_gate_canary_20260701_003853`.
- Round 29d reversed-order repeat:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round29d_local_attention_warp_gate_reverse_20260701_004812`.

| gate | variant | aggregate samples/sec | forward samples/sec | forward sec | atom encoder | atom decoder | diffusion | confidence | peak allocated MiB | finite coords |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 29c | 4 warps old | 6.849 | 6.869 | 93.179 | 16.263 s | 11.892 s | 72.521 s | 10.900 s | 30788 | yes |
| 29c | 1 warp candidate | 6.867 | 6.887 | 92.935 | 15.858 s | 11.484 s | 71.717 s | 11.231 s | 30788 | yes |
| 29d | 1 warp candidate | 6.890 | 6.910 | 92.625 | 15.859 s | 11.481 s | 71.721 s | 10.931 s | 30788 | yes |
| 29d | 4 warps old | 6.844 | 6.864 | 93.235 | 16.270 s | 11.908 s | 72.605 s | 10.826 s | 30788 | yes |

Interpretation:
- Averaged over both gate orders, BF16 one-warp local attention measured
  `6.879` aggregate samples/sec versus `6.847` for the previous four-warp
  setting, a `+0.47%` full-workload gain with unchanged memory.
- The isolated hotspot win is much larger than the full-workload win because
  local attention is now one component inside the longer diffusion/confidence
  path. The component breakdown still moves in the expected direction: atom
  encoder plus decoder time falls from about `28.17 s` to `27.34 s`.
- The result is small, so it is not a new headline speedup, but it passed the
  required reverse-order repeat and has a clear roofline-backed mechanism.

Decision:
- Promote the BF16 one-warp default. Continue to compare future candidates
  against this current stack.

Next:
- The local-attention roofline and warp screen both point at memory locality and
  bias/value traffic rather than more batching. The next kernel-level work
  should attack the actual local-attention data movement, or fuse adjacent
  memory-bound LayerNorm/elementwise paths, not just adjust launch parameters.

### Round 25 - fused local-attention pair-bias projection

Status: promoted

Hypothesis:
- In atom local attention, `AttentionPairBias.local_multihead_attention`
  materializes the pair-bias producer as
  `LayerNorm(c_atompair=16) -> Linear(16 -> 4 heads) -> permute`, then the
  Triton local-attention kernel immediately rereads the bias. At `N_sample=640`
  this is a large linear-scaling memory path in every atom-transformer block.
- A fused producer can write the final
  `[..., n_heads, n_trunks, 32, 128]` bias layout directly and remove the
  normalized-pair intermediate while preserving the existing local-attention
  consumer kernel.

Change:
- Added `protenix/model/modules/local_attention_bias_triton.py`.
- Added `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS`. If unset, it now follows
  `PROTENIX_TRITON_LOCAL_ATTN`, so the fused bias producer is part of the
  guarded Triton local-attention fast path and can still be explicitly disabled.
- Wired the fused producer into `AttentionPairBias.local_multihead_attention`
  with exact-shape guards and fallback to the existing PyTorch path.
- Added `scripts/perf/local_attention_bias_hotspot.py` to compare the bias
  producer alone and the full local-attention block.

Implementation note:
- The first naive Triton producer read each 16-channel pair vector once for
  LayerNorm statistics and then again once per head. It was rejected by the
  hotspot screen (`15.56 ms` reference full block vs `19.70 ms` candidate at
  N320) and exposed an int32 address overflow at N640.
- The promoted kernel reads `z` once and accumulates all four head projections
  during that pass:
  `projected_h = (sum(z * ln_weight * linear_weight_h) - mean * sum(ln_weight * linear_weight_h)) * rstd`.
  Address arithmetic was widened to int64 for the multi-GiB N640 `z` tensor.

Hotspot screen:
- Rejected naive run:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round30c_fused_local_bias_hotspot_20260701_010439`.
- Promoted optimized run:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round31_fused_local_bias_hotspot_opt_20260701_010622`.

| shape | metric | reference | optimized fused producer | speedup | parity |
| --- | --- | ---: | ---: | ---: | --- |
| N320 BF16 | bias projection | 11.727 ms | 3.862 ms | 3.04x | max bias diff `0.03125` |
| N320 BF16 | full local-attention block | 15.626 ms | 6.473 ms | 2.41x | max output diff `0.0078125` |
| N640 BF16 | bias projection | 23.193 ms | 7.716 ms | 3.01x | max bias diff `0.03125` |
| N640 BF16 | full local-attention block | 30.811 ms | 12.865 ms | 2.39x | max output diff `0.009765625` |

Representative full inference gate:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round32_fused_local_bias_full_gate_20260701_010815`.
- Commit: `aa0131e`.
- Workload: `N_sample=640`, BF16 diffusion core, BF16 atom attention, true
  unchunked diffusion, one warmup 7r6r item and one timed 7r6r item per
  variant, dump disabled, one H100 on `gpu-canary-0`.

| variant | aggregate samples/sec | forward samples/sec | forward sec | diffusion | atom encoder | atom decoder | transformer | confidence | peak allocated MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fused bias off | 6.923 | 6.943 | 92.185 | 71.728 s | 15.865 s | 11.475 s | 43.066 s | 10.808 s | 30788 | yes |
| fused bias on | 7.162 | 7.183 | 89.103 | 68.742 s | 14.263 s | 9.877 s | 43.277 s | 10.774 s | 30788 | yes |

Interpretation:
- The full workload improved `+3.45%` aggregate throughput with unchanged peak
  allocation/reservation and finite coordinates.
- The movement is in the intended component: atom encoder plus decoder time
  falls from `27.34 s` to `24.14 s`, while trunk transformer and confidence are
  effectively unchanged.
- This is a real fusion win rather than config tuning: it removes a
  memory-heavy local-attention bias producer intermediate and preserves the
  existing attention kernel.

Decision:
- Promote fused local-attention pair-bias projection as part of the current
  Triton local-attention throughput stack.

Next:
- Profile the fused-bias stack to see whether atom local attention is now
  dominated by the softmax/value kernel, q/k/v projections, or adjacent
  adaptive LayerNorm/gating. The next candidate should be another producer/
  consumer fusion in the atom transformer or a trunk transition epilogue fusion
  that preserves cuDNN SDPA and tensor-core GEMMs.

### Round 26 - rejected fused self-attention QKV projection

Status: rejected and reverted

Hypothesis:
- The trunk transformer hotspot after the fused-bias promotion showed the
  `attention_qkv_sdpa_gate_out` subrange dominates a diffusion trunk block:
  at N640 it measured `9.76 ms` of a `16.27 ms` block, while conditioned
  transition was `4.15 ms` and pair-bias projection was only `0.036 ms`.
- Because trunk self-attention uses the same normalized `a` tensor for q/k/v,
  concatenating q/k/v weights into one larger projection might remove two GEMM
  launches and improve GEMM efficiency while preserving cuDNN SDPA.

Change:
- Added an opt-in `PROTENIX_FUSED_SELF_QKV=1` path in `Attention._prep_qkv`
  in commit `661441a`, with cached concatenated q/k/v weights and fallback to
  the existing three-projection path.

Experiment:
- Hotspot run:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round35_fused_self_qkv_hotspot_20260701_012856`.
- Workload: diffusion trunk/token `DiffusionTransformerBlock`, BF16, 245
  tokens, 16 heads, efficient conv2d pair-bias path, one H100.

Results:
| shape | fused qkv | full block ms | attention subrange ms | peak allocated MiB | parity |
| --- | ---: | ---: | ---: | ---: | --- |
| N320 | 0 | 8.294 | 4.925 | 2518 | reference |
| N320 | 1 | 8.384 | 5.066 | 2647 | finite |
| N640 | 0 | 16.267 | 9.736 | 4931 | reference |
| N640 | 1 | 16.490 | 9.979 | 5174 | max diff `0.0` on N64 check |

Interpretation:
- The fused projection was numerically exact in the parity check but slower
  and used more memory. On H100/PyTorch, three separate BF16 projection GEMMs
  for this shape are already efficient enough; the larger concatenated GEMM and
  cached fused weights do not pay for themselves.

Decision:
- Rejected. Reverted the code in `1ac182a`; do not carry the default-off path.

Next:
- Trunk attention remains material, but QKV concatenation is not the right
  mechanism. Focus on either conditioned-transition epilogue/activation traffic
  or atom-transformer post-fused-bias profiling.

### Round 27 - post-fusion launch attribution with NCU

Status: diagnostic

Question:
- After promoting fused atom local-attention bias projection and rejecting
  trunk QKV concatenation, which exact launches still dominate the atom and
  trunk transformer paths? This was run with Nsight Compute installed in the
  cluster micromamba NCU environment:
  `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/nsight-compute/bin/ncu`.

Experiments:
- Atom transformer hotspot:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round36_atom_transformer_hotspot_20260701_013323`.
- NCU atom attention/transition reports:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round39_atom_ncu_20260701_013753`.
- Trunk block hotspot:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round34_trunk_block_hotspot_20260701_012229`.
- Commit for NCU capture harness: `ceb6cd1`.

Atom transformer hotspot, `N_sample=640`, BF16, one H100:

| subrange | ms |
| --- | ---: |
| full atom `DiffusionTransformerBlock` | 23.239 |
| `attention_pair_bias_local_attention` | 17.665 |
| `conditioned_transition` | 4.854 |

NCU launch attribution for one warmed atom attention call at `N_sample=640`:

| launch family | time | share | NCU indication |
| --- | ---: | ---: | --- |
| `_project_local_attention_bias_kernel` | 7.772 ms | 44.0% | memory-pipeline dominated; `gpu__compute_memory_throughput` 96.9% |
| `_local_attention_kernel` | 1.763 ms | 10.0% | memory/softmax/value path; DRAM 62.8%, compute-memory 66.2% |
| q/k/v/out GEMMs, grouped | 3.221 ms | 18.3% | cuBLAS/nvjet tensor-core GEMMs; already reasonably efficient |
| four `LayerNormForwardV2` launches | 1.327 ms | 7.5% | memory-bound LN |
| sigmoid/gating/other elementwise | 3.586 ms | 20.3% | mostly memory-bound elementwise |

NCU launch attribution for one warmed conditioned-transition call at
`N_sample=640`:

| launch family | time | share | NCU indication |
| --- | ---: | ---: | --- |
| transition GEMMs, grouped | 2.347 ms | 49.2% | tensor-core GEMMs; not a good hand-fusion target |
| `_silu_mul_kernel` | 0.806 ms | 16.9% | memory-bound fused activation/mul |
| `_sigmoid_mul_add_kernel` | 0.528 ms | 11.1% | memory-bound gate/add |
| `_sigmoid_mul_kernel` | 0.400 ms | 8.4% | memory-bound gate |
| two `LayerNormForwardV2` launches | 0.661 ms | 13.9% | memory-bound LN |

Trunk block hotspot, `N_sample=640`, BF16, 245 tokens, 16 heads:

| subrange | ms |
| --- | ---: |
| full trunk block | 16.268 |
| `attention_pair_bias` total | 11.015 |
| adaptive LN | 1.290 |
| pair-bias projection | 0.036 |
| `attention_qkv_sdpa_gate_out` | 9.759 |
| conditioned transition | 4.147 |

Interpretation:
- The exact top atom launch after Round 25 is still the local pair-bias
  producer, not the Triton local-attention softmax itself. It is a
  linear-scaling, memory-dominated path. Bigger `N_sample` therefore mostly
  makes this launch proportionally longer; there is little fixed overhead left
  to amortize.
- The trunk pair-bias projection is no longer material in the trunk hotspot.
  The efficient `conv2d` projection path leaves trunk attention dominated by
  q/k/v + cuDNN SDPA + output/gating, and the separate fused-QKV screen showed
  that naive GEMM concatenation regresses.
- A full "mega-kernel" that fuses atom bias projection with local attention is
  only attractive if it reuses the 16-channel `z` vector across all four heads.
  A per-head fused kernel would reread `z` four times and likely lose to the
  current producer/consumer split. A true all-head local-attention mega-kernel
  is possible but substantially more complex because it must avoid holding four
  full 32 x 128 score matrices in registers at once.

Decision:
- Use the NCU result to screen a narrower producer-kernel improvement first.
  Do not replace cuDNN SDPA or tensor-core GEMMs wholesale without an isolated
  screen.

### Round 28 - rejected preweighted/tuned local-bias producer

Status: rejected and reverted

Hypothesis:
- The NCU top launch was `_project_local_attention_bias_kernel`.
  Precomputing the tiny `layernorm_weight * linear_weight` vectors once per
  parameter version and retuning the Triton program width/warps might reduce
  scalar weight-load and scheduler pressure in that launch.

Change:
- Added an opt-in preweighted local-bias projection kernel and temporary
  screening tunables in commits `181e453`, `ec82b8d`, and `58d7590`.
- The candidate setting was
  `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_PREWEIGHT=1`,
  `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_BLOCK_SIZE=256`,
  `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_NUM_WARPS=8`.

Isolated screens:
- Preweight screen:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round40_preweighted_bias_hotspot_20260701_014311`.
- Block/warp micro-sweep:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round42_bias_kernel_sweep_20260701_014604`.
- Best hotspot:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round43_bias_best_hotspots_20260701_014708`.

Hotspot results:

| shape | metric | current promoted | best candidate | delta | parity |
| --- | --- | ---: | ---: | ---: | --- |
| N640 | bias producer only | 7.774 ms | 5.034 ms | -35.2% | max bias diff `0.015625` |
| N640 | full local-attention block | 12.799 ms | 10.312 ms | -19.4% | max output diff `0.0078125` |
| N640 | full atom transformer block | 23.239 ms | 20.747 ms | -10.7% | finite |

Representative full gates:
- Four single-item paired gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round44_tuned_bias_full_gate_20260701_014912`.
- Lower-noise repeat-3 gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round45_tuned_bias_repeat3_gate_20260701_020655`.

| gate | variant | aggregate samples/sec | forward samples/sec | forward sec | peak allocated MiB | finite coords |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| round44 mean of two | old producer | 7.108 | 7.129 | 89.778 | 30788 | yes |
| round44 mean of two | tuned producer | 7.176 | 7.197 | 88.928 | 30788 | yes |
| round45 repeat-3 | old producer | 7.180 | 7.215 | 266.118 total | 30788 | yes |
| round45 repeat-3 | tuned producer | 7.164 | 7.199 | 266.691 total | 30788 | yes |

Interpretation:
- The isolated kernel and block wins are real, but the stronger full-workload
  repeat rejected the default: tuned was `-0.22%` aggregate and `-0.21%`
  mean forward throughput versus the current promoted producer.
- The reason is scope. The atom local-attention block is a small part of the
  full `N_sample=640` inference path; the repeat-3 component timings show
  diffusion atom encoder/decoder and trunk transformer times were essentially
  unchanged. The initial single-item gain was dominated by confidence/pairformer
  noise, not the intended atom-local component.

Decision:
- Rejected. Reverted the preweighted kernel, temporary tunables, and default
  change in `bdf617c`, `2698b70`, and `f45189b`.
- Keep the Round 25 fused local-bias producer as the promoted implementation.

Next:
- Do not spend more time shaving the atom local-bias producer unless the target
  workload changes to one where atom transformer blocks dominate. The next
  full-workload bottlenecks are the diffusion trunk transformer, confidence
  summary/pairformer work, and broad memory-bound LN/gating paths.

### Round 29 - trunk attention NCU launch attribution

Status: diagnostic

Question:
- The trunk block hotspot showed `attention_qkv_sdpa_gate_out` at `9.759 ms`
  for `N_sample=640`, but fused self-QKV regressed. Which launches inside that
  subrange actually cost GPU time?

Change:
- Added an NCU capture mode to `scripts/perf/trunk_attention_hotspot.py` in
  commit `a0d1c9e`.

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round46_trunk_ncu_20260701_022956`.
- Workload: `N_sample=640`, 245 tokens, 16 heads, head dim 48, BF16, efficient
  pair-bias projection enabled, one H100.

NCU launch attribution for one warmed trunk attention call:

| launch family | GPU kernel time | share of profiled GPU kernels | NCU indication |
| --- | ---: | ---: | --- |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | 2.040 ms | 54.2% | fused attention kernel, 54.7% SM, 76.9% tensor pipe |
| q/k/v/out tensor-core GEMMs, grouped | 1.305 ms | 34.7% | 75-80% tensor pipe; not an obvious hand-GEMM target |
| `_sigmoid_mul_kernel` gate | 0.232 ms | 6.2% | memory-bound gate, 90.9% compute-memory throughput |
| final multiply | 0.157 ms | 4.2% | memory-bound elementwise |
| small elementwise/copies | 0.030 ms | 0.8% | not material individually |

NCU launch attribution for one warmed trunk conditioned-transition call:

| launch family | GPU kernel time | share of profiled GPU kernels | NCU indication |
| --- | ---: | ---: | --- |
| transition tensor-core GEMMs, grouped | 1.965 ms | 59.3% | 59-88% tensor pipe depending GEMM |
| `_silu_mul_kernel` | 0.468 ms | 14.1% | memory-bound |
| `_sigmoid_mul_add_kernel` | 0.307 ms | 9.3% | memory-bound |
| `_sigmoid_mul_kernel` | 0.231 ms | 7.0% | memory-bound |
| two `LayerNormForwardV2` launches | 0.309 ms | 9.3% | memory-bound |

Interpretation:
- This is the strongest evidence so far that the trunk attention wall time is
  not explained by one bad GPU kernel. NCU sees only `3.764 ms` of GPU kernels
  inside the same attention target that the CUDA-event hotspot measured near
  `9.76 ms`. The missing time is consistent with CPU/framework/launch gaps or
  library orchestration around many small launches and SDPA setup.
- A "mega-kernel" for trunk attention would need to replace an already-good
  cuDNN/CUTLASS FMHA kernel plus efficient GEMMs. That is a high-complexity
  path with limited roofline upside unless it also removes orchestration and
  adjacent pointwise/LN traffic.
- The more credible high-leverage mechanisms are:
  1. CUDA graph capture/replay or compile-style capture for the static-shape
     diffusion trunk loop to remove launch/orchestration gaps.
  2. Fusing the cheap memory-bound gate/multiply epilogue around existing GEMMs
     only if the fusion does not pessimize cuBLAS/cuDNN kernels.
  3. Algorithmic confidence-output work, because confidence still takes about
     10-11 s in the full gate and may compute more pair logits than design
     ranking actually needs.

Decision:
- Do not attempt a hand-written replacement for cuDNN SDPA as the next step.
  The next ambitious candidate should target launch/orchestration overhead in
  the diffusion trunk loop, with CUDA graphs or `torch.compile`/Inductor as the
  smallest decisive screen before considering custom mega-kernels.

### Round 30 - trunk capture and roofline diagnostics

Status: diagnostic; capture rejected

Question:
- Is the remaining trunk/atom transformer time launch/orchestration limited,
  or are the important launches already near a relevant H100 roofline?

Experiments:
- `torch.compile` screen:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round47_trunk_compile_screen_20260701_023557`.
- CUDA Graph screen with fast LayerNorm:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round48_trunk_cudagraph_screen_20260701_023940`.
- CUDA Graph screen with `LAYERNORM_TYPE=torch` fallback and NCU roofline
  reports from Nsight Compute `2026.1.1`:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round49_graph_ln_fallback_roofline_20260701_024352`.

Capture results at `N_sample=640`, BF16, one H100:

| target | eager ms | candidate ms | speedup | parity / status |
| --- | ---: | ---: | ---: | --- |
| `torch.compile` attention | 4.741 | 5.070 | 0.935x | finite; slower |
| `torch.compile` transition | 3.355 | 3.771 | 0.890x | finite; slower |
| `torch.compile` block | 9.121 | 10.048 | 0.908x | finite; slower |
| `torch.compile` stack4 | 36.614 | 39.669 | 0.923x | finite; slower |
| CUDA Graph, fast LN, attention | 4.717 | 4.352 | 1.084x | invalid; graph output all NaN |
| CUDA Graph, fast LN, transition | 3.336 | 2.922 | 1.142x | invalid; graph output contains NaNs |
| CUDA Graph, torch LN, attention | 4.952 | 4.965 | 0.997x | exact parity; finite |
| CUDA Graph, torch LN, transition | 3.592 | 3.580 | 1.003x | exact parity; finite |
| CUDA Graph, torch LN, block | 9.589 | 9.579 | 1.001x | exact parity; finite |
| CUDA Graph, torch LN, stack4 | 38.496 | 38.277 | 1.006x | exact parity; finite |

Interpretation:
- `torch.compile(mode="reduce-overhead")` is rejected in this environment. It
  graph-breaks on the custom fast LayerNorm extension and is slower on every
  static-shape trunk target.
- CUDA Graph replay with the default fast LayerNorm is not numerically valid.
  The PyTorch/OpenFold LayerNorm fallback makes graph replay finite, but then
  the speedup is effectively zero. Capture is therefore not a legitimate
  throughput path unless the fast LayerNorm extension itself is made graph-safe
  and the full model gate proves a real win.

NCU SpeedOfLight/roofline launch-family summary, `N_sample=640`:

| target | launch family | time | share | roofline indication |
| --- | --- | ---: | ---: | --- |
| atom attention | local bias projection Triton | 7.720 ms | 43.9% | memory-pipeline saturated: memory 96.9%, L1/TEX 97.0%, DRAM 32.4%, SM 35.2% |
| atom attention | tensor-core GEMMs | 3.225 ms | 18.3% | memory-limited small GEMMs: memory/DRAM 73.4%, SM 16.9% |
| atom attention | local attention Triton | 1.750 ms | 9.9% | mixed memory/softmax/value path: memory 66.1%, DRAM 63.2%, SM 44.8% |
| atom attention | LN/gates/elementwise/copies | 4.908 ms | 27.9% | mostly memory-bound, many already fused pointwise kernels |
| atom transition | tensor-core GEMMs | 2.351 ms | 49.3% | memory-heavy GEMMs: memory/DRAM 76.2%, SM 21.0% |
| atom transition | SiLU/sigmoid/gates/LN | 2.397 ms | 50.2% | memory-bound epilogues/LN, 74-93% memory throughput |
| trunk attention | cuDNN/CUTLASS FMHA | 2.040 ms | 54.0% | mixed compute/cache path: memory 77.0%, DRAM 14.1%, SM 54.7% |
| trunk attention | tensor-core GEMMs | 1.322 ms | 35.0% | reasonably hot tensor path: memory 72.3%, SM 78.8% |
| trunk attention | gates/multiply/copies | 0.417 ms | 11.0% | memory-bound but small |
| trunk transition | tensor-core GEMMs | 1.998 ms | 59.6% | reasonably hot tensor path: memory 72.9%, SM 78.0% |
| trunk transition | SiLU/sigmoid/gates/LN | 1.324 ms | 39.5% | memory-bound, 65-93% memory throughput |

Decision:
- Higher `N_sample` does not help much beyond the promoted point because the
  important kernels scale linearly and are already memory-pipeline, DRAM, or
  tensor-pipe limited. There is little fixed launch overhead left to amortize
  in the finite/correct path.
- The next kernel work should not be a broad trunk SDPA replacement. The
  highest isolated atom launch is the pair-bias producer, but any fusion must
  reuse the 16-channel `z` vector across all four heads; a per-head fusion is
  expected to reread too much memory.

### Round 31 - rejected true fused atom bias-attention kernel

Status: rejected and reverted

Hypothesis:
- A more ambitious "mega-kernel" could remove the 7.7 ms atom local pair-bias
  producer and the 1.75 ms local-attention consumer boundary by computing
  `LayerNorm(z) @ W_head` inside the local-attention score kernel.

Change:
- Added an opt-in `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS_ATTN=1` prototype in
  commits `9e34672`, `2513d5c`, and `92bfc9f`.
- The first version worked at `N_sample=64` but hit illegal memory access at
  `N_sample=640`; the cause was 32-bit pointer offset overflow for the large
  `z` tensor. Commit `92bfc9f` fixed this by using int64 Triton offsets.

Experiments:
- Initial warp sweep:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round50_fused_bias_attention_screen_20260701_025320`.
- Int64 confirmation:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round51_fused_bias_attention_int64_20260701_025515`.

Results:

| shape | split fused-bias + local attention | true fused bias+attention | speed vs split | parity |
| --- | ---: | ---: | ---: | --- |
| N64, best 8-warps | 1.509 ms | 6.774 ms | 0.223x | max diff `0.00390625`, finite |
| N640, 8-warps | 14.073 ms | 64.743 ms | 0.217x | max diff `0.0078125`, finite |

Interpretation:
- The candidate failed for the predicted reason: the current producer computes
  all four heads in one pass over the 16-channel pair vector, while the
  per-head fused attention prototype rereads `z` once per head and keeps a
  large 32x128 bias/score tile live. Removing a launch and a materialized bias
  tensor did not compensate for the extra memory traffic and register pressure.
- This is an example where "mega-kernel" fusion is the wrong boundary. A
  credible all-head fused atom kernel would need a different design that reuses
  pair-vector statistics across heads without holding four full score matrices
  live in registers.

Decision:
- Rejected. Reverted the prototype and hotspot extension in `d1467e0`,
  `a1be288`, and `845d304`.
- Do not retry per-head fused atom bias-attention. If revisiting atom local
  attention, write a design note for an all-head/staged producer-consumer
  approach before coding.

### Round 32 - rejected confidence sample chunking

Status: rejected and reverted

Hypothesis:
- The confidence head loops over `N_sample` one structure at a time. Processing
  small sample chunks together might amortize kernel launches and raise
  occupancy in the confidence pairformer path without changing confidence
  semantics.

Change:
- Added opt-in `PROTENIX_CONFIDENCE_SAMPLE_CHUNK_SIZE` in commit `73c4e9c`.
  The default path still used the original one-sample loop.

Experiment:
- Failed first attempt without `PROTENIX_ROOT_DIR`, cancelled:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round52_confidence_chunk_screen_20260701_025909`.
- Corrected paired screen, job `92743`:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round53_confidence_chunk_screen_20260701_030333`.
- Workload: one H100, `N_sample=640`, `N_step=20`, `N_cycle=10`, unchunked
  diffusion, promoted kernel flags, `PROTENIX_ROOT_DIR=/mnt/lustre/users/kiarash-eitgbi/protenix_data`.

Results:

| confidence sample chunk | forward sec | forward samples/sec | pairformer | diffusion | confidence | peak allocated MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.116 | 14.507 | 11.850 s | 12.281 s | 12.399 s | 30410 |
| 4 | 41.838 | 15.297 | 9.847 s | 12.171 s | 12.543 s | 30410 |
| 8 | 41.722 | 15.340 | 9.830 s | 12.178 s | 12.513 s | 30410 |
| 16 | 41.990 | 15.242 | 9.895 s | 12.176 s | 12.617 s | 30410 |
| 32 | 41.341 | 15.481 | 9.682 s | 12.120 s | 12.327 s | 30410 |
| 64 | 41.585 | 15.390 | 9.831 s | 12.166 s | 12.424 s | 30410 |

Interpretation:
- The intended metric did not move: `model_log.time.confidence` stayed
  `12.3-12.6 s` for all chunk sizes, and peak memory was unchanged. The
  apparent small forward-time improvement came from pairformer/diffusion
  variation, not from a confidence-kernel speedup.
- This suggests the existing confidence/pairformer kernels do not gain useful
  occupancy or launch amortization from adding a leading sample batch in this
  path, or the dominant confidence cost is elsewhere in the output/summary
  pipeline.

Decision:
- Rejected. Reverted the opt-in chunking path in `d5a6146`.
- Do not promote `PROTENIX_CONFIDENCE_SAMPLE_CHUNK_SIZE`.

Next:
- Profile the post-confidence summary/output work explicitly. The remaining
  full forward wall time still contains several seconds outside the recorded
  `model_forward` component, so the next useful measurement is whether
  confidence summary/logit processing is a GPU, CPU, or synchronization
  bottleneck before writing another kernel.

### Round 33 - promoted confidence-summary sample chunking

Status: promoted

Question:
- The previous model timing ended `model_forward` before
  `compute_full_data_and_summary`. Is the post-confidence confidence-summary
  path material, and can it be batched safely over samples?

Change:
- Added optional synchronized phase timing with `PROTENIX_SYNC_TIMINGS=1` in
  commit `8cbadc9`.
- Added opt-in `PROTENIX_SUMMARY_SAMPLE_CHUNK_SIZE` in commit `2ceea90`.
  The default value preserves the historical one-sample wrapper; the promoted
  throughput stack sets it to `32`.

Diagnostics:
- Nsight Compute `2026.1.1` is installed in the cluster environment at
  `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/nsight-compute/bin/ncu`;
  as expected, it is used from Slurm compute-node jobs, not the login node.
- Synchronized N640/N20 timing gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round54_sync_summary_timing_20260701_031749`.
  The timed row showed `summary_confidence=7.25 s` out of `28.01 s` forward,
  so summary processing was about 26% of the short-forward wall time.
- Summary chunk screen:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round56_summary_chunk_fastscreen_20260701_032534`.
- Full N640/N200 promotion gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round57_summary_chunk_full_gate_20260701_034709`.
- Synthetic remote parity check passed for chunk 4 versus chunk 1 across all
  summary and full-data keys.

Summary chunk screen, `N_sample=640`, `N_step=20`, synchronized timings:

| summary chunk | forward sec | forward samples/sec | summary confidence | peak allocated MiB | finite coords |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 37.613 | 17.015 | 7.098 s | 30788 | yes |
| 4 | 33.697 | 18.993 | 2.531 s | 30788 | yes |
| 8 | 33.242 | 19.252 | 1.804 s | 30788 | yes |
| 16 | 31.924 | 20.047 | 1.278 s | 30788 | yes |
| 32 | 31.755 | 20.154 | 1.072 s | 30788 | yes |
| 64 | 32.478 | 19.706 | 1.018 s | 30788 | yes |
| 128 | 31.869 | 20.083 | 0.949 s | 30788 | yes |
| all | 31.972 | 20.018 | 0.931 s | 47426 | yes |

Full gate, `N_sample=640`, `N_step=200`, normal async runtime timing:

| summary chunk | forward sec | forward samples/sec | end-to-end samples/sec | summary confidence | peak allocated MiB | finite coords |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 90.826 | 7.046 | 7.026 | 7.010 s | 30788 | yes |
| 32 | 85.251 | 7.507 | 7.484 | 1.039 s | 30788 | yes |

Interpretation:
- The bottleneck was not the confidence head pairformer itself; it was the
  post-confidence summary/logit processing wrapper enumerating samples one by
  one. The inner `_compute_full_data_and_summary` already supports a leading
  sample batch for the tensor-heavy probability and chain-score reductions.
- Chunk 32 removes about six seconds from the full N640/N200 workload without
  increasing peak allocated memory. Larger chunks shave a little more summary
  time in the short screen but do not improve total forward time and the
  all-sample chunk increases peak allocated memory by about 16.6 GiB.

Decision:
- Promote `PROTENIX_SUMMARY_SAMPLE_CHUNK_SIZE=32` in the high-throughput stack.
- Keep the code path opt-in rather than changing upstream default behavior,
  because very large token counts or sample-specific contact-prob tensors may
  need the conservative one-sample wrapper.

Next:
- Re-run the high-memory N960 point with the promoted summary chunk, because
  summary chunking keeps peak allocation unchanged at N640 and may improve the
  previous N960 result without approaching the full-output memory cliff.

### Round 34 - high-memory N960 check after summary chunking

Status: diagnostic; max-throughput point updated

Question:
- Once post-confidence summary processing is batched, does increasing
  `N_sample` beyond the safe N640 point finally improve per-GPU throughput?

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round58_n960_summary_chunk32_gate_20260701_040416`.
- Workload: one H100, `N_sample=960`, `N_step=200`, `N_cycle=10`, unchunked
  diffusion, promoted kernel flags, `PROTENIX_SUMMARY_SAMPLE_CHUNK_SIZE=32`.

Result:

| N_sample | forward sec | forward samples/sec | end-to-end samples/sec | summary confidence | peak allocated MiB | peak reserved MiB | finite coords |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 960 | 126.631 | 7.581 | 7.565 | 1.590 s | 45027 | 73558 | yes |

Interpretation:
- Higher batch does help once the summary wrapper is fixed, but only modestly:
  N960 is about `+1.1%` end-to-end throughput versus the N640 chunk-32 full
  gate (`7.565` vs `7.484` samples/sec).
- The cost is memory headroom. Peak allocated rises from `30.8 GiB` to
  `45.0 GiB`, and peak reserved reaches `73.6 GiB` on an 80 GB H100. This is
  usable for max-throughput single-job runs but fragile for additional memory
  pressure, different inputs, larger token counts, or allocator fragmentation.

Decision:
- Keep `N_sample=640` as the safe high-throughput default in the run block.
- Record `N_sample=960` as the current max-throughput/high-memory point when
  the workload can tolerate running close to the 80 GB memory cliff.

Next:
- Do not retry normal N1280 without first changing confidence-logit
  materialization; the known failure mode is the full FP32 PAE/PDE stack, not
  the now-fixed summary wrapper.

### Round 35 - streamed confidence logits unlock higher batches

Status: promoted behind an explicit inference flag

Hypothesis:
- The remaining memory cliff is not diffusion; it is the full FP32
  `pae`/`pde` logit stack materialized by `ConfidenceHead.forward` before
  summary/full-data processing. If logits are generated in sample chunks and
  immediately consumed by `compute_full_data_and_summary`, N1280 should run and
  the confidence-summary path should retain exact output parity for dumped
  inference fields.

Change:
- Added `ConfidenceHead.iter_logit_chunks(...)` and
  `Protenix.run_confidence_summary_stream(...)` in commit `075d860`.
- Enabled by `PROTENIX_STREAM_CONFIDENCE_SUMMARY=1` for inference-only calls
  with no labels or symmetric permutation. The raw `plddt`/`pae`/`pde`/`resolved`
  logits are intentionally omitted in this mode; inference dumping uses
  `coordinate`, `contact_probs`, `summary_confidence`, and `full_data`.
- `PROTENIX_CONFIDENCE_LOGIT_CHUNK_SIZE` controls the logit chunk; the promoted
  stack uses `32`, matching the existing summary chunk size.

Correctness:
- Direct same-tensor parity passed in
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round62_stream_confidence_direct_parity3_20260701_042929`.
  The test compared normal confidence logits plus summary against streamed
  confidence summary/full-data for the same `s_inputs`, `s`, `z`, coordinates,
  and contact probabilities across all summary and full-data keys.
- On the direct N32 screen, the streamed confidence-summary block reduced time
  from `0.774 s` to `0.609 s` and peak allocated memory from `3344 MiB` to
  `2739 MiB`.

Full 7r6r gate, `N_step=200`, one H100, normal async throughput timing:

| variant | forward sec | forward samples/sec | end-to-end samples/sec | confidence head | summary confidence | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N640, no stream | 84.924 | 7.536 | 7.513 | 10.792 s | 1.025 s | 30788 | 49858 | yes |
| N640, stream | 84.482 | 7.576 | 7.552 | 10.316 s | 1.045 s | 10190 | 10700 | yes |
| N960, stream | 125.859 | 7.628 | 7.611 | 16.117 s | 1.611 s | 14133 | 15620 | yes |
| N1280, stream | 165.455 | 7.736 | 7.723 | 20.758 s | 2.102 s | 18076 | 21598 | yes |

Interpretation:
- Streaming is only a small throughput win at N640 (`+0.5%`) because the
  normal path already uses chunked summary and most time is diffusion. It is a
  large memory win: peak reserved drops from `49.9 GiB` to `10.7 GiB`.
- The previous N960 memory cliff (`73.6 GiB` reserved) was confidence-logit
  materialization, not true model activation pressure. With streaming, N960
  reserves only `15.6 GiB`, and N1280 runs at `21.6 GiB` reserved.
- Increasing batch size now helps, but only modestly, because the per-sample
  diffusion and confidence costs are already close to linear.

Decision:
- Promote `PROTENIX_STREAM_CONFIDENCE_SUMMARY=1` and
  `PROTENIX_CONFIDENCE_LOGIT_CHUNK_SIZE=32` in the high-throughput stack.
- Update the conservative high-throughput point to N1280 and keep testing
  higher batches until the marginal gain disappears.

### Round 36 - current N1280 Nsight Compute roofline

Status: diagnostic

Question:
- Are the remaining hot kernels actually hitting H100 peak compute, or are
  higher batches limited by memory traffic, layout/copy traffic, and occupancy?

Experiment:
- NCU `2026.1.1` was run from a Slurm compute-node job, not the login node:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round65_current_ncu_roofline_n1280_20260701_050930`.
- Command shape: isolated hotspot scripts at `samples=1280`, current commit
  `075d860`, promoted streaming and Triton flags, with `--section SpeedOfLight`,
  `--section SpeedOfLight_RooflineChart`, and
  `--section SpeedOfLight_HierarchicalTensorRooflineChart`.

Aggregated roofline counters:

| hotspot | main family | report time | SM % | memory % | DRAM % | L2 % | L1 % | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| atom attention | local bias projection | 15.420 ms | 35.2 | 97.0 | 32.5 | 34.4 | 97.0 | L1/shared-memory or on-chip memory pipe bound, not tensor compute bound |
| atom attention | local attention kernel | 3.482 ms | 45.1 | 66.4 | 63.7 | 75.8 | 66.4 | mixed memory/occupancy bound; high registers (`244/thread`) |
| atom attention | GEMMs | 6.447 ms | 16.9 | 73.9 | 73.9 | 70.1 | 21.5 | memory dominated despite tensor ops |
| atom transition | GEMMs | 4.703 ms | 20.9 | 76.5 | 76.5 | 71.0 | 25.1 | memory dominated, low tensor utilization |
| trunk attention | SDPA | 4.035 ms | 54.8 | 77.1 | 14.2 | 34.1 | 77.2 | not peak compute; mostly on-chip traffic/attention pipeline |
| trunk attention | GEMMs | 2.593 ms | 80.4 | 74.0 | 54.6 | 67.3 | 75.0 | the closest to compute-saturated, but still memory-active |
| trunk transition | GEMMs | 3.961 ms | 79.4 | 74.9 | 48.5 | 65.3 | 73.9 | near tensor-pipe saturation, not enough alone to dominate full runtime |
| trunk transition | elementwise/copy | 2.053 ms | 28.7 | 90.8 | 90.7 | 81.1 | 14.9 | DRAM bandwidth dominated |

Interpretation:
- The model is not globally hitting maximum H100 compute. Some trunk GEMMs are
  healthy (`~79-80%` SM/tensor utilization), but atom attention/transition and
  elementwise/copy paths are memory-pipe limited.
- The worst atom-attention launch remains the fused pair-bias projection
  (`_project_local_attention_bias_kernel`): it removed Python/PyTorch overhead,
  but now sits at `~97%` memory throughput and only `~35%` SM throughput. A
  naive mega-kernel that rereads pair features per head is expected to lose,
  matching the rejected true fused bias-attention prototype.
- Higher batches do not buy much more once fixed pairformer/launch overhead is
  amortized because the dominant diffusion kernels scale linearly and are
  already memory-pipeline or tensor-pipe limited.

Decision:
- Do not keep increasing batch size expecting large gains. It is useful only
  until fixed overhead is amortized.
- The next legitimate kernel work must reduce memory traffic or improve
  arithmetic intensity in atom local attention/transition and elementwise
  paths; broad whole-block "mega kernels" are too likely to increase traffic
  unless they are designed around all-head reuse of pair features.

### Round 37 - high-batch saturation after streaming

Status: diagnostic; max-throughput point updated

Question:
- After removing confidence-logit materialization, does 4x larger batch improve
  throughput materially, or are we at the linear-scaling asymptote?

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round66_stream_confidence_high_batch_20260701_051340`.
- Workload: one H100, 7r6r input only, `N_step=200`, `N_cycle=10`, unchunked
  diffusion, promoted kernel flags, streamed confidence summary.

Results:

| N_sample | forward sec | forward samples/sec | end-to-end samples/sec | diffusion | confidence head | summary confidence | peak allocated MiB | peak reserved MiB | finite coords |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1280 | 165.455 | 7.736 | 7.723 | 140.007 s | 20.758 s | 2.102 s | 18076 | 21598 | yes |
| 1920 | 247.380 | 7.761 | 7.753 | 210.518 s | 30.940 s | 3.340 s | 25961 | 29658 | yes |
| 2560 | 329.254 | 7.775 | 7.769 | 281.547 s | 40.797 s | 4.344 s | 33846 | 36398 | yes |

Interpretation:
- N2560 is the best measured per-GPU throughput so far: `7.769` end-to-end
  samples/sec, `14.71x` over the original default per-GPU baseline.
- The marginal return is now tiny: N1920 is only `+0.38%` over N1280, and N2560
  is only `+0.21%` over N1920 (`+0.59%` over N1280). This matches the roofline
  picture: fixed pairformer overhead is already amortized, and diffusion plus
  confidence scale nearly linearly with `N_sample`.
- Memory is no longer the limiter at these sizes for the 245-token 7r6r input:
  N2560 reserves `36.4 GiB`. For larger token/atom counts, keep N1280 as the
  conservative high-throughput default until separately gated.

Decision:
- Record N2560 as the current max-throughput point for this input.
- Keep N1280 in the promoted run block as the conservative default because it
  gives `99.4%` of max throughput with much lower memory and shorter individual
  job runtime.
- Further batch-only sweeps are not a legitimate path to a large win. The next
  work should be kernel redesign around atom local-attention/transition memory
  traffic and elementwise/copy fusion.

### Round 38 - BF16 local-attention output store

Status: promoted behind an explicit inference flag

Hypothesis:
- In the BF16 atom-attention path, the Triton local-attention kernel still
  accumulates in FP32 but stores a FP32 output tensor. `Attention.forward`
  already allocates that tensor with transposed Q strides, so the following
  `transpose(-2, -3)` is contiguous and is not the main copy bottleneck. The
  avoidable traffic is the FP32 attention output feeding the BF16 gate/output
  projection path.

Change:
- Added `PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16=1` in commit `c9f9584`.
- The flag only affects the Triton local-attention path when inputs are BF16.
  Scores and attention-value accumulation remain FP32 inside the kernel; only
  the output store dtype changes from FP32 to BF16. The default remains FP32 to
  preserve upstream semantics unless the throughput stack opts in.

Local correctness and layout diagnostics:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round67_bf16_local_attention_output_20260701_054331`.
- Direct local attention, `N_sample=1280`, BF16 inputs:
  - default output dtype: `torch.float32`
  - candidate output dtype: `torch.bfloat16`
  - both use output stride `[323712, 32, 128, 1]`; after the existing
    transpose, the post-transpose view is contiguous with stride
    `[323712, 128, 32, 1]`
  - candidate-vs-default local-attention error: max `0.0156`, mean `0.00081`
- Full `attention_pair_bias` same-block/same-input check at `N_sample=640`:
  output dtype stayed BF16 and candidate-vs-default output was bit-identical
  (`max_abs_error=0.0`, all finite). Peak allocated memory in that direct check
  dropped from `22694 MiB` to `21903 MiB`.

Hotspot screen, one atom transformer block:

| samples | output store | attention ms | transition ms | full block ms | peak allocated MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 320 | FP32 | 8.923 | 2.462 | 11.733 | 7591 |
| 320 | BF16 | 8.345 | 2.461 | 11.181 | 6998 |
| 640 | FP32 | 17.629 | 4.855 | 23.202 | 15147 |
| 640 | BF16 | 16.431 | 4.850 | 21.946 | 13962 |
| 1280 | FP32 | 35.078 | 9.638 | 46.040 | 30260 |
| 1280 | BF16 | 32.627 | 9.616 | 43.600 | 27889 |

Full 7r6r gate, `N_step=200`, one H100, normal async throughput timing:

| variant | forward sec | forward samples/sec | end-to-end samples/sec | diffusion | confidence head | summary confidence | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N1280, FP32 local-attn output | 167.077 | 7.661 | 7.649 | 140.656 s | 21.571 s | 2.147 s | 18076 | 21598 | yes |
| N1280, BF16 local-attn output | 163.327 | 7.837 | 7.824 | 137.897 s | 20.733 s | 2.080 s | 15705 | 18438 | yes |
| N2560, BF16 local-attn output | 325.095 | 7.875 | 7.868 | 275.907 s | 42.099 s | 4.451 s | 29104 | 34818 | yes |

Targeted NCU after the BF16-output change:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round69_bf16out_ncu_atom_20260701_061813`.
- Commit profiled: `c9f9584` (before the later final-attention-gate fusion
  experiment).
- NCU version: `2026.1.1`, run inside a Slurm GPU job.

Atom-attention target at `N_sample=1280`:

| family | FP32 output time | BF16 output time | BF16 SM % | BF16 memory % | BF16 DRAM % | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| local bias projection | 15.537 ms | 15.537 ms | 35.2 | 97.0 | 32.2 | unchanged; still the largest memory-pipe launch |
| attention/linear GEMMs | 6.453 ms | 6.448 ms | 17.0 | 73.9 | 73.9 | unchanged; tensor utilization remains low/memory-active |
| PyTorch elementwise/copy | 5.031 ms | 1.931 ms | 23.7 | 89.1 | 88.8 | main removed traffic from matching BF16 gate dtype |
| Triton local attention | 3.488 ms | 3.393 ms | 45.5 | 66.3 | 58.1 | small store-bandwidth win, still occupancy/memory bound |
| LayerNorm | 2.636 ms | 2.634 ms | 84.9 | 75.5 | 75.5 | unchanged |
| AdaLN sigmoid-mul-add | 2.116 ms | 2.116 ms | 25.3 | 93.2 | 93.2 | unchanged, DRAM-bound elementwise |
| Triton attention gate | not used | 0.806 ms | 30.0 | 91.5 | 91.5 | replaces fallback sigmoid/mul elementwise traffic |

Current atom-transition target at `N_sample=1280`, BF16 local-attention output:

| family | time | SM % | memory % | DRAM % | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| transition GEMMs | 4.699 ms | 21.1 | 76.5 | 76.5 | memory-active low tensor utilization |
| Triton SiLU-mul | 1.616 ms | 31.5 | 91.5 | 91.5 | DRAM-bound elementwise |
| LayerNorm | 1.318 ms | 84.9 | 75.4 | 75.4 | high SM but still memory-active |
| AdaLN sigmoid-mul-add | 1.057 ms | 25.4 | 93.3 | 93.3 | DRAM-bound elementwise |
| final transition gate | 0.806 ms | 29.8 | 91.5 | 91.5 | DRAM-bound elementwise |

Interpretation:
- The paired N1280 gate shows a real but bounded win: `+2.3%` end-to-end
  throughput, `-13.1%` peak allocated memory, and `-14.6%` peak reserved
  memory.
- The N2560 candidate is the best measured per-GPU point so far:
  `7.868` end-to-end samples/sec, `14.90x` over the original default per-GPU
  baseline. The marginal gain of N2560 over N1280 remains only `+0.56%`, so
  N1280 remains the conservative stack default.
- This is the kind of fusion/detail work that the roofline supported: reduce
  memory traffic on the repeated atom-attention path without re-reading pair
  features per head or making a giant kernel with worse locality.
- The NCU launch-level mechanism is specific: BF16 output does not speed up the
  bias projection or GEMMs, but it enables the existing Triton BF16 attention
  gate and removes about `3.1 ms` of fallback PyTorch elementwise/copy time from
  the isolated atom-attention target.

Decision:
- Promote `PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16=1` in the high-throughput
  stack.
- Keep the implementation opt-in until more input sizes are gated; the current
  validation is strong for the 7r6r-style benchmark but not yet a broad
  upstream default policy.
- Continue with current-candidate NCU and then target atom transition and
  elementwise/copy traffic, where the previous roofline still showed high DRAM
  pressure.

### Round 39 - rejected final attention-gate fusion

Status: rejected and reverted

Hypothesis:
- `AttentionPairBias.forward` used a plain
  `torch.sigmoid(self.linear_a_last(s)) * a` for the final adaLN-Zero attention
  gate, while the inner attention gate and transition gate already used the
  Triton `fused_sigmoid_mul` helper. Replacing this non-inplace branch with
  `fused_sigmoid_mul` might remove one repeated elementwise launch/memory pass.

Change:
- Commit `97fd021` replaced the non-inplace final gate with
  `fused_sigmoid_mul(gate, a)`.
- The change was reverted in commit `ac1a0cd` after the full gate failed.

Hotspot:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round70_fused_final_attention_gate_20260701_062207`.

| samples | revision | attention ms | transition ms | full block ms | finite |
| ---: | --- | ---: | ---: | ---: | --- |
| 320 | previous | 8.333 | 2.458 | 11.162 | yes |
| 320 | fused gate | 8.168 | 2.456 | 10.947 | yes |
| 640 | previous | 16.415 | 4.858 | 21.998 | yes |
| 640 | fused gate | 16.145 | 4.851 | 21.682 | yes |
| 1280 | previous | 32.599 | 9.630 | 43.695 | yes |
| 1280 | fused gate | 32.060 | 9.625 | 43.153 | yes |

Full N1280 gate:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round71_fused_final_gate_full_20260701_062328`.

| variant | forward sec | forward samples/sec | end-to-end samples/sec | diffusion | confidence head | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| previous | 164.692 | 7.772 | 7.759 | 138.334 s | 21.509 s | 15705 | 18438 | yes |
| fused gate | 164.781 | 7.768 | 7.755 | 138.362 s | 21.583 s | 15705 | 18438 | yes |

Interpretation:
- The isolated atom-block hotspot improved by about `1.2%`, but this did not
  survive a full inference gate. The paired N1280 full run was slightly slower
  (`7.755` vs `7.759` end-to-end samples/sec) with unchanged memory.
- This is a useful guardrail: not every elementwise fusion that wins in a block
  screen changes the whole-model critical path once variance, confidence, and
  pairformer timing are included.

Decision:
- Reject and revert the fused final attention gate.
- Keep looking at transition/elementwise traffic, but promote only candidates
  that pass the full N200 inference gate.

### Round 40 - fused transition residual add

Status: promoted behind an explicit inference flag

Hypothesis:
- Atom transition NCU showed DRAM-bound elementwise launches in the transition
  path: the final transition gate (`triton_sigmoid_mul`) and the following
  block residual add are separate tensor passes. Fusing the final transition
  gate with the residual add should reduce one repeated memory pass per atom
  transformer block.

Change:
- Added `PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL=1` in commit `24fa911`.
- The fast path is inference-only, opt-in, and only used when DropPath is
  identity. It changes:
  `fused_sigmoid_mul(gate, projected) + residual` into
  `fused_sigmoid_mul_add(gate, projected, residual)`.
- The default remains the original two-step expression.

Hotspot and parity:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round72_fused_transition_residual_hotspot_20260701_064101`.
- Same-block/same-input parity at `N_sample=320`: all finite, max abs error
  `0.03125`, mean abs error `7.4e-05`. The error is expected from fusing the
  multiply/add before the BF16 store.

| samples | residual fusion | attention ms | transition ms | full block ms | peak allocated MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 320 | off | 8.323 | 2.462 | 11.163 | 6998 |
| 320 | on | 8.314 | 2.458 | 10.943 | 6998 |
| 640 | off | 16.412 | 4.856 | 21.959 | 13962 |
| 640 | on | 16.400 | 4.853 | 21.670 | 13962 |
| 1280 | off | 32.509 | 9.623 | 43.679 | 27889 |
| 1280 | on | 32.590 | 9.622 | 43.141 | 27889 |

Full 7r6r gate, `N_step=200`, one H100:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round73_fused_transition_residual_full_20260701_064218`.

| variant | forward sec | forward samples/sec | end-to-end samples/sec | diffusion | confidence head | summary confidence | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N1280, residual fusion off | 165.808 | 7.720 | 7.707 | 138.052 s | 22.675 s | 2.216 s | 15705 | 18438 | yes |
| N1280, residual fusion on | 163.970 | 7.806 | 7.793 | 136.654 s | 22.239 s | 2.185 s | 15705 | 18438 | yes |
| N2560, residual fusion on | 324.176 | 7.897 | 7.890 | 273.848 s | 42.971 s | 4.589 s | 29104 | 35338 | yes |

Interpretation:
- The paired N1280 gate shows a `+1.12%` end-to-end throughput win with
  unchanged memory.
- The new max-throughput point is `N_sample=2560` with `7.890` end-to-end
  samples/sec, `7.897` forward samples/sec, and `35.3 GiB` peak reserved
  memory. This is `14.94x` over the original default per-GPU baseline.
- The win is modest, but unlike the rejected final attention-gate fusion it
  survives the full N200 model gate.

Decision:
- Promote `PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL=1` in the high-throughput
  stack.
- Continue targeting atom-transition and local-pair-bias memory traffic; batch
  scaling remains nearly saturated.

### Round 41 - rejected 2D local-pair bias projection

Status: rejected and reverted

Hypothesis:
- NCU showed `_project_local_attention_bias_kernel` as the largest remaining
  atom-attention launch in the synthetic atom-block target: about `15.5 ms`,
  `~97%` memory-pipe utilization, and only `~35%` SM utilization at
  `N_sample=1280`.
- The old vector-lane kernel maps one pair per lane and loops over the 16
  pair channels. A 2D pair-by-channel Triton tile should load the small
  `c_atompair=16` channel dimension more contiguously and reduce memory-pipe
  pressure.

Change:
- Commit `4c35071` added an opt-in
  `PROTENIX_TRITON_LOCAL_ATTN_BIAS_2D=1` path with
  `PROTENIX_TRITON_LOCAL_ATTN_BIAS_2D_BLOCK_PAIRS=128`.
- The path was guarded to `c_z == 16` and left the previous fused bias kernel
  as the default fallback.

Hotspot and profiler evidence:
- Local-bias run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round74_2d_local_bias_hotspot_20260701_071350`.
- Atom-block run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round75_2d_local_bias_atom_hotspot_20260701_071555`.
- NCU run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round77_2d_bias_ncu_atom_20260701_074901`.
- NCU was run inside a Slurm GPU job using
  `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/nsight-compute/bin/ncu`
  (`2026.1.1`), not on the login node.

Synthetic local-attention and atom-block screens were strongly positive:

| screen | samples | 2D off | 2D on, block 128 | delta |
| --- | ---: | ---: | ---: | ---: |
| bias projection only | 640 | 7.775 ms | 3.330 ms | `-57.2%` |
| local attention full path | 640 | 11.659 ms | 7.295 ms | `-37.4%` |
| atom block attention subrange | 1280 | 32.594 ms | 23.899 ms | `-26.7%` |
| full atom block | 1280 | 43.112 ms | 34.496 ms | `-20.0%` |

NCU confirmed the launch-level mechanism in the synthetic atom-attention
target:

| launch family | 2D off time | 2D on time | 2D on SM % | 2D on memory % | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| local bias projection | 15.537 ms | 6.608 ms | 67.7 | 75.7 | real kernel win; memory-pipe pressure reduced |
| attention/linear GEMMs | 6.445 ms | 6.453 ms | 16.8 | 73.9 | unchanged |
| Triton local attention | 3.395 ms | 3.396 ms | 45.5 | 66.3 | unchanged |
| LayerNorm | 2.639 ms | 2.624 ms | 84.8 | 75.8 | unchanged |
| elementwise/copy | 1.934 ms | 1.932 ms | 23.5 | 89.0 | unchanged |

Full inference gate:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round76_2d_local_bias_full_20260701_071852`.
- Workload: one H100, 7r6r input, `N_step=200`, unchunked diffusion, current
  promoted stack, paired `N1280` off/on plus candidate `N2560`.

| variant | forward sec | forward samples/sec | end-to-end samples/sec | diffusion | confidence head | peak allocated MiB | peak reserved MiB | finite coords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N1280, 2D off | 164.510 | 7.781 | 7.767 | 137.065 s | 22.359 s | 15705 | 18438 | yes |
| N1280, 2D on, block 128 | 163.693 | 7.820 | 7.806 | 136.957 s | 21.814 s | 15705 | 18438 | yes |
| N2560, 2D on, block 128 | 328.578 | 7.791 | 7.784 | 274.111 s | 46.646 s | 29104 | 35338 | yes |

Full-path instrumentation explained the mismatch:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round79_2d_bias_full_instrument_20260701_080509`.
- Real full inference, `N_step=5`, `N_sample=1280`, current stack, no giant
  profiler trace. The monkeypatch counted calls into
  `triton_project_local_attention_bias`.
- Both variants selected the fused bias path for all 33 calls. With 2D on, all
  33 calls were eligible for the 2D path.
- But the actual full-path shapes were not the synthetic hotspot shape:
  - 30 per-step calls used `z.shape == (1, 80, 32, 128, 16)`.
  - 3 cached calls used `z.shape == (80, 32, 128, 16)`.
  - No full-path call projected a `(1280, 80, 32, 128, 16)` pair-bias tensor.

Interpretation:
- The NCU/kernel win is real, but it optimizes a synthetic `N_sample`-expanded
  pair-bias projection that the real cached inference path mostly avoids.
- Protenix's diffusion shared-vars cache keeps the atom pair representation
  sample-broadcasted with leading sample dimension `1`; the expensive
  high-`N_sample` work is elsewhere. This is why the full `N1280` gate moved
  only `+0.5%` end-to-end and why most of that apparent delta came from
  pairformer/confidence noise rather than the diffusion timer.
- The `N2560` candidate was slower than the current promoted `N2560` point
  (`7.784` versus `7.890` end-to-end samples/sec), so there is no evidence for
  a max-throughput promotion.

Decision:
- Reject and revert the 2D local-bias path. It is a good synthetic-kernel
  improvement but not a legitimate full-workload throughput optimization.
- Future atom-local-attention screens must either use the real cached full-path
  shapes or include the call-shape instrumentation before a full gate.

Next:
- Do not spend more effort on the cached local pair-bias projection.
- Target the real full-path costs: diffusion token transformer attention/GEMMs,
  confidence-head/sample-summary cost, and atom-transformer work that is still
  sample-scaled rather than cached with sample dimension `1`.

### Round 42 - promoted-stack detailed full-path timing

Status: diagnostic

Question:
- After rejecting the synthetic 2D local-bias win, where is the real
  high-throughput full-model time spent under the current promoted stack?

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round80_promoted_detailed_timing_20260701_081505`.
- Commit: `ee55ece`.
- Workload: one H100, 7r6r input, `N_sample=1280`, `N_step=200`,
  unchunked diffusion, current promoted stack, `PROTENIX_TIMING_DIFFUSION=1`.

Result:

| component | time |
| --- | ---: |
| forward total | 164.639 s |
| end-to-end throughput | 7.761 samples/sec |
| pairformer | 2.731 s |
| diffusion total | 137.899 s |
| diffusion token transformer | 90.478 s |
| diffusion atom encoder | 26.755 s |
| diffusion atom decoder | 18.072 s |
| diffusion conditioning | 1.976 s |
| confidence head | 21.475 s |
| summary confidence | 2.145 s |
| peak reserved | 18.0 GiB |

Interpretation:
- The current dominant same-output bottleneck is the diffusion token
  transformer, not local pair-bias production. It is about `55%` of full
  forward time and `66%` of diffusion time.
- Atom encoder/decoder remain material (`44.8 s` combined), but the rejected
  2D-bias experiment showed that the cached pair-bias producer is not the
  sample-scaled part of that cost.
- Confidence head is still a large linear-in-sample cost (`21.5 s`), but
  earlier BF16 and sample-chunking attempts regressed or failed to move it.
  Skipping or deferring confidence would be an output/semantics tradeoff, not a
  same-output kernel optimization.

Decision:
- Use this as the current bottleneck map for the next round.
- Same-output kernel work should target the diffusion token transformer first:
  24 blocks per denoising step, 200 denoising steps, `N_sample x 245 tokens x
  16 heads x head_dim 48`, with already efficient cuDNN/SDPA and conv2d
  pair-bias projection.

### Round 43 - N1280 trunk-block NCU roofline refresh

Status: diagnostic

Question:
- At the current promoted `N_sample=1280` point, which exact diffusion token
  transformer launches dominate, and are they underutilized enough to justify a
  broad custom attention mega-kernel?

Experiments:
- CUDA-event trunk hotspot plus bounded NCU capture:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round81_n1280_trunk_roofline_20260701_082945`.
- Corrected NCU SpeedOfLight run using `--set basic`:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round83_n1280_trunk_basic_ncu_20260701_083204`.
- Workload: one H100, `N_sample=1280`, 245 tokens, BF16,
  `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`,
  `PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL=1`, efficient conv2d pair-bias
  projection.

Hotspot timing:

| subrange | time per trunk block |
| --- | ---: |
| full block | 17.879 ms |
| attention pair-bias module | 9.323 ms |
| attention q/k/v + SDPA + gate/out | 7.473 ms |
| conditioned transition | 6.750 ms |
| adaptive LayerNorm | 1.906 ms |
| pair-bias projection | 0.038 ms |

NCU launch-family attribution, attention subrange:

| launch family | invocations | kernel time | share | SM % | compute/mem % | L2 % | L1 % | tensor % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuDNN/CUTLASS FMHA | 1 | 4.070 ms | 54.6% | 54.8 | 77.1 | 33.9 | 77.3 | 17.4 |
| nvJitLink/cuBLASLt GEMM | 5 | 2.580 ms | 34.6% | 80.3 | 73.9 | 67.1 | 74.9 | 80.3 |
| Triton sigmoid gate | 1 | 0.467 ms | 6.3% | 30.1 | 91.4 | 82.6 | 15.6 | 0.4 |
| Q-scale elementwise multiply | 1 | 0.313 ms | 4.2% | 12.4 | 89.8 | 81.6 | 20.9 | 0.0 |
| copy/cast/fill elementwise | 8 | 0.030 ms | 0.4% | low | low | low | low | low |

NCU launch-family attribution, transition subrange:

| launch family | invocations | kernel time | share | SM % | compute/mem % | L2 % | L1 % | tensor % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nvJitLink/cuBLASLt GEMM | 6 | 3.941 ms | 59.3% | 76.7 | 73.9 | 67.3 | 71.8 | 76.7 |
| Triton SiLU/mul | 1 | 0.938 ms | 14.1% | 31.6 | 91.5 | 82.5 | 15.5 | 0.4 |
| fast LayerNorm | 2 | 0.657 ms | 9.9% | 67.7 | 63.5 | 67.9 | 28.8 | 0.5 |
| Triton sigmoid/mul/add | 1 | 0.614 ms | 9.2% | 25.4 | 93.1 | 80.4 | 13.2 | 0.3 |
| Triton sigmoid/mul | 1 | 0.467 ms | 7.0% | 30.1 | 91.3 | 82.4 | 15.7 | 0.4 |
| copy/cast elementwise | 9 | 0.033 ms | 0.5% | low | low | low | low | low |

Interpretation:
- Larger batches do not materially improve throughput because the dominant
  trunk launches are sample-scaled and already near the relevant H100 pipes:
  GEMMs are tensor-pipe hot, FMHA is the largest single kernel, and the
  remaining pointwise kernels are memory-pipe/L2 limited. Memory capacity is
  not the limiter at `N_sample=1280`; arithmetic and memory bandwidth per
  generated sample are.
- The missing time between CUDA-event subranges and summed NCU kernel time is
  still real framework/library orchestration, but the finite, graph-correct
  capture attempts did not reduce it. A broad trunk attention mega-kernel would
  need to beat a `4.07 ms` FMHA kernel plus `2.58 ms` of tensor-core GEMMs, not
  merely remove Python launch overhead.
- The most credible small same-output target exposed by this refresh was the
  separate Q-scale multiply (`0.313 ms` per attention call), because it is a
  mathematically foldable memory-bound pass.

Decision:
- Do not replace the trunk cuDNN/CUTLASS attention wholesale yet. Try the
  narrow foldable Q-scale mechanism first; reject it if the GEMM path regresses.

### Round 44 - rejected fused query scaling

Status: rejected and reverted

Hypothesis:
- The trunk attention path computes `linear_q(q_x)` and then launches a
  separate memory-bound multiply for `q / sqrt(head_dim)`. Folding the scale
  into the query projection with `torch.addmm(..., alpha=scale, beta=scale)`
  should remove the `0.313 ms` Q-scale launch without changing attention math.

Change:
- Commit `588db8f` added an opt-in `PROTENIX_FUSED_Q_SCALE=1` path in
  `Attention._prep_qkv` for no-grad CUDA inference with contiguous inputs and
  ordinary `Linear` precision.

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round84_fused_q_scale_hotspot_20260701_083632`.
- Workload: same N1280 trunk hotspot as Round 43, paired flag off/on in one
  Slurm job.

Results:

| target | Q-scale fusion off | Q-scale fusion on | delta | memory note | parity |
| --- | ---: | ---: | ---: | --- | --- |
| attention subrange | 7.524 ms | 7.708 ms | -2.4% | peak allocated +459 MiB | max abs `0.00098` |
| attention-pair-bias module | 10.583 ms | 10.675 ms | -0.9% | unchanged peak | max abs `0.00018` |
| full trunk block | 17.984 ms | 18.167 ms | -1.0% | unchanged peak | max abs `0.03125` |

Interpretation:
- The standalone Q-scale multiply is real and memory-bound, but using
  `addmm(alpha=scale,beta=scale)` changed the projection implementation enough
  to lose more than the removed multiply saved. This is another case where a
  mathematically foldable operation does not automatically produce a better
  cuBLASLt schedule.

Decision:
- Reject and revert `PROTENIX_FUSED_Q_SCALE`. Do not carry the default-off path.
- Keep the Round 43 profile as the current launch-level map for trunk work.

### Round 45 - broadcast diffusion conditioning over samples

Status: promoted

Hypothesis:
- In inference, each denoising step uses one scalar `t_hat` that is expanded
  across all samples. The diffusion conditioning single embedding `s` is
  therefore identical for every sample lane at a given step, but the old path
  recomputed conditioning with shape `[N_sample, N_token, c_s]`.
- Computing `s` with sample dimension `1` and relying on downstream
  broadcasting should remove redundant conditioning work and reduce repeated
  broadcast-gate traffic in the trunk without changing model outputs.

Change:
- Commit `cce4c76` added `PROTENIX_BROADCAST_DIFFUSION_S=1`.
- The guard is inference-only (`not training`, no grad, `N_sample > 1`) and
  only fires when `t_hat_noise_level` is provably sample-identical. The common
  inference path has stride `0` in the sample dimension, so the guard avoids a
  synchronization; the equality check is only a fallback for materialized but
  still identical tensors.

Experiments:
- Isolated trunk screen:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round85_broadcast_s_hotspot_20260701_084011`.
- Corrected warmed full gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round95_warmed_broadcast_s_gate_n200_7r6r_20260701_091451`.
- Workload: one H100, 7r6r input, `N_sample=1280`, `N_step=200`, unchunked
  diffusion, current promoted stack, one warmup item before the timed item.

Hotspot results with broadcast `s`:

| target | full `s` | broadcast `s` | delta | parity |
| --- | ---: | ---: | ---: | --- |
| adaptive LayerNorm | 1.902 ms | 1.646 ms | -13.5% | exact |
| attention-pair-bias module | 10.532 ms | 9.764 ms | -7.3% | exact |
| transition with residual | 6.859 ms | 6.754 ms | -1.5% | max abs `0.03125` |
| full trunk block | 17.841 ms | 17.313 ms | -3.0% | max abs `0.03125` |

Warmed full-gate results:

| variant | e2e samples/s | forward samples/s | forward sec | diffusion | conditioning | transformer | peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no broadcast | 7.850 | 7.863 | 162.781 | 136.713 s | 1.972 s | 89.360 s | 18438 |
| broadcast `s` | 8.250 | 8.265 | 154.871 | 128.526 s | 0.035 s | 83.426 s | 17980 |

Interpretation:
- The initial no-warmup N20/N200 screens were directionally useful but
  overestimated the win because the first variant paid one-time kernel/cache
  costs. The corrected warmed gate is the promotion result.
- Broadcasting `s` still gives a real `+5.1%` warmed end-to-end win and cuts
  diffusion conditioning from `~2.0 s` to `~0.04 s`. The larger movement is in
  the diffusion transformer because all sample-broadcast `s` projections and
  gates become smaller or cheaper.

Decision:
- Promote `PROTENIX_BROADCAST_DIFFUSION_S=1` as part of the throughput stack.
- This also creates a new kernel opportunity: many `sigmoid(s_projection) *
  full_sample_tensor (+ residual)` operations now have a broadcast first
  operand that the old same-shape Triton elementwise fusion cannot handle.

### Round 46 - broadcast-aware fused sigmoid gates

Status: promoted

Hypothesis:
- After Round 45, trunk and atom blocks repeatedly apply small broadcast gates
  with shape `[1, N_token, c]` to full sample tensors with shape
  `[N_sample, N_token, c]`.
- The existing `PROTENIX_TRITON_FUSED_ELEMENTWISE=1` kernels only support
  same-shape tensors, so these cases fell back to separate PyTorch broadcast
  elementwise launches. A narrow Triton kernel that indexes the gate with
  `offset % gate_numel` should recover fusion without attempting a broad
  attention mega-kernel.

Change:
- Commit `8142521` added guarded first-dimension-broadcast variants for
  `fused_sigmoid_mul` and `fused_sigmoid_mul_add`.
- The optimized path is intentionally narrow: contiguous CUDA tensors,
  matching dtype, rank-3 shapes, `x.shape == [1, T, C]`, and
  `y.shape == [S, T, C]`. The addend `z` may be either broadcast-shaped or
  full-shaped. All other shapes keep the previous fallback behavior.
- `AttentionPairBias` now routes the final output gate through
  `fused_sigmoid_mul`, so broadcast gates in the trunk attention path can use
  the same fused implementation.

Experiments:
- Direct broadcast-gate microbenchmark:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round93_broadcast_gate_direct_20260701_091232`.
- Warmed full gate:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round95_warmed_broadcast_s_gate_n200_7r6r_20260701_091451`.
- Workload: one H100, 7r6r input, `N_sample=1280`, `N_step=200`, unchunked
  diffusion, broadcast `s` enabled, one warmup item before the timed item.

Direct kernel results for `x=[1,245,768]`, `y=[1280,245,768]`:

| operation | base fallback | broadcast Triton | delta | parity |
| --- | ---: | ---: | ---: | --- |
| `sigmoid(x) * y` | 0.610 ms | 0.321 ms | -47.4% | max abs `0.03125` |
| `sigmoid(x) * y + z_full` | 1.081 ms | 0.474 ms | -56.2% | max abs `0.03125` |
| `sigmoid(x) * y + z_broadcast` | 1.218 ms | 0.321 ms | -73.7% | max abs `0.03125` |

Warmed full-gate results:

| variant | e2e samples/s | forward samples/s | forward sec | diffusion | transformer | atom decoder | peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no broadcast | 7.850 | 7.863 | 162.781 | 136.713 s | 89.360 s | 18.026 s | 18438 |
| broadcast `s` only | 8.250 | 8.265 | 154.871 | 128.526 s | 83.426 s | 17.834 s | 17980 |
| broadcast `s` + broadcast gates | 9.043 | 9.060 | 141.274 | 114.890 s | 72.724 s | 15.191 s | 19560 |

Interpretation:
- This is a real full-workload win, not just a microbenchmark win. The
  candidate improves warmed end-to-end throughput by `+9.6%` over broadcast
  `s` alone and by `+15.2%` over the prior no-broadcast stack.
- The movement is exactly where expected: diffusion transformer time drops
  from `83.4 s` to `72.7 s`; atom decoder also drops because it uses the same
  broadcast-gate primitive in sample-scaled blocks.
- This is the right level of fusion for the current profile. It removes
  memory-bound broadcast elementwise launches around every block while leaving
  cuDNN FMHA and cuBLASLt GEMMs intact.

Decision:
- Promote commit `8142521` with `PROTENIX_BROADCAST_DIFFUSION_S=1`.
- The current measured one-H100 speedup is `17.13x` over the original
  `0.528` samples/sec default.
- Next credible kernel target is a broadcast-aware AdaLN fusion that combines
  full-token layernorm with the broadcast gate/add, rather than a wholesale
  trunk-attention mega-kernel.

### Round 47 - current-stack batch ladder

Status: promoted `N_sample=2560` as the max-throughput point; larger unchunked
batches rejected for now

Question:
- With broadcast `s` and broadcast-aware gates promoted, does a larger
  `N_sample` now improve per-GPU throughput, or is the workload still limited
  by sample-linear compute/memory traffic?

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round96_batch_ladder_current_20260701_093445`.
- Follow-up `N_sample=3840` run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round97_n3840_current_20260701_094703`.
- Commit: `8142521`.
- Workload: one H100, 7r6r input, `N_step=200`, unchunked diffusion, current
  promoted stack, one warmup item before the timed item.
- Shapes: `N_sample=2560`, `3840`, and `5120`.

Results:

| shape | status | e2e samples/s | forward samples/s | forward sec | diffusion | confidence head | peak reserved MiB | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N1280 | stable | 9.043 | 9.060 | 141.274 | 114.890 s | 21.213 s | 19560 | Round 46 warmed gate |
| N2560 | stable | 9.178 | 9.187 | 278.640 | 229.201 s | 41.736 s | 37120 | max screened throughput |
| N3840 | failed | - | - | - | - | - | - | atom encoder BF16 cuBLAS failure during warmup |
| N5120 | failed | - | - | - | - | - | - | same atom encoder BF16 cuBLAS failure during warmup |

Interpretation:
- Doubling from N1280 to N2560 gives only `+1.5%` end-to-end throughput while
  nearly doubling reserved memory (`19.1 GiB -> 36.3 GiB`). This confirms that
  memory capacity was not the limiter at N1280; the dominant diffusion and
  confidence kernels still scale close to linearly with `N_sample`.
- Increasing batch by 3x or 4x is not just low-value; it is currently unsafe in
  the unchunked path. Both N3840 and N5120 fail early in the atom encoder at
  `linear_no_bias_q(q_l)` with `CUBLAS_STATUS_EXECUTION_FAILED`, followed by a
  CUDA illegal-memory-access report. That points to a large-GEMM/workspace or
  kernel-limit failure in the sample-scaled atom path rather than ordinary
  Python overhead.
- Chunking the sample dimension would avoid the failure but would serialize
  the same work and should land near the N2560 throughput ceiling unless the
  atom path is redesigned.

Decision:
- Use `N_sample=2560` for maximum throughput and `N_sample=1280` for
  lower-memory/debug runs.
- Do not pursue larger batch as the main speed strategy. Fixing the remaining
  bottleneck requires reducing per-sample work or making the atom/trunk
  sample-scaled kernels better, not just raising `N_sample`.

### Round 48 - current-stack trunk NCU refresh

Status: diagnostic

Question:
- After promoting broadcast `s` and broadcast-aware gates, what are the real
  launch-level bottlenecks in the diffusion trunk block, and does the profile
  now justify a broader custom kernel?

Experiment:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round98_current_trunk_ncu_20260701_095928`.
- Commit: `1714d91`.
- Workload: one H100, synthetic trunk block with the promoted broadcast shape:
  `a=[1280,245,768]`, `s=[1,245,384]`, `z=[1,128,245,245]`, BF16,
  efficient pair-bias conv2d, residual transition enabled.
- NCU was run inside Slurm using
  `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/nsight-compute/bin/ncu`;
  the login node was not used for profiling.

CUDA-event hotspot:

| subrange | time per trunk block |
| --- | ---: |
| full block | 14.624 ms |
| attention-pair-bias path | 8.305 ms |
| attention q/k/v + SDPA + gate/out | 7.568 ms |
| conditioned transition with residual | 5.351 ms |
| adaptive LayerNorm | 0.756 ms |
| pair-bias projection | 0.035 ms |

NCU launch-family attribution, full block:

| launch family | invocations | kernel time | share | SM % | compute/mem % | L2 % | L1 % | tensor % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FMHA | 1 | 4.038 ms | 46.7% | 54.8 | 77.1 | 34.1 | 77.2 | 17.4 |
| GEMM | 15 | 1.316 ms | 15.2% | 79.1 | 73.1 | 63.7 | 74.2 | 78.7 |
| Triton SiLU/mul | 1 | 0.938 ms | 10.9% | 31.3 | 91.5 | 82.7 | 15.5 | 0.4 |
| Torch elementwise/copy | 12 | 0.797 ms | 9.2% | 11.7 | 89.3 | 80.7 | 17.7 | 0.0 |
| Triton sigmoid/mul | 1 | 0.466 ms | 5.4% | 29.9 | 91.6 | 82.5 | 15.5 | 0.4 |
| LayerNorm | 2 | 0.405 ms | 4.7% | 51.8 | 71.5 | 70.8 | 27.5 | 0.4 |
| Triton broadcast sigmoid/mul/add | 3 | 0.366 ms | 4.2% | 45.4 | 90.2 | 81.5 | 22.2 | 0.5 |
| Triton broadcast sigmoid/mul | 1 | 0.315 ms | 3.6% | 45.9 | 89.5 | 80.8 | 22.5 | 0.6 |

NCU launch-family attribution, residual transition target:

| launch family | invocations | kernel time | share | SM % | compute/mem % | L2 % | L1 % | tensor % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMM | 6 | 1.036 ms | 37.4% | 84.2 | 76.4 | 61.3 | 77.1 | 84.1 |
| Triton SiLU/mul | 1 | 0.939 ms | 33.9% | 31.3 | 91.4 | 82.4 | 15.4 | 0.4 |
| LayerNorm | 1 | 0.400 ms | 14.4% | 52.8 | 72.6 | 71.9 | 28.0 | 0.5 |
| Triton broadcast sigmoid/mul/add | 2 | 0.391 ms | 14.1% | 42.8 | 90.4 | 81.5 | 21.0 | 0.5 |

Interpretation:
- The promoted broadcast gates moved the easy broadcast elementwise bottleneck,
  but the remaining trunk block still has three large classes of work:
  FMHA, tensor-core GEMMs, and memory-bound pointwise kernels.
- A whole-block or whole-attention mega-kernel is still not an obvious first
  move: it would need to beat a good `4.04 ms` FMHA plus tensor-core GEMMs,
  not merely remove launch overhead.
- The next credible fusion target is the memory traffic around AdaLN and
  MLP/transition gates: combine full-token layernorm with broadcast gate/add
  where possible, then consider a larger transition-MLP rewrite only if the
  smaller fusion cannot move the full gate.

Decision:
- Use this as the new launch map.
- Next implementation candidate: broadcast-aware AdaLN fusion for inference
  shapes, with a full-workload gate required before promotion.

### Round 49 - rejected broadcast-aware AdaLN fusion

Status: rejected and reverted

Hypothesis:
- Round 48 showed that trunk AdaLN plus broadcast gate/add still made repeated
  memory-bound launches. Since diffusion conditioning `s` is now computed with
  sample dimension `1` and broadcast across the sample batch, a Triton kernel
  that computes LayerNorm over `a=[S,T,C]` and immediately applies
  `sigmoid(gate[1,T,C]) * norm + shift[1,T,C]` should remove one full-token
  write/read pair in both trunk attention and transition blocks.

Change:
- Commit `5393abe` added an inference-only broadcast-aware AdaLN Triton path
  behind `PROTENIX_TRITON_FUSED_ELEMENTWISE=1`.
- The kernel was guarded to contiguous CUDA rank-3 tensors with
  `a.shape=[S,T,C]`, `gate.shape=shift.shape=[1,T,C]`, matching dtypes, no
  gradients, and `C <= 1024`. Fallback remained the existing fast LayerNorm
  plus fused broadcast gate/add.

Hotspot screen:
- Run directory:
  `/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/round99_adaln_hotspot_20260701_101303`.
- Same-input parity versus fallback: all finite, max abs error `0.0078125`,
  mean abs error `1.0e-08`.

| target | base | fused AdaLN | delta |
| --- | ---: | ---: | ---: |
| adaptive LayerNorm subrange | 0.754 ms | 0.418 ms | -44.6% |
| attention-pair-bias path | 8.239 ms | 7.904 ms | -4.1% |
| conditioned transition | 5.292 ms | 5.101 ms | -3.6% |
| full trunk block | 14.346 ms | 13.758 ms | -4.1% |
| peak reserved | 8846 MiB | 8386 MiB | -5.2% |

Full N2560 gates, one H100, `N_step=200`, current promoted stack:

| run | order | variant | end-to-end samples/sec | forward samples/sec | forward sec | diffusion | transformer | atom decoder | confidence head | peak reserved MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round100 | base then candidate | base `84e21e2` | 9.180 | 9.190 | 278.575 | 228.198 s | 144.605 s | 30.021 s | 42.635 s | 37120 |
| round100 | base then candidate | AdaLN `5393abe` | 9.271 | 9.280 | 275.849 | 225.193 s | 139.249 s | 32.405 s | 42.776 s | 32420 |
| round101 | candidate then base | AdaLN `5393abe` | 9.022 | 9.031 | 283.459 | 228.059 s | 142.079 s | 32.430 s | 46.952 s | 32420 |
| round101 | candidate then base | base `84e21e2` | 9.049 | 9.059 | 282.592 | 231.295 s | 147.643 s | 30.078 s | 43.357 s | 37120 |

Interpretation:
- The isolated mechanism is real: fused AdaLN consistently reduces the trunk
  transformer subpath and lowers reserved memory by about `4.6 GiB` at N2560.
- It does not survive the full-workload promotion rule. The first full gate was
  only `+0.98%` end-to-end, which required a repeat. The reversed-order repeat
  was `-0.30%` end-to-end. Averaging both paired gates gives only about
  `+0.35%`, below the noise and below the bar for carrying another custom
  Triton path in the promoted stack.
- The full gate also shows the practical issue: the AdaLN transformer win can
  be erased by atom-decoder and confidence-head variance. For this workload,
  the next larger gain must remove a more central per-sample cost, not just a
  sub-millisecond trunk epilogue.

Decision:
- Reject and revert `5393abe`.
- Keep Round 48 as the current trunk launch map. The next credible "mega"
  candidate is a design-level transition-MLP or confidence-head kernel that
  removes hidden activation materialization or repeated per-sample confidence
  pairformer work. Do not write a whole-block attention mega-kernel unless the
  design can explain how it beats cuDNN FMHA plus tensor-core GEMMs, not just
  how it removes launches.
