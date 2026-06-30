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
