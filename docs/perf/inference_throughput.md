# Protenix inference throughput notes

This note documents the H100 inference-throughput path added in this branch.
It is intentionally written as a learning resource: the important part is not
just which flags were faster, but why they were faster and which tempting
optimizations failed.

The workload optimized here is **same-output Protenix inference with confidence
enabled** on one H100 80 GB GPU.  Throughput is measured as generated samples
per second per GPU.  The representative benchmark input is `examples/example.json`
index `0` (`7r6r`, 245 tokens, 2529 atoms), with `N_step=200`,
`N_cycle=10`, one warmup item, one timed item, and output dumping disabled.

## Current high-throughput configuration

The following stack is the best measured same-output configuration for the
7r6r-style benchmark:

```bash
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

Measured one-H100 throughput:

| mode | e2e samples/sec | forward samples/sec | forward sec | peak reserved MiB | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| default Protenix, `N_sample=5` | 0.528 | 0.544 | 2414 s | - | baseline |
| optimized stack, `N_sample=1280` | 9.043 | 9.060 | 141.274 s | 19560 | lower-memory setting |
| optimized stack, `N_sample=2560` | 9.178-9.206 | 9.187-9.215 | 277.8-278.6 s | 37120 | best same-output setting |

The same-output speedup is about **17x per GPU** over the original default.
Numbers vary slightly by run and node state; changes below about 1% should be
treated as noise unless repeated in a paired gate.

Later boundary-fusion experiments added three more opt-in flags:

```bash
export PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE=1
export PROTENIX_TRITON_FUSED_ATTENTION_RESIDUAL=1
export PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1
```

They are not included in the headline throughput table yet.  The atom-attention
boundary fusion is a clean isolated win and reduces peak allocated memory, but
the paired full `N_sample=1280`, `N_step=200` gate improved forward throughput
by only 0.15%, which is below the promotion threshold.  Keep these flags as
experimental until a repeated `N_sample=1280/2560` gate shows a robust samples/s
gain.

## Reproducing the benchmark

Use the benchmark harness rather than timing ad hoc commands.  It records the
Git commit, environment, input shape, per-item timings, and memory peaks.

```bash
PROTENIX_REPO=/path/to/Protenix \
PROTENIX_ROOT_DIR=/path/to/protenix_data \
PROTENIX_RUN_ROOT=/path/to/runs \
PROTENIX_RUN_NAME=n2560_same_output \
N_SAMPLE=2560 \
N_STEP=200 \
N_CYCLE=10 \
SAMPLE_DIFFUSION_CHUNK_SIZE=none \
CHUNK_SIZE=none \
INPUT_JSON=examples/example.json \
INPUT_INDICES=0 \
WARMUP_ITEMS=1 \
DUMP_OUTPUTS=false \
scripts/perf/tokyo_protenix_benchmark.sbatch
```

On Sandpit Tokyo this sbatch requests one H100, 14 CPUs, and 128 GB host RAM.
For other systems, use the Python harness directly:

```bash
python scripts/perf/protenix_inference_benchmark.py \
  --input-json examples/example.json \
  --input-indices 0 \
  --dump-dir runs/n2560_same_output \
  --model-name protenix_base_default_v1.0.0 \
  --seeds 101 \
  --repeat 1 \
  --warmup-items 1 \
  --n-sample 2560 \
  --n-step 200 \
  --n-cycle 10 \
  --use-default-params false \
  --dtype bf16 \
  --chunk-size none \
  --sample-diffusion-chunk-size none \
  --dump false
```

## What moved the needle

### 1. Keep sample-independent work sample-independent

During diffusion, the conditioning noise level is the same scalar for every
sample in a denoising step.  The previous path expanded that scalar across all
samples and then recomputed sample-invariant conditioning for each lane.

`PROTENIX_BROADCAST_DIFFUSION_S=1` keeps that conditioning in a singleton sample
lane when it is provably identical, then lets later tensor operations broadcast
it.  This is a common GPU optimization pattern: do not materialize or recompute
data just because broadcasting makes it convenient.

### 2. Fuse memory-bound elementwise gates

Many blocks compute patterns like:

```python
torch.sigmoid(x) * y
torch.sigmoid(x) * y + z
F.silu(x) * y
```

Each unfused expression launches multiple kernels and writes intermediate
tensors to HBM.  `PROTENIX_TRITON_FUSED_ELEMENTWISE=1` uses small Triton kernels
to do these operations in one pass.  The broadcast-aware variants are especially
important after `PROTENIX_BROADCAST_DIFFUSION_S=1`, because the gate tensor has
sample dimension `1` while the activation tensor has sample dimension
`N_sample`.

This is the right kind of fusion: the fused operation is simple, memory-bound,
and avoids intermediate reads/writes without replacing a high-quality GEMM.

### 3. Specialize atom local attention

Atom local attention has a stable inference shape: 32 query atoms attend over a
128-key local window with head dimension 32.  The generic PyTorch SDPA path is
not ideal for this tiny fixed window.

`PROTENIX_TRITON_LOCAL_ATTN=1` adds a guarded Triton forward path for exactly
this shape.  It returns `None` for unsupported shapes so callers fall back to
the original PyTorch implementation.  The bias projection feeding local
attention is also fused by `PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS=1`, which
computes LayerNorm, projection, and final attention-bias layout directly.

Two details matter:

- the local-attention path is inference-only and disabled when gradients are
  enabled;
- the Triton kernel is deliberately narrow.  A broad replacement would be
  harder to validate and easier to make slower.

The next boundary around this kernel is the attention gate.  With
`PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE=1`, the local-attention kernel consumes
the precomputed gate and stores `sigmoid(gate) * attention` directly.  That
removes one elementwise launch and one full attention-output read/write.

At `N_sample=1280` with cached/broadcast pair bias, the isolated
local-attention-plus-gate boundary improved from 9.72 ms to 8.92 ms
(1.09x).  A full atom block improved from 27.33 ms to 26.50 ms (about 3.0%)
and peak allocation fell by about 790 MiB when combined with
`PROTENIX_TRITON_FUSED_ATTENTION_RESIDUAL=1`.

The full same-output inference gate was much smaller:

| gate | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| `N_sample=1280`, `N_step=200`, forward samples/s | 9.156 | 9.170 | +0.15% |
| forward sec | 139.80 | 139.59 | -0.15% |
| peak allocated MiB | 15245.8 | 14455.4 | -790.3 |

Decision: useful kernel-learning result and memory reduction, but not promoted
as a throughput win until repeated full gates clear the noise floor.

### 3b. Fuse only across a boundary the consumer can still use

Two token-transformer projection fusions were screened because the diffusion
transformer is the largest remaining diffusion subcomponent.

First, combining self-attention `q/k/v/g` projections into one wider GEMM was
rejected.  It had exact parity on a smaller shared-input check, but the
representative trunk hotspot got slower:

| path | block ms | attention ms | peak allocated MiB |
| --- | ---: | ---: | ---: |
| current | 14.06-14.12 | 7.47-7.50 | 7001 |
| fused `q/k/v/g` | 14.86-14.90 | 8.29 | 7477 |

The likely mechanism is that the wider GEMM and cached concatenated weights did
not beat the existing cuBLAS launch schedule for this shape.  The code path was
removed rather than left as default-off cruft.

Second, combining the two transition input projections (`a1/a2`) was promising
only after the consumer boundary was fixed.  The raw concatenated GEMM was 1.07x
faster in a microbenchmark, but its split halves are non-contiguous and made the
existing `fused_silu_mul` fall back to slower PyTorch.  A split-aware Triton
consumer now computes `silu(first_half) * second_half` directly from the
concatenated projection.  With that fix, the trunk block was speed-neutral
within noise but peak allocation dropped by about 900 MiB:

| path | block ms | transition ms | peak allocated MiB |
| --- | ---: | ---: | ---: |
| current | 13.91 / 13.91 | 5.20 / 5.27 | 7915 |
| fused transition input | 13.80 / 13.94 | 5.22 / 5.22 | 7010 |

Decision: keep the implementation guarded by
`PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1` for follow-up memory/batch-size
screens, but do not count it as a promoted throughput optimization yet.

Third, fusing the two AdaLN conditioning projections was rejected.  The idea was
reasonable: every block computes both a gate and an offset from the same
normalized `s`, and the high-throughput diffusion path broadcasts `s` over all
samples.  The prototype used one wider `s -> [gate, offset]` GEMM plus a
split-aware Triton consumer.  It had exact parity in a nonzero-weight BF16
screen, but it did not move the representative trunk block:

| screen | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| AdaLN micro, ms | 0.749 | 0.747 | +0.26% |
| trunk block, paired average ms | 14.047 | 14.041 | +0.046% |
| peak allocated MiB | 7000 | 7007 | worse |

The mechanism is now clear: this boundary is real but too small after
sample-broadcast conditioning.  The code path was removed so future work does
not carry another default-off branch with no throughput signal.

### 4. Use lower precision only where profiling supports it

`PROTENIX_BF16_DIFFUSION_CORE=1` and `PROTENIX_BF16_ATOM_ATTENTION=1` reduce
traffic in the diffusion core and atom attention path.  Full token attention is
allowed to stay BF16 with `PROTENIX_ATTENTION_FORCE_FP32=0`, but local atom
attention still has guards and fallbacks because not every cuDNN/SDPA shape is
safe or fast in BF16.

The lesson is to use precision as a local performance decision, not a global
switch.

### 5. Stream confidence summary

At high `N_sample`, full PAE/PDE logits are large:

```text
[N_sample, N_token, N_token, bins]
```

`PROTENIX_STREAM_CONFIDENCE_SUMMARY=1` computes confidence logits in sample
chunks and immediately consumes each chunk into `summary_confidence` and
`full_data`.  This preserves confidence outputs while avoiding materializing
the full logit tensors for all samples at once.  The promoted chunk size is:

```bash
export PROTENIX_CONFIDENCE_LOGIT_CHUNK_SIZE=32
```

This improved memory behavior and enabled larger sample batches without making
confidence a separate/deferred stage.

## Bottleneck map after optimization

At `N_sample=2560`, the forward pass is still dominated by real per-sample
diffusion and confidence work.  Increasing batch size does not deliver another
large gain because the hot kernels scale almost linearly with `N_sample`.

Representative full-path timing at the promoted point:

| component | scale |
| --- | --- |
| diffusion | about 229 s |
| confidence head | about 42 s |
| feature/input work | subsecond |

With `PROTENIX_TIMING_DIFFUSION=1`, a shorter `N_sample=1280`, `N_step=20`
screen split diffusion itself as:

| diffusion subcomponent | time |
| --- | ---: |
| token diffusion transformer | 7.86 s |
| atom encoder | 3.74 s |
| atom decoder | 1.96 s |
| conditioning | 0.016 s |
| input/output scale | 0.005 s |

This is why the most attractive remaining throughput target is not another
atom-local-attention tweak; it is the token diffusion transformer, especially
attention/transition boundaries where a custom kernel must preserve the
consumer layout.

An early confidence pairformer hotspot was profiled in a deliberately FP32
standalone screen.  That was useful for understanding the worst case, but it
was not the production inference precision path:

| family | launch share | interpretation |
| --- | ---: | --- |
| FP32 SGEMM / CUTLASS / cuBLAS | 40.5% | moderately SM/FMA busy, not tensor-core peak |
| elementwise / copy / cast | 19.8% | real memory traffic, but not enough alone for a large end-to-end win |
| FMHA / SDPA | 19.1% | mostly L1/on-chip pressure, not DRAM-saturated |
| cuequivariance fused gated dual GEMM | 11.2% | already a specialized kernel |
| LayerNorm | 6.4% | material but not dominant |

This is why a single "mega-kernel" is not an obvious next win.  Some hot
launches are already efficient vendor/library kernels, while the memory-bound
pieces are spread across the pairformer.

### Full-path precision audit

Use `scripts/perf/full_inference_precision_audit.py` to rank matmul-like work
by inferred compute dtype across a real `runner.predict()` call.  The script
records the explicit `module.precision` on Protenix/OpenFold `Linear` modules,
because those modules can force FP32 internally and then cast the output back.
Output dtype alone is not a reliable precision audit.

On one H100, commit `2396ab2`, input `7r6r`, `N_sample=32`, `N_step=20`, and
the optimized inference flags above, the audit reported:

| family | estimated matmul FLOPs | dtype / reason |
| --- | ---: | --- |
| diffusion module | 53.2 TF | BF16, promoted optimized path |
| trunk pairformer stack | 21.5 TF | BF16 |
| non-module attention/einsum ops | 6.5 TF | BF16 |
| confidence head | 5.8 TF | FP32 because `update_inference_configs()` sets `skip_amp.confidence_head=True` for base models under 2560 tokens |
| cuequivariance triangle-multiplicative einsums | 1.0 TF | FP32, associated with the FP32 confidence path |
| explicitly forced FP32 Protenix linears | 0.013 TF | too small to matter end-to-end |

The largest explicitly forced FP32 linears were in diffusion conditioning and
coordinate projection:

| module/op | shape/count in audit | estimated work |
| --- | --- | ---: |
| `diffusion_conditioning.linear_no_bias_z` | `[245,245,256] -> [245,245,128]`, once | 3.93 GF |
| `diffusion_conditioning.linear_no_bias_s` | `[245,833] -> [245,384]`, 20 calls | 3.13 GF |
| `diffusion_module.linear_no_bias_s` | `[1,245,384] -> [1,245,768]`, 20 calls | 2.89 GF |
| atom encoder/decoder coordinate linears | 20 calls each | 2.49 GF combined |

These are numerically plausible lower-precision candidates, but they are not
legitimate throughput targets: together they are less than 0.02% of the
estimated matmul work in the audit.

The material FP32 candidate was therefore the whole confidence head, not the
small forced-FP32 coordinate linears.  `scripts/perf/protenix_inference_benchmark.py`
now has a default-neutral screening override:

```bash
--skip-amp-confidence-head false
```

That override keeps the production default unless explicitly passed, then lets
the confidence head run under the outer BF16 autocast context after
`update_inference_configs()`.

A paired screen at `N_sample=128`, `N_step=20` rejected this idea:

| run | baseline forward samples/s | confidence AMP samples/s | delta | confidence-head time |
| --- | ---: | ---: | ---: | --- |
| baseline then candidate | 20.79 | 19.64 | -5.6% | 2.11s -> 2.38s |
| candidate then baseline | 20.96 | 20.35 | -2.9% | 2.10s -> 2.28s |

Decision: do not promote whole-confidence BF16/AMP.  With
`enable_tf32=True`, the current FP32 confidence GEMMs can still use TF32 tensor
cores where cuBLAS/cuDNN allow it.  The BF16 autocast path adds casts and sends
some library/einsum work down a slower route for this shape.

A narrower precision screen loaded the real
`protenix_base_default_v1.0.0.pt` confidence-head weights and tested only the
final confidence logits under BF16.  In that isolated direct-module path, the
pairformer matmuls can produce BF16 outputs; the remaining FP32 matmuls are the
final confidence logits:

| module/op | shape | status |
| --- | --- | --- |
| `linear_no_bias_pae` | `[245, 245, 128] -> [245, 245, 64]` | FP32 |
| `linear_no_bias_pde` | `[245, 245, 128] -> [245, 245, 64]` | FP32 |
| pLDDT `einsum` | `[2529, 384] x [2529, 384, 50] -> [2529, 50]` | FP32 |
| resolved `einsum` | `[2529, 384] x [2529, 384, 2] -> [2529, 2]` | FP32, tiny |

On one H100 confidence sample, Nsight Compute reported 9.21 ms total profiled
kernel time for the current production-like path.  FP32 GEMM/GEMV launches were
only 0.16 ms (1.7%).  Allowing the final logit matmuls to use BF16 reduced that
FP32 family to 0.07 ms, but added cast traffic and raised total profiled time to
9.32 ms.  The multi-sample screen showed similarly small deltas: 73.72 ms for
`prod`, 73.15 ms for BF16 logits, and 72.77 ms for `torch.set_float32_matmul_precision("high")`.
Those are below the noise floor for a full inference gate, and BF16 logits also
raised peak allocated memory in the screen (about 705 MiB versus 652 MiB) due to
upcast traffic.

Decision: do not promote BF16 confidence logits or global TF32 precision as a
throughput optimization.  The safe lower-precision candidates exist, but either
they are too small in the real path or the full lower-precision route is slower
than the current FP32+TF32 policy.

## Negative results worth remembering

These failed because the real workload, not the isolated kernel, is the gate:

| idea | result | lesson |
| --- | --- | --- |
| Larger unchunked `N_sample=3840/5120` | failed in atom encoder BF16 cuBLAS / illegal access | memory headroom is not the only batch-size limit |
| Fusing activation into a custom Triton matmul | slower despite less memory | do not replace cuBLAS with a mediocre matmul |
| Query-splitting the local-attention CTA | slower and more memory | reducing register pressure can duplicate other work |
| Forcing confidence pairformer non-inplace | +0.08% e2e, below noise | wrapper clone cleanup is too small |
| BF16 final confidence logits | reduced FP32 logit matmuls but no NCU/timing win | the remaining FP32 matmuls are too small and cast traffic dominates |
| Whole confidence head under AMP | 2.9-5.6% slower at `N_sample=128`, `N_step=20` | lower dtype is not automatically faster when FP32 uses TF32 tensor cores and casts/library paths change |
| Generic full-token Triton attention | slower than cuDNN/SDPA | vendor kernels win unless the shape mismatch is severe |
| Fused self-attention `q/k/v/g` projection | exact parity but trunk block slowed from about 14.1 ms to 14.9 ms | fewer launches are not enough if the wider GEMM/cache/layout is worse |
| Fused transition `a1/a2` projection without split-aware consumer | transition slowed from about 5.3 ms to 7.0 ms | fusing the producer can break the consumer by creating non-contiguous halves |
| CUDA graph capture of the denoiser | slower in this setup | graph overhead/constraints can outweigh launch savings |

The main learning point: isolated wins are hypotheses, not promotions.  A
candidate becomes part of the throughput stack only after a representative
same-output gate improves samples/sec and preserves memory/numerical guardrails.

## Code map

Production code touched by the throughput stack:

- `protenix/model/generator.py`: precomputes scalar diffusion schedules once
  outside the sampling loop.
- `protenix/model/utils.py`: optional GPU-side random augmentation, avoiding
  CPU SciPy rotation generation for inference batches.
- `protenix/model/modules/diffusion.py`: BF16/autocast controls, component
  timing, and sample-broadcast diffusion conditioning.
- `protenix/model/modules/primitives.py`: SDPA fallback policy, optional
  Triton local attention, and fused elementwise gate calls.
- `protenix/model/modules/transformer.py`: local-attention bias fusion and
  attention/transition residual fusion hooks plus guarded transition
  input-projection fusion.
- `protenix/model/modules/local_attention_triton.py`: narrow Triton atom local
  attention forward kernel, including optional local-attention gate fusion.
- `protenix/model/modules/local_attention_bias_triton.py`: fused LayerNorm +
  projection + layout for atom local-attention bias.
- `protenix/model/modules/fused_elementwise_triton.py`: small inference-only
  Triton gate kernels with PyTorch fallbacks, including the split-aware
  `silu(first_half) * second_half` consumer for concatenated transition
  projections.
- `protenix/model/modules/confidence.py` and `protenix/model/protenix.py`:
  confidence-logit chunk iteration and streaming summary consumption.

Profiling and reproducibility helpers:

- `scripts/perf/protenix_inference_benchmark.py`: full-model throughput gate.
- `scripts/perf/full_inference_precision_audit.py`: full-run matmul precision
  audit; records forced-FP32 modules and non-module einsums/matmuls.
- `scripts/perf/atom_transformer_hotspot.py`: atom block hotspot screen.
- `scripts/perf/local_attention_hotspot.py`: local-attention kernel screen.
- `scripts/perf/local_attention_bias_hotspot.py`: bias-projection screen.
- `scripts/perf/trunk_attention_hotspot.py`: trunk attention block screen.
- `scripts/perf/confidence_head_hotspot.py`: confidence pairformer screen.
- `scripts/perf/confidence_precision_screen.py`: confidence precision/parity
  screen with real checkpoint weights and matmul dtype tracing.
- `scripts/perf/trace_aggregate.py`: compact profiler trace summarizer.

## If continuing from here

Do not expect another large gain from config changes.  The next proper
same-output campaign should be one of:

1. a deliberate atom/trunk kernel-layout rewrite, especially around producers
   feeding local attention;
2. a confidence-pairformer precision/tiling campaign with explicit numerical
   acceptance criteria;
3. vendor-quality fused epilogues for specific matmul-adjacent patterns.

All three are larger kernel-engineering projects.  They should start with a
fresh profile, an isolated hotspot screen, and a full same-output N1280/N2560
gate before promotion.
