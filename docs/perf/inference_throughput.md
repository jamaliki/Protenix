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

A follow-up precision screen loaded the real
`protenix_base_default_v1.0.0.pt` confidence-head weights and ran the
production-like BF16 autocast path.  In that path, the confidence pairformer
matmuls already produce BF16 outputs.  The remaining FP32 matmuls are the final
confidence logits:

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
throughput optimization.  The safe lower-precision candidates exist, but the
matmuls are too small in the real path to be the next bottleneck.

## Negative results worth remembering

These failed because the real workload, not the isolated kernel, is the gate:

| idea | result | lesson |
| --- | --- | --- |
| Larger unchunked `N_sample=3840/5120` | failed in atom encoder BF16 cuBLAS / illegal access | memory headroom is not the only batch-size limit |
| Fusing activation into a custom Triton matmul | slower despite less memory | do not replace cuBLAS with a mediocre matmul |
| Query-splitting the local-attention CTA | slower and more memory | reducing register pressure can duplicate other work |
| Forcing confidence pairformer non-inplace | +0.08% e2e, below noise | wrapper clone cleanup is too small |
| BF16 final confidence logits | reduced FP32 logit matmuls but no NCU/timing win | the remaining FP32 matmuls are too small and cast traffic dominates |
| Generic full-token Triton attention | slower than cuDNN/SDPA | vendor kernels win unless the shape mismatch is severe |
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
  transition-residual fusion hooks.
- `protenix/model/modules/local_attention_triton.py`: narrow Triton atom local
  attention forward kernel.
- `protenix/model/modules/local_attention_bias_triton.py`: fused LayerNorm +
  projection + layout for atom local-attention bias.
- `protenix/model/modules/fused_elementwise_triton.py`: small inference-only
  Triton gate kernels with PyTorch fallbacks.
- `protenix/model/modules/confidence.py` and `protenix/model/protenix.py`:
  confidence-logit chunk iteration and streaming summary consumption.

Profiling and reproducibility helpers:

- `scripts/perf/protenix_inference_benchmark.py`: full-model throughput gate.
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
