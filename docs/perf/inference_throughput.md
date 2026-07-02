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

They are not included in the headline large-`N_sample` throughput table yet.
The atom-attention boundary fusion is a clean isolated win and reduces peak
allocated memory, but the paired full `N_sample=1280`, `N_step=200` gate
improved forward throughput by only 0.15%, which is below the promotion
threshold.  `PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1` is validated for the
many-sequence/few-sample path described below; its pairformer use is
shape-guarded so larger batches fall back when the fused temporary would be too
large.

For the many-independent-sequence path, two triangle-attention layout fusions
are now default-on when Triton is available.  They can be disabled independently
for profiling:

```bash
export PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE=0
export PROTENIX_TRITON_CUEQ_QKV_LAYOUT=0
```

### Many independent exact-shape inputs

Protein-design campaigns often care less about one sequence with thousands of
samples and more about many independent sequences with `N_sample=1-5`.  The
model already supports a leading protein-batch dimension for same-shape feature
tensors, but the public inference path historically ran one input per
`runner.predict()` call.  The `pred` CLI now exposes exact-shape batching:

```bash
protenix pred \
  -i many_same_shape_inputs.json \
  -o ./output \
  -s 101 \
  -c 10 \
  -p 200 \
  -e 1 \
  -d bf16 \
  --trimul_kernel cuequivariance \
  --triatt_kernel cuequivariance \
  --enable_cache true \
  --enable_fusion true \
  --enable_tf32 true \
  --batch_size 32
```

The batching rule is intentionally conservative: only records whose feature
tensor trees have exactly matching shapes and dtypes are stacked.  Ragged
proteins are not padded into a shared batch because padding is not a proven
semantic no-op for the triangular trunk.  If shapes differ, the runner keeps
separate buckets and flushes each bucket independently.  Directory inputs are
handled as one campaign: all preprocessed JSON records are merged into a
short-lived file before inference so many one-record JSONs can still be packed
into full exact-shape batches.

Public CLI gates on one H100, repeated `7r6r` inputs, `N_sample=1`,
`N_step=200`, confidence enabled, and normal CIF/JSON dumping:

| run | inputs | wall sec | runner job sec | predict+dump sec | seq/s | outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| singleton loop | 32 x `--batch_size 1` | 406.3 | 361.8 | 346.5 | 0.079 | 32 CIFs |
| exact-shape batch | 32 x `--batch_size 32` | 112.5 | 69.3 | 54.5 | 0.284 | 32 CIFs |

The user-visible wrapper speedup is `3.6x`; after model initialization, the
runner speedup is `5.2x`; for the actual predict+dump section it is `6.4x`.
The smaller B8 public gate also passed correctness (8 CIFs) but only reached
`1.9x` wrapper speedup because checkpoint/model initialization dominated such a
short run.

The real campaign shape is often a directory containing one JSON file per
design.  That used to defeat batching because the runner called
`infer_predict()` once per file.  The directory path now resolves the whole
directory as one campaign, keeps JSONs with incompatible `modelSeeds` in
separate transient merged files, and lets the existing exact-shape tensor-tree
bucket decide what can actually stack.  Generated preprocessing siblings such
as `foo-update-msa.json` are ignored when the source `foo.json` is present, so a
rerun does not silently infer both the source and generated JSON.

Paired directory-campaign gate, one H100, job `94948`, run directory
`/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/campaign_json_batching_gate_20260702_112333_fair2`:

| run | inputs | wall sec | seq/s | outputs | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| current `main` | 32 one-record JSON files | 342.7 | 0.093 | 32 CIFs, 0 errors | 32 singleton model calls |
| campaign merge | 32 one-record JSON files | 121.5 | 0.263 | 32 CIFs, 0 errors | one `Predicting 32 same-shape input(s)` call |

This is a `2.82x` end-to-end speedup for the practical one-file-per-design
workflow.  Both variants still write generated MSA JSON siblings during
preprocessing; the promoted change is about not treating those siblings as new
inputs on directory reruns.

The model-only same-shape sweep shows where larger batches stop helping:

| shape | sequence throughput | output-sample throughput | wall sec | peak reserved GiB | dominant work |
| --- | ---: | ---: | ---: | ---: | --- |
| B1, `N_sample=1` | 0.101 seq/s | 0.101 sample/s | 9.86 | 2.65 | diffusion |
| B8, `N_sample=1` | 0.449 seq/s | 0.449 sample/s | 17.81 | 6.84 | pairformer/diffusion |
| B32, `N_sample=1` | 0.842 seq/s | 0.842 sample/s | 37.99 | 20.83 | pairformer |
| B64, `N_sample=1` | 0.867 seq/s | 0.867 sample/s | 73.86 | 39.39 | pairformer |
| B96, `N_sample=1` | 0.876 seq/s | 0.876 sample/s | 109.59 | 58.13 | pairformer |
| B128, `N_sample=1` | 0.881 seq/s | 0.881 sample/s | 145.27 | 74.07 | pairformer |
| B16, `N_sample=5` | 0.412 seq/s | 2.06 sample/s | 38.80 | 13.37 | diffusion |
| B32, `N_sample=5` | 0.430 seq/s | 2.15 sample/s | 74.45 | 24.42 | diffusion |
| B64, `N_sample=5` | 0.436 seq/s | 2.18 sample/s | 146.80 | 46.74 | diffusion |
| B96, `N_sample=5` | 0.438 seq/s | 2.19 sample/s | 219.23 | 69.06 | diffusion |

For this 245-token input, `B=32-64` is the practical knee.  Larger batches keep
raising memory but add little throughput.  The low-sample workload is therefore
not the same bottleneck as the `N_sample=2560` same-output workload: `N_sample=1`
is pairformer-heavy because the trunk runs once per sequence, while
`N_sample=5` shifts more time back to diffusion.

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

#### Pairformer triangle-attention epilogue for many-sequence batches

The many-independent-sequence workload (`batch=B`, `N_sample=1`) is trunk
pairformer-bound.  At B64/N245 on one H100, one pairformer block spends about
half its time in the two triangle-attention modules:

| block subrange | B64 mean ms | block share |
| --- | ---: | ---: |
| starting triangle attention | 19.91 | 25.1% |
| ending triangle attention | 19.88 | 25.1% |
| pair transition | 14.26 | 18.0% |
| outgoing triangle multiplication | 9.64 | 12.2% |
| incoming triangle multiplication | 9.60 | 12.1% |

Within one B64 triangle-attention module, the CUEQ attention kernel is the
largest launch (~10.1 ms), but the output epilogue is also material:
sigmoid-gate projection/layout/multiply/flatten/output-projection costs about
4.97 ms.  The existing CUEQ kernel returns heads in `[..., H, Q, C]` layout.
The default PyTorch epilogue transposes that layout, multiplies by a sigmoid
gate, then flattens to `[..., Q, H * C]`; at high sequence batch, flattening is
a real HBM copy rather than a metadata view.

`triton_sigmoid_mul_heads_first` fuses sigmoid, multiply, head-layout
conversion, and flattening into one streaming Triton write while leaving the
CUEQ attention kernel and output projection unchanged.  This path is
inference-only, shape-guarded, and now default-on for triangle attention.  Set
`PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE=0` to disable only this epilogue,
or set `PROTENIX_TRITON_FUSED_ELEMENTWISE=0` to disable all elementwise Triton
fusions unless the triangle-attention-specific variable is set explicitly.

Representative one-H100 screens, `N_token=245`, BF16, CUEQ triangle ops,
in-place inference:

| screen | baseline | fused epilogue | result |
| --- | ---: | ---: | ---: |
| B64 pairformer block | 76.34 ms | 71.41 ms | 1.069x |
| B96 pairformer block | 114.24 ms | 106.75 ms | 1.070x |
| B64 one-block sequence throughput | 828.6 seq/s | 883.0 seq/s | +6.6% |
| B96 one-block sequence throughput | 828.3 seq/s | 887.8 seq/s | +7.2% |

The candidate kept exact BF16 parity in the batched-vs-loop hotspot screen for
this boundary (`z` max abs 0.0).  The remaining dominant cost is the CUEQ core
itself; replacing that needs a real attention-kernel design, not another
layout cleanup.

#### Pairformer CUEQ q/k/v layout producer

Kernel-level profiling of one B64 starting-triangle-attention module showed
that the apparent `cueq_attention` range was not a single launch: it was the
cuDNN/CUEQ attention kernel plus multiple layout/materialization launches around
it.  The fused q/k/v projection produces `[*, I, J, 3 * H * D]`; CUEQ consumes
three contiguous tensors in `[*, I, H, J, D]`.  The original path reached that
shape with view/transpose metadata, then paid for contiguous q/k/v copies before
the CUEQ CUDA call.

`triton_qkv_to_cueq_heads_first` keeps the fused projection GEMM unchanged and
only changes the boundary after it: one streaming Triton kernel splits q/k/v and
writes CUEQ's physical layout directly.  This is intentionally narrower than a
"mega-kernel"; it removes measured HBM traffic while preserving the high-quality
matrix multiply and attention kernels.  The kernel uses 64-bit offsets because
B96/N245 addresses beyond 2**31 elements in the fused qkv tensor.  A previous
32-bit version was fast but corrupted B96 outputs; this is exactly the kind of
shape-specific overflow a performance path must guard against.

This path is inference-only, shape-guarded, default-on when Triton is present,
and can be disabled with `PROTENIX_TRITON_CUEQ_QKV_LAYOUT=0`.

Representative one-H100 gate, `N_token=245`, `batch=96`, `N_sample=1`, BF16,
CUEQ triangle ops, confidence enabled:

| gate | default layout | direct CUEQ layout | result |
| --- | ---: | ---: | ---: |
| one-block stack | 107.81 ms | 101.78 ms | 1.059x |
| one-block sequence throughput | 890.4 seq/s | 943.2 seq/s | +5.9% |
| full inference wall time | 116.67 s | 109.47 s | 1.066x |
| full inference throughput | 0.8228 seq/s | 0.8770 seq/s | +6.6% |
| pairformer time | 70.80 s | 65.88 s | 1.075x |
| peak allocated/reserved | 57096 / 59526 MiB | 57096 / 59526 MiB | unchanged |

The full gate produced finite coordinates for all 96 sequences and preserved
the confidence head.  Direct q/k/v parity at B96 after the 64-bit offset fix was
bitwise exact against the original layout path.

#### Pairformer transition input fusion for many-sequence batches

The many-independent-sequence workload (`batch=B`, `N_sample=1`) exposes a
different transition boundary in the trunk pairformer.  Pairformer
`Transition` applies LayerNorm, computes two projections from the same giant
`[B * N_token * N_token, C]` activation, and then forms
`SiLU(a) * b`.  With `PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1` and
`PROTENIX_TRITON_FUSED_ELEMENTWISE=1`, the eval path can use one wider
projection followed by the split-aware Triton `SiLU(first_half) * second_half`
consumer.

The wide-input implementation is intentionally shape-guarded by
`PROTENIX_FUSED_TRANSITION_INPUT_MAX_ELEMENTS` (default `4000000000`).  The
fused producer writes a temporary with `2 * hidden` channels; at B64/N245 this
is large but profitable, while an unguarded B96 run failed because the
concatenated activation was too large.  Larger batches therefore keep the
original two GEMMs, but now still fuse the fallback `SiLU(a) * b` expression
with an in-place Triton kernel.  That fallback fusion preserves the original
activation memory model: it overwrites `b` rather than allocating a third giant
hidden tensor.

Two indexing details mattered:

- The split-aware wide-input kernel uses 64-bit element offsets.  At B64/N245
  the output has about 1.97B elements and the concatenated input is addressed
  past 3.9B elements, so 32-bit pointer arithmetic silently overflows and
  produces NaNs.
- The in-place fallback kernel chunks tensors above 1B elements using narrowed
  flat views.  Passing one giant Triton vector with large offsets looked fast
  but failed parity/illegal-access checks near B96.  Narrowed chunks keep each
  kernel's local offsets small while reducing PyTorch's SiLU+multiply launches
  from eight large chunks to three fused chunks at B96.

Representative H100 screens after adding the in-place fallback:

| screen | baseline | candidate | result |
| --- | ---: | ---: | --- |
| B32 transition hotspot | 7.922 ms | 6.429 ms | 1.232x, wide input |
| B64 transition hotspot | 15.835 ms | 12.851 ms | 1.232x, wide input |
| B96 transition hotspot | 23.732 ms | 19.829 ms | 1.197x, in-place fallback |
| B64 full exact-shape gate | 0.47473 seq/s | 0.47468 seq/s | neutral, peak 46.5 GiB |
| B96 full exact-shape gate | 0.50211 seq/s | 0.50818 seq/s | +1.21%, peak 57.8 GiB |

The full gate used one H100, repeated `7r6r`, `N_sample=1`, `N_step=200`,
confidence enabled, exact-shape batching, and the same optimized environment
for main and candidate:
`PROTENIX_TRITON_FUSED_ELEMENTWISE=1` and
`PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1`.  B64 is neutral because the
wide-input path was already active there.  B96 improves because the previous
guarded fallback still had separate memory-bound SiLU and multiply launches.
All full-gate coordinates were finite and confidence summaries were preserved.
The BF16 transition output is not bitwise-identical because fusion changes
rounding order; hotspot parity was `max_abs <= 0.03125` and mean abs about
`0.0013` after the final projection.

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

### 3c. CuTe target from launch-level NCU, not guesswork

Nsight Compute is installed on Tokyo in
`/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/bin/ncu`.  The
first focused reports were captured on `gpu-canary-0` with the representative
trunk shape `N_sample=1280`, `N_token=245`, `c_a=768`, BF16, and broadcast
conditioning:

- `round140_ncu_trunk_boundary_20260701_172113`: baseline transition and
  attention boundary.
- `round141_ncu_transition_fused_input_20260701_172504`: existing
  `PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1` boundary.

The transition block is the better CuTe target than a whole attention
"mega-kernel".  The attention path is dominated by vendor kernels:

| attention launch | duration | SOL signal | interpretation |
| --- | ---: | --- | --- |
| cuDNN/FMHA BF16 attention | 4.01 ms | SM 55%, memory 77%, DRAM 14% | large library kernel; replacing it is high risk |
| q/k/v/g/out GEMMs | 0.47-0.53 ms each | tensor pipe 77-81% | already good H100 tensor-core kernels |
| attention gate | 0.47 ms | DRAM 91%, tensor 0% | memory-bound epilogue candidate |
| layout/copy elementwise | 0.31 ms | DRAM 89% | memory-bound, but secondary |

The transition path has two memory-bound epilogues adjacent to large GEMMs:

| transition launch | duration | SOL signal | interpretation |
| --- | ---: | --- | --- |
| input projection GEMM `a1` | 1.00 ms | tensor pipe 83%, memory 76% | good cuBLASLt/CUTLASS kernel |
| input projection GEMM `a2` | 1.00 ms | tensor pipe 83%, memory 76% | good cuBLASLt/CUTLASS kernel |
| `silu(a1) * a2` Triton kernel | 0.94 ms | DRAM 91%, tensor 0% | real HBM boundary |
| output projection GEMM | 0.95 ms | tensor pipe 89%, memory 79% | good cuBLASLt/CUTLASS kernel |
| final `sigmoid(gate) * projected + residual` | 0.47 ms | DRAM 91%, tensor 0% | real HBM boundary |
| AdaLN LayerNorm on `a` | 0.40 ms | DRAM 71% | material but not adjacent to a GEMM epilogue |
| AdaLN gate/offset application | 0.31 ms | DRAM 90% | already fused/broadcast-aware; smaller target |

The existing wide-input projection experiment explains what a real kernel must
do.  Combining `a1/a2` into one cuBLAS GEMM costs 1.99 ms, essentially the same
as two 1.00 ms GEMMs, and the split-aware `silu` consumer still costs 0.94 ms.
So the remaining win is not "one wider GEMM"; it is avoiding the intermediate
`[a1, a2]` write/read entirely.

Concrete CuTe/CUTLASS target order:

1. **Transition output GEMM epilogue**:
   compute `linear_b(b)` and store
   `sigmoid(linear_s(s)) * linear_b(b) + residual` directly.  This is the
   lower-risk first custom GEMM because it is a standard matmul with an
   auxiliary epilogue load.  Maximum observed launch removal is about 0.47 ms
   per transition block, roughly 3% of the measured 14 ms trunk block.
2. **Dual-input transition GEMM epilogue**:
   compute the two `a -> a1/a2` projections and store
   `silu(a1) * a2` directly.  This is the larger prize, removing the 0.94 ms
   memory-bound SiLU/multiply launch and the 2H intermediate traffic.  It is
   also harder because the kernel must keep the 1.99 ms tensor-core throughput
   of the wide GEMM while pairing the two output halves in the epilogue.
3. **Only then consider attention epilogues**.  The main attention kernel is
   already a vendor FMHA launch, and the rejected `q/k/v/g` fusion shows that
   producer fusion can lose when it disrupts the library schedule.

The combined transition-epilogue ceiling is about 1.4 ms per 14 ms trunk block
before custom-matmul overheads.  That is a real target, but not a 2x target:
the implementation has to be vendor-quality, or the lost GEMM efficiency will
erase the saved HBM traffic.

#### Round 142-147: first custom-kernel boundary attempts

The transition-output epilogue was the first CUTLASS target because it looked
like a standard GEMM plus one auxiliary gate/residual load.  The correct tensor
mapping is a strided-batched GEMM over tokens: each token batch multiplies
`b[:, token, :] @ W.T`, shares the projection weight, and broadcasts
`gate[0, token, :]` over samples.  The legacy CUTLASS
`GemmUniversalWithBroadcast` wrapper did not become a viable implementation
path on the Tokyo CUDA 13 stack: SM90 BF16 has no default configuration in that
wrapper, the SM80 fallback needed extra output-op overloads, and compilation
then hit a CUDA launch-stub failure around CUTLASS `Kernel2`.  This is a
tooling rejection of that wrapper, not of the epilogue boundary itself.  A real
attempt should use native CuTe/SM90 collectives rather than this CUTLASS-2
broadcast path.

The larger dual-input boundary was then prototyped directly in Triton: one
kernel computed both `a @ W1.T` and `a @ W2.T`, then stored
`silu(a1) * a2`.  A synthetic random-weight screen was initially encouraging:
best tile `M128/N128/K64` measured 3.28 ms versus 3.67 ms for two cuBLAS GEMMs
plus the Triton SiLU/multiply.  The representative model gate rejected it:

| screen | baseline | candidate | decision |
| --- | ---: | ---: | --- |
| synthetic random weights | 3.67 ms | 3.28 ms | useful hypothesis only |
| real transition input projection, autocast/module weights | 3.09 ms | 3.54 ms | reject |
| full `ConditionedTransitionBlock` | 5.49 ms | 5.64 ms | reject |
| trunk block | 25.89 ms | 26.21 ms | reject |

The mechanism is clear: the true baseline is not the synthetic cuBLAS timing,
but PyTorch/cuBLAS on the real module weights under autocast.  That path already
runs the input projections in about 3.08 ms including the fused SiLU/multiply,
so a custom kernel must preserve cuBLAS-class GEMM efficiency while deleting
the HBM epilogue.  The Triton prototype did not.  It was reverted rather than
kept as a default-off branch.

The output boundary was also screened with a direct Triton GEMM epilogue in
`round148_transition_output_epilogue_triton_screen_20260701_180003`.  This used
real `ConditionedTransitionBlock` weights under BF16 autocast, a real
`gate=[1, 245, 768]` broadcast, and `N_sample=1280`.  The baseline was
`linear_b(b)` followed by the promoted `fused_sigmoid_mul_add` gate/residual
kernel.  The best direct Triton tile (`M128/N256/K64`) measured 2.00 ms, slower
than the 1.51 ms baseline (`linear_b` 1.02 ms plus gate/residual 0.47 ms).
Parity was acceptable for BF16 (`max_abs=0.03125`, `mean_abs=3.8e-5`), so this
is a performance rejection, not a correctness rejection.

The transition-output epilogue remains a good first **native CuTe** target, but
the implementation has to keep the vendor GEMM schedule and add the epilogue
load/store logic.  Replacing cuBLAS with a generic Triton matmul loses more GEMM
efficiency than the 0.47 ms memory-bound epilogue can pay back.

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

For many independent low-sample inputs, the bottleneck shifts.  At
`B=64, N_sample=1`, full inference spends about 44.0 s in the trunk pairformer
and 26.9 s in diffusion.  At `B=64, N_sample=5`, diffusion grows to 98.4 s
while pairformer remains about 43.5 s.  This means the next kernel target
depends on the campaign:

| campaign shape | first-class target | reason |
| --- | --- | --- |
| many sequences, `N_sample=1` | pairformer triangle attention / transition boundaries | trunk is the largest non-amortized cost |
| many sequences, `N_sample=5` | diffusion token transformer plus pairformer | diffusion dominates but pairformer is still material |
| one sequence, very large `N_sample` | diffusion token transformer / confidence streaming | pairformer is amortized over samples |

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

## Post-QKV kernel screens

After promoting the CUEQ q/k/v layout producer, the B64/B96 pairformer block
profile was rerun on `main` (`6962922`).  At B96/N245, one block is about
100.1 ms.  Triangle attention remains the largest target, but the character of
the bottleneck changed:

| block subrange | B96 mean ms | block share |
| --- | ---: | ---: |
| starting triangle attention | 21.94 | 21.9% |
| ending triangle attention | 21.89 | 21.9% |
| pair transition | 20.42 | 20.4% |
| outgoing triangle multiplication | 13.61 | 13.6% |
| incoming triangle multiplication | 13.53 | 13.5% |

Nsight Compute on one B64 starting triangle-attention module showed 34 launches.
The remaining time is concentrated in the vendor attention path and the two
layout/gate kernels we already exposed:

| launch family | launches | total ms | interpretation |
| --- | ---: | ---: | --- |
| cuDNN FMHA/SDPA inside CUEQ | 1 | 6.07 | main compute kernel |
| CUEQ/nvjet transforms | 12 | 2.78 | wrapper/layout work around cuDNN |
| Triton q/k/v layout producer | 1 | 2.06 | 85% DRAM, near memory roof |
| Triton gate/flatten epilogue | 1 | 0.96 | 91% DRAM |
| LayerNorm | 1 | 0.81 | material, memory-heavy |
| small GEMMs | 4 | 0.56 | not the bottleneck |

This profile motivated an opt-in fused triangle-attention prototype on branch
`codex/protenix-fused-triangle-attention-prototype`.  The prototype reads the
fused qkv projection, applies the triangular key bias/mask, multiplies by the
sigmoid gate, and writes the flattened output directly.  Small N64 start/end
screens were numerically sane (`max_abs_error=0.00195`), but the schedule was
not competitive:

| B16/N245 screen | CUEQ default | fused Triton prototype | decision |
| --- | ---: | ---: | --- |
| tile 16 module forward | 4.26 ms | 6.22 ms | reject |
| tile 32 module forward | 4.24 ms | 7.78 ms | reject |

The boundary is still attractive, but the implementation needs a vendor-quality
CuTe/CUTLASS/CUEQ extension.  A simple Triton FlashAttention-style clone creates
too many small programs and loses the cuDNN schedule faster than it saves q/k/v
layout and epilogue traffic.

CUEQ triangle-multiplication autotuning was also screened because triangle
multiplication is now about 27% of the B64/B96 block.  `CUEQ_TRITON_TUNING=ONDEMAND`
improved each triangle-multiplication module by about 3%, but only moved total
B64 block time from 65.79 ms to 65.57 ms (+0.34%).  That is below promotion
threshold and not worth depending on a per-shape external tuning cache.

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
| Custom Triton transition input GEMM+SiLU | synthetic +12%, but real transition input path slowed from 3.09 ms to 3.54 ms | custom GEMM screens must use real module weights/autocast; cuBLAS efficiency is the bottleneck to match |
| Unguarded pairformer transition input fusion at B96 | failed before producing model output | the concatenated pair-transition activation is too large; keep the shape guard |
| Custom Triton transition output GEMM+gate/residual | best tile 2.00 ms versus 1.51 ms for cuBLAS plus fused gate/residual | the output epilogue is still attractive, but only if implemented as a vendor-quality CuTe/CUTLASS epilogue |
| Naive fused Triton triangle attention | correct at N64, but B16/N245 slowed from 4.24-4.26 ms to 6.22-7.78 ms | the right boundary still needs a vendor-quality CUEQ/CuTe schedule |
| CUEQ triangle-multiplication ONDEMAND tuning | total B64 block +0.34%, below noise | the shipped/default schedule is close enough for this shape |
| Default BF16 full-token attention after the promoted stack | isolated trunk-attention hotspot improved, but full exact-shape gates moved only +0.16% at B64 and +0.27% at B96 | do not promote dtype-policy changes unless the representative throughput gate moves, not just the local kernel |
| Inference DataLoader workers for exact-shape batches | B8 had a small screen win, but B32 hit tensor-sharing failures unless switched to slower file-system sharing; clean B32 no-worker batching was better | do not add host-pipeline complexity unless it survives the actual many-input gate |
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
  Triton local attention, fused elementwise gate calls, and shape-guarded
  pairformer `Transition` input-projection fusion.
- `protenix/model/modules/transformer.py`: local-attention bias fusion and
  attention/transition residual fusion hooks plus guarded transition
  input-projection fusion.
- `protenix/model/modules/local_attention_triton.py`: narrow Triton atom local
  attention forward kernel, including optional local-attention gate fusion.
- `protenix/model/modules/local_attention_bias_triton.py`: fused LayerNorm +
  projection + layout for atom local-attention bias.
- `protenix/model/modules/fused_elementwise_triton.py`: small inference-only
  Triton gate kernels with PyTorch fallbacks, including the 64-bit-indexed
  split-aware `silu(first_half) * second_half` consumer for concatenated
  transition projections.
- `protenix/model/triangular/qkv_layout_triton.py`: inference-only CUEQ q/k/v
  layout producer for triangle attention; keeps the fused projection GEMM and
  CUEQ attention kernel unchanged while removing hidden contiguous-copy traffic.
- `protenix/model/modules/confidence.py` and `protenix/model/protenix.py`:
  confidence-logit chunk iteration and streaming summary consumption.
- `runner/inference.py`, `runner/batch_inference.py`, and
  `configs/configs_inference.py`: public exact-shape inference batching.  The
  runner stacks only identical feature tensor trees, splits predictions back to
  one output per input, and leaves ragged shapes as separate buckets.

Profiling and reproducibility helpers:

- `scripts/perf/protenix_inference_benchmark.py`: full-model throughput gate.
- `scripts/perf/full_inference_precision_audit.py`: full-run matmul precision
  audit; records forced-FP32 modules and non-module einsums/matmuls.
- `scripts/perf/atom_transformer_hotspot.py`: atom block hotspot screen.
- `scripts/perf/local_attention_hotspot.py`: local-attention kernel screen.
- `scripts/perf/local_attention_bias_hotspot.py`: bias-projection screen.
- `scripts/perf/trunk_attention_hotspot.py`: trunk attention block screen.
- `scripts/perf/triangle_attention_breakdown.py`: per-launch/subrange screen
  for CUEQ triangle attention and its layout/epilogue boundaries.
- `scripts/perf/confidence_head_hotspot.py`: confidence pairformer screen.
- `scripts/perf/confidence_precision_screen.py`: confidence precision/parity
  screen with real checkpoint weights and matmul dtype tracing.
- `scripts/perf/trace_aggregate.py`: compact profiler trace summarizer.

## If continuing from here

Do not expect another large gain from config changes.  The next proper
same-output campaign should be one of:

1. a deliberate atom/trunk kernel-layout rewrite, especially around producers
   feeding attention kernels;
2. a confidence-pairformer precision/tiling campaign with explicit numerical
   acceptance criteria;
3. vendor-quality fused epilogues for specific matmul-adjacent patterns.

All three are larger kernel-engineering projects.  They should start with a
fresh profile, an isolated hotspot screen, and a full same-output N1280/N2560
gate before promotion.
