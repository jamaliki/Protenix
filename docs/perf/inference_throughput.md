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

### Many independent inputs

Protein-design campaigns often care less about one sequence with thousands of
samples and more about many independent sequences with `N_sample=1-5`.  The
model already supports a leading protein-batch dimension for same-shape feature
tensors, but the public inference path historically ran one input per
`runner.predict()` call.  The `pred` CLI now exposes campaign batching:

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

The default `--batch_mode auto` chooses the fastest safe boundary for each
bucket:

- If full feature tensor trees match, the whole model is run as one exact-shape
  batch.  This is still the fastest path because diffusion and confidence are
  batched too.
- If token/MSA/template trunk shapes match but atom counts differ, only the
  atom-independent trunk is batched.  The runner computes atom input embeddings
  per record, stacks `s_inputs` and token-trunk features, runs one pairformer
  trunk call, then finishes diffusion and confidence per original atom-shaped
  record.
- If token counts differ, the CLI first length-sorts the raw JSON records, then
  pads only token-trunk tensors inside each batch.  A real `pair_mask` is passed
  through template, MSA, and pairformer blocks, and the valid `s`/`z` crop is
  handed back to the atom-shaped diffusion/confidence tail.

This fixes same-length mutation campaigns without padding fake atoms through
atom attention or confidence reductions, and it also supports the common
different-length design campaign case.  Padded-token trunk batching is a
controlled numerical-change feature, not a bitwise singleton reproduction path:
the masks block padded content, but BF16/TF32 reductions can still differ
slightly because kernels see a different physical token length.  Use
`--batch_mode exact --batch_size 1` for bitwise singleton debugging.  Directory
inputs are handled as one campaign: all preprocessed JSON records are merged
into a short-lived, length-sorted file before inference so many one-record JSONs
can still fill batches.

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
separate transient merged files, and lets the auto batcher decide whether each
bucket can use full-model exact batching, same-token trunk batching, or
padded-token trunk batching.  Generated preprocessing siblings such as
`foo-update-msa.json` are ignored when the source `foo.json` is present, so a
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

Same-token trunk batching gate: job `95368`
(`runs/same_token_auto_gate_20260702_145350`, commit `6962fda`) used two
40-token proteins with different side-chain atom counts (`polyA_len40`:
`N_atom=201`; `polyW_len40`: `N_atom=561`) and only `--batch_size 2`, relying
on the default `--batch_mode auto`.  The log shows one shared trunk batch:

```text
Predicting 2 same-token-trunk (auto) input(s) [seed:101]: N_token 40, N_atom 201-561, N_msa 1
```

Both CIFs and summaries were produced.  A paired `--batch_mode exact` run on
the earlier commit (job `95280`) correctly fragmented into two singleton batches
because the full feature trees had different atom shapes.  These tiny jobs are
functional gates, not speed claims: they are dominated by first-run compilation
and model initialization.  Representative throughput gates should use larger
same-token campaigns and warmed timing windows.

Different-token trunk batching gate: commit `0ca0852`, job `95432`
(`runs/var_token_length_sort_gate_20260702_152656`) used 64 shuffled
single-chain proteins spanning 40-220 tokens, `--batch_size 8`, `N_sample=1`,
and a short `N_step=1` scout workload.  The CLI wrote a transient length-sorted
campaign JSON before featurization:

```text
Merged and length-sorted 1 JSON file(s) for campaign inference into ...
Predicting 8 padded-token-trunk (auto) input(s) [seed:101]: N_token 184-196, token_pad_eff 96.9%, ...
```

The baseline input-order run at commit `4fe5e01`, job `95428`
(`runs/var_token_order_screen_20260702_152043`), mixed lengths such as
`N_token 40-220` inside one batch.  Automatic length sorting produced the same
tight token ranges as a manually sorted input file and improved batch-section
throughput on this scout gate:

| run | batch token ranges | batch-section sec | records/s | warm predict records/s |
| --- | --- | ---: | ---: | ---: |
| shuffled input order | `40-208`, `40-220`, ..., `52-196` | 32.94 | 1.94 | 3.10 |
| automatic length sort | `40-52`, `64-76`, ..., `208-220` | 12.15 | 5.27 | 8.16 |

The `token_pad_eff` log field is the quick health check for mixed-length
batches.  Values near 90-100% mean most pairformer work is on real tokens;
values much lower than that mean batches are still mixing lengths too broadly.

Mixed-token full-diffusion profile: commit `fdedb00`, job `95439`
(`runs/var_token_nstep200_profile_20260702_153834`) used 32 shuffled poly-A
proteins spanning 40-220 tokens, `--batch_size 8`, `N_sample=1`, full
`N_step=200`, confidence enabled, no MSA/template, and synchronized timing plus
diffusion CUDA-event subranges.  The CLI length-sorted the campaign into four
padded-token trunk batches:

| token range | `token_pad_eff` | predict sec | pairformer | diffusion | diffusion transformer | atom encoder | atom decoder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40-76 | 76.3% | 48.55 | 4.69 | 42.99 | 26.39 | 7.38 | 6.53 |
| 88-124 | 85.5% | 47.23 | 2.68 | 43.97 | 27.38 | 7.37 | 6.56 |
| 136-172 | 89.5% | 47.63 | 2.99 | 43.78 | 27.45 | 7.24 | 6.48 |
| 184-220 | 91.8% | 49.81 | 4.68 | 44.33 | 27.90 | 7.27 | 6.50 |

Totals: predict `193.22s`, diffusion `175.07s`, diffusion transformer
`109.11s`, atom encoder `29.26s`, atom decoder `26.07s`, pairformer `15.03s`,
dump `0.60s`.  Warm predict throughput was only `0.166 records/s` because the
diffusion tail still runs per record after the shared trunk.  This makes the
next optimization boundary clear: for the many-sequence, low-sample workload,
more pairformer fusion is no longer the main lever; batching or fusing the
diffusion tail is.

The first low-risk sub-boundary is the token-level diffusion transformer, not
fully padded atom attention.  Fake atom padding is unsafe without an atom
attention key mask because zero fake values still change local-attention softmax
denominators.  Commit `4c9bad4` therefore only adds dormant token-mask plumbing
through `DiffusionTransformer` and a hotspot screen,
`scripts/perf/diffusion_transformer_batch_probe.py`.  Job `95448`
(`runs/diffusion_transformer_batch_probe_20260702_154757`) compared 8 exact
sequential transformer calls with one padded masked batch on one H100:

| lengths | sequential median | padded masked batch median | isolated speedup | valid-prefix max abs diff |
| --- | ---: | ---: | ---: | ---: |
| 40,40,52,52,64,64,76,76 | 173.70 ms | 25.08 ms | 6.93x | 0.0033 |
| 88,88,100,100,112,112,124,124 | 186.36 ms | 26.60 ms | 7.01x | 0.0036 |
| 184,184,196,196,208,208,220,220 | 193.38 ms | 27.51 ms | 7.03x | 0.0034 |

This isolated screen does not prove an end-to-end win, but it strongly supports
the next implementation: keep atom encoder/decoder ragged and exact, batch only
the token diffusion transformer inside each diffusion step, then crop back to
each record before atom decoding.

Mixed-token batched diffusion-transformer gate: commit `bffbc94`, job `95569`
(`runs/mixed_token_batched_diff_n200_20260702_160545`) implemented that exact
boundary.  The sampler still runs diffusion conditioning, atom encoder, atom
decoder, confidence, and output dumping per record at true atom shape, but it
pads and masks the token activations immediately around the 24-block diffusion
transformer once per denoising step.  The timing source changes from
`token-trunk-batch` to `token-trunk+diffusion-transformer-batch`, which is the
quick log check that the mixed-token fast path is active.

Same input and flags as job `95439`, still `N_sample=1`, `N_step=200`,
confidence enabled, synchronized timing, and no MSA/template:

| token range | predict sec | diffusion | diffusion transformer | atom encoder | atom decoder |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40-76 | 25.54 | 19.51 | 4.15 | 6.75 | 5.87 |
| 88-124 | 22.35 | 19.42 | 4.11 | 6.74 | 5.88 |
| 136-172 | 22.74 | 19.23 | 4.09 | 6.67 | 5.81 |
| 184-220 | 24.71 | 19.51 | 4.13 | 6.90 | 5.81 |

Totals: predict `95.34s`, diffusion `77.67s`, diffusion transformer `16.48s`,
atom encoder `27.06s`, atom decoder `23.36s`, pairformer `15.22s`, dump
`0.61s`.  Relative to the immediately preceding mixed-token profile, this is
`2.03x` lower total predict time and `2.07x` higher warm predict throughput
(`0.166 -> 0.344 records/s`).  The targeted diffusion-transformer hotspot moved
by `6.62x` (`109.11s -> 16.48s`); after that win, the ragged atom
encoder/decoder plus remaining pairformer work are the next large terms.

This is why mixed token counts are supportable despite the original limitation:
the model can operate on variable lengths, but every padded boundary must prove
that fake entries are masked before softmax/reduction.  Token attention now has
that mask.  Atom-local attention still does not, so full atom padding remains
deliberately avoided rather than silently wrong.

A follow-up branch tried to merge source JSONs before MSA/template preprocessing
so preprocessing would run once for the transient campaign file instead of once
per source file.  That cleaned up generated JSON siblings but did not materially
move throughput: job `94969`, 64 one-record `7r6r` JSONs, current `main`
`197.54s` versus candidate `195.70s` (`1.009x`).  It also made preprocessing
failure granularity coarser, so it was not promoted.

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

Padded-token batching is not bitwise equivalent to singleton inference, so the
padding diagnostics remain important.  A corrected padding diagnostic snapshots
every stage because the pairformer uses in-place kernels during inference;
without those snapshots, an earlier row can be mutated by a later stage and
point to the wrong boundary.
`scripts/perf/pairformer_padding_breakdown.py` reports two separate quantities:
`*_valid_region` compares a true short tensor with the same valid crop embedded
in a larger zero-padded tensor, while `*_invalid_region_sensitivity` compares
zero-padded and random-invalid padded tensors with the same mask.  The former
finds physical-length-dependent numerical drift; the latter would find a real
mask leak.  `scripts/perf/pairformer_padding_mechanism.py` is the smaller
primitive-level reproducer for the same question: it runs triangle
multiplication, triangle attention, a per-cell pair transition, and
token attention independently so the first bad boundary is not hidden by the
rest of the block.  `scripts/perf/pairformer_padding_deep_dive.py` goes one
level deeper for triangle multiplication and records layer norm, projection,
masked projection, token-reduction matmul, output projection, and gating stages.
From the repository root:

```bash
PYTHONPATH=$PWD python scripts/perf/pairformer_padding_mechanism.py \
  --short-tokens 245 --full-tokens 384
PYTHONPATH=$PWD python scripts/perf/pairformer_padding_deep_dive.py \
  --short-tokens 245 --full-tokens 384
```

The exact broken boundary is not a mask leak.  Job `95118`
(`runs/padding_deep_dive_20260702_134126`, commit `7f9322d`) compared a true
245-token tensor, the same values embedded in a 384-token zero-padded tensor,
and the same padded tensor with random invalid rows/columns.  Invalid values
never changed the valid crop at any measured stage.  The first nonzero
valid-crop difference appears at `tri_mul_out.combine_token_reduction`, the
triangle-multiplication matmul whose reduction dimension is token length.  The
per-cell operations before it were exactly invariant:

| BF16 deep-dive stage, 245 -> 384 | valid max abs | valid mean abs | invalid-region sensitivity |
| --- | ---: | ---: | ---: |
| input, layer norm, A/B projections, A/B masked projections | 0 | 0 | 0 |
| `combine_token_reduction` | 0.0625 | 1.93e-7 | 0 |
| `layer_norm_out` | 0.015625 | 1.06e-7 | 0 |
| `output_projection` | 0.00390625 | 1.48e-7 | 0 |
| final triangle-multiplication output | 0.001953125 | 7.45e-8 | 0 |

Precision matters because the failure is accumulation schedule, not masking:

| isolated op / mode | valid max abs | valid mean abs | invalid-region sensitivity | note |
| --- | ---: | ---: | ---: | --- |
| triangle multiplication, BF16 CUEQ | 0.001953125 | 9.46e-8 | 0 | first drift at token-reduction matmul |
| triangle multiplication, FP32 with TF32 allowed | 7.57e-5 | 1.05e-7 | 0 | TF32 matmul still schedule-dependent |
| triangle multiplication, FP32 with TF32 disabled | 2.38e-7 | 2.80e-8 | 0 | CUEQ residual-scale noise only |
| token attention, BF16 SDPA | 4.88e-4 | 1.09e-7 | 0 | no key/value leakage |
| token attention, FP32 with TF32 allowed | 9.10e-5 | 1.25e-5 | 0 | SDPA schedule-dependent |
| token attention, FP32 with TF32 disabled | 8.20e-8 | 1.20e-8 | 0 | near roundoff |

One padded 245-token block inside a 384-token tensor therefore starts with tiny
differences, but BF16 differences compound through the full trunk.  On a
48-block random pairformer stack in the same job:

| mode | valid `s` mean abs | valid `s` max abs | valid `z` mean abs | valid `z` max abs |
| --- | ---: | ---: | ---: | ---: |
| FP32 CUEQ, TF32 disabled, 245 -> 384 tokens | 1.61e-5 | 1.18e-4 | 7.34e-6 | 7.49e-5 |
| BF16 CUEQ, 245 -> 384 tokens | 0.2350 | 1.46875 | 0.1682 | 1.5625 |

So the actionable rule is: exact-shape batching is bitwise safest; masked
padded-token trunk batching is numerically different because the trunk sees a
different physical reduction length.  The promoted inference path accepts that
small model-level drift for throughput, sorts records into tight length batches,
passes masks through all token-trunk readers, and keeps atom-shaped diffusion and
confidence work ragged.  A future custom kernel could reduce only over each
sequence's real length while still sharing a launch, but that is a larger kernel
project.

The first broken invariant is therefore not "masked values leak"; it is
"valid-crop output is invariant to physical token length".  The first nonzero
boundary is the first triangle multiplication, whose reduction over the
third/residue dimension changes from `N=245` to `N=384` even though the extra
terms are masked to zero.  Token attention can add the same class of drift
because SDPA also chooses schedules from the physical key length.  Pair and
single transitions are not the root cause; they consume already-different
activations and amplify them through LayerNorm, SiLU/gates, residual updates,
and the 48-block trunk.

So zeroing or masking harder is not the fix; the masks already block padded
content.  The problem is that cuDNN SDPA, CUEQ, and related matmul reductions are
not bitwise invariant to padded physical length.  The runner therefore exposes
masked padded-token batching as the default throughput path for campaign runs,
with `--batch_mode exact --batch_size 1` remaining available for strict
singleton/parity debugging.

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
`/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/ncu`.  The
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

CuTe/CUTLASS extension toolchain status on Tokyo:

| item | value |
| --- | --- |
| Python/runtime env | `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2` |
| CUDA toolkit / `nvcc` | `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/kaveh` |
| CUTLASS/CuTe headers | `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-cutlass/include` |
| Nsight Compute | `/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/ncu` |

The toolchain was smoke-tested in
`cute_toolchain_smoke_20260702_124554`, Slurm job `95060`: a PyTorch CUDA
extension compiled for `sm_90`, included `<cute/tensor.hpp>` and
`<cutlass/cutlass.h>`, linked against the CUDA 13 runtime, launched on an H100,
and returned the expected result.  Future native-kernel screens should use this
environment rather than installing another compiler stack.

For Hopper GMMA kernels, `sm_90` is not enough.  CUTLASS emits architecture-
conditional GMMA instructions that must be compiled as `sm_90a`; otherwise the
kernel can compile and load but trip a device-side assert at runtime.  The Tokyo
transition-epilogue harness now defaults to `TORCH_CUDA_ARCH_LIST=9.0a`, and the
CUTLASS candidate import path fails early if a non-`90a` arch list is supplied.

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

A first SM90 CUTLASS EVT version now exists as an isolated benchmark candidate:
`scripts/perf/transition_output_epilogue_cutlass_candidate.py` plus
`scripts/perf/transition_output_epilogue_cutlass_sm90.cu`.  It is intentionally
not wired into model inference.  It demonstrates the correct fusion boundary and
argument mapping:

- `M = N_sample`, `N = c_a`, `K = hidden`, `L = N_token`;
- `b[:, token, :]` is the A operand;
- `weight[c_a, hidden]` is used as a shared B operand across token batches;
- `gate[0, token, c_a]` is an auxiliary epilogue load with `stride_m=0`; and
- the visitor tree computes `residual + sigmoid(gate) * accumulator`.

Job `95156`
(`transition_epilogue_cutlass_sm90a_prebuilt_20260702_141033`, commit
`1732bd5`) compiled the candidate for `sm_90a` and ran the warmed H100 screen.
It is a correctness success but a performance rejection:

| implementation | boundary ms | effective TFLOP/s | parity max / mean | decision |
| --- | ---: | ---: | ---: | --- |
| cuBLAS GEMM + promoted Triton epilogue | 1.464 | 731.7 GEMM-only | reference | keep |
| CUTLASS SM90 EVT fused epilogue | 2.374 | 505.2 boundary | 0.03125 / 1.42e-4 | reject |

The fused kernel removes the projected store/reload on paper, but its generic
CUTLASS mainloop/EVT path loses too much tensor-core throughput.  This confirms
the core constraint for any future CuTe kernel: preserve the cuBLAS-class GMMA
schedule first, then fuse the epilogue.  A slower matmul with a nicer epilogue is
a net loss.

The follow-up mainloop isolation probe separates the GEMM schedule from the
gate/residual epilogue: `scripts/perf/transition_output_cutlass_mainloop_probe.py`
and `scripts/perf/transition_output_cutlass_mainloop_probe_sm90.cu` time only
`b @ weight.T` with the same BF16 tensors.  It compares PyTorch/cuBLAS's
flattened large GEMM (`M=N_sample*N_token`) with two CUTLASS mappings: the same
flattened shape, and the token-batched shape needed by the fused gate epilogue
(`M=N_sample`, `L=N_token`).  Job `95168`
(`transition_output_cutlass_mainloop_probe_20260702_1425`, commit `6dd2418`)
used the prebuilt `sm_90a` extension cache from
`transition_output_cutlass_mainloop_compile_6dd2418_sm90a`:

| implementation | GEMM shape | ms | effective TFLOP/s | parity | decision |
| --- | --- | ---: | ---: | ---: | --- |
| PyTorch/cuBLAS | `M=313600, N=768, K=1536` | 0.976 | 757.8 | reference | keep |
| CUTLASS SM90 flattened | `M=313600, N=768, K=1536, L=1` | 2.176 | 340.1 | exact BF16 | reject |
| CUTLASS SM90 token-batched | `M=1280, N=768, K=1536, L=245` | 1.956 | 378.3 | exact BF16 | reject |

This makes the rejection sharper: the first EVT candidate was not only paying
for a complicated epilogue.  The generic CUTLASS collective/tile chosen here is
already about 2x slower than cuBLAS on the plain transition GEMM.  The transition
output boundary should not be touched in model code until a CUTLASS/CuTe
mainloop reaches cuBLAS-class throughput on this exact shape; otherwise fusing
the 0.47 ms epilogue cannot recover the lost GEMM time.

`scripts/perf/transition_output_epilogue_hotspot.py` is the reusable screen for
the next attempt.  It builds a real `ConditionedTransitionBlock`, fixes the
output-boundary tensors, times the baseline `linear_b(b)` plus promoted
`fused_sigmoid_mul_add`, and optionally imports a candidate op with signature
`fn(b, weight, gate, residual) -> out`.  Use it before touching model code:

```bash
PROTENIX_REPO=/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile \
PROTENIX_RUN_NAME=transition_epilogue_baseline_$(date -u +%Y%m%d_%H%M%S) \
sbatch scripts/perf/tokyo_transition_epilogue_hotspot.sbatch
```

To test a native extension candidate, pass the candidate function as an
environment variable:

```bash
CANDIDATE=/path/to/native_epilogue_candidate.py:transition_output_epilogue \
PROTENIX_RUN_NAME=transition_epilogue_candidate_$(date -u +%Y%m%d_%H%M%S) \
sbatch scripts/perf/tokyo_transition_epilogue_hotspot.sbatch
```

Before a native extension is ready, smoke-test the import/call path with the
reference candidate:

```bash
CANDIDATE=scripts/perf/transition_output_epilogue_reference_candidate.py:transition_output_epilogue \
PROTENIX_RUN_NAME=transition_epilogue_reference_candidate_$(date -u +%Y%m%d_%H%M%S) \
sbatch scripts/perf/tokyo_transition_epilogue_hotspot.sbatch
```

The wrapper writes a clean `$RUN_DIR/result.json` directly from the Python
harness, so extension compile logs on stdout do not corrupt the machine-readable
timing record.  The JSON also includes a small roofline sanity block under
`problem`: GEMM FLOPs, lower-bound boundary bytes, effective GEMM TFLOP/s,
standalone-epilogue bandwidth, removable projected store/reload bytes, and the
ideal speedup from deleting only the current epilogue launch.  These are lower
bounds, not replacements for Nsight Compute, but they keep the native-kernel
gate honest: the candidate must beat the full baseline boundary, not just claim
to remove memory traffic on paper.

A candidate should only advance if this screen shows both acceptable BF16 parity
and a win over the full baseline boundary, not merely over the elementwise
epilogue launch in isolation.

Latest warmed one-H100 roofline screen, commit `0cba4ac`, `N_sample=1280`,
`N_token=245`, BF16, `gate_batch=1`:

| job | run | boundary ms | GEMM ms | epilogue ms | GEMM TFLOP/s | epilogue TB/s lower bound | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `95081` | `transition_epilogue_roofline_warm_baseline_20260702_132714` | 1.448 | 1.012 | 0.474 | 731 | 3.05 | baseline |
| `95082` | `transition_epilogue_roofline_warm_reference_20260702_132714` | 1.468 | 1.027 | 0.474 | 721 | 3.05 | reference candidate 1.482 ms, exact parity, 0.991x |
| `95156` | `transition_epilogue_cutlass_sm90a_prebuilt_20260702_141033` | 1.464 baseline / 2.374 candidate | 1.011 baseline | 0.473 baseline | 731 baseline / 505 candidate boundary | 3.06 baseline | CUTLASS EVT candidate valid but 0.617x, reject |

The lower-bound byte model reports 2.90 GB of logical boundary traffic for the
baseline and 1.93 GB for a fused-epilogue GEMM.  The removable projected
store/reload is 0.96 GB.  The ideal "delete only the current epilogue launch"
boundary would be about 0.97-0.99 ms, a 1.48x speedup for this boundary.  That
does **not** mean the full model can move by 1.48x: it is a 0.47 ms saving on
one transition-output boundary, and any native CuTe implementation has to keep
roughly 720-730 TFLOP/s GEMM throughput or it will lose the whole win.

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

The q/k/v layout producer itself was retuned after the README usage guide was
merged (`main` commit `9d11636`).  A one-H100 isolated screen compared Triton
block sizes `128-8192` at B32/B64/B96, `N_token=245`, BF16, `no_heads=4`,
`head_dim=32`.  Block size `256` was consistently best and preserved exact
parity against the original view/transpose/contiguous layout, but the absolute
gain over the current `1024` default was small:

| batch | best block | best mean ms | current mean ms | isolated gain |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 256 | 1.022 | 1.066 | 1.042x |
| 64 | 256 | 2.027 | 2.103 | 1.037x |
| 96 | 256 | 3.000 | 3.122 | 1.041x |

That saves only about `0.04-0.12` ms per q/k/v layout launch, so the expected
full-inference movement is below noise once this launch is embedded in two
triangle attentions per pairformer block and the diffusion/confidence/output
path.  Do not add a user-facing knob or change the default for this alone; it
is a useful implementation detail only if a future fused-attention boundary
reuses the same producer.

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

The pairformer orientation boundary was screened after the exact-shape campaign
path was promoted.  The existing block implements ending-node triangle attention
by physically transposing the full pair tensor before and after the ending
attention call:

```python
z = z.transpose(-2, -3).contiguous()
z += tri_att_end(z, ...)
z = z.transpose(-2, -3).contiguous()
```

At B32/B64 those two explicit copies are measurable but small, about 3.35% of a
full pairformer block:

| batch | block ms | explicit transpose ms | transpose share | triangle attention share | pair transition share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 39.87 | 1.34 | 3.36% | 44.5% | 21.1% |
| 64 | 78.92 | 2.64 | 3.35% | 44.6% | 21.3% |

A logical-ending prototype used `TriangleAttention(starting=False)` and added
the returned non-contiguous ending update into the original contiguous `z`,
removing the two explicit pair-tensor copies.  It was bitwise exact for one
block at B32/B64/B96, but the full-block speedup was only `1.0076-1.0089x`.
This is below the promotion threshold and confirms that the remaining "meat" is
inside CUEQ attention, triangle multiplication, and transition GEMM epilogues,
not the outer orientation copies.

Run directories:

- `pairformer_orientation_screen_20260702_123911`, Slurm job `95054`
- `pairformer_logical_ending_screen_20260702_124119`, Slurm job `95058`

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
| CUEQ q/k/v layout block-size retuning | block `256` beat the current `1024` by 3.7-4.2% in isolation, but only saves `0.04-0.12` ms per layout launch | do not promote tiny tile retunes without a full-gate signal |
| Logical ending-node triangle attention without explicit pair transposes | bitwise exact one-block screen but only `1.0076-1.0089x` at B32/B64/B96 | explicit pair-tensor orientation copies are measurable but not the dominant pairformer boundary |
| Default BF16 full-token attention after the promoted stack | isolated trunk-attention hotspot improved, but full exact-shape gates moved only +0.16% at B64 and +0.27% at B96 | do not promote dtype-policy changes unless the representative throughput gate moves, not just the local kernel |
| Inference DataLoader workers for exact-shape batches | B8 had a small screen win, but B32 hit tensor-sharing failures unless switched to slower file-system sharing; clean B32 no-worker batching was better | do not add host-pipeline complexity unless it survives the actual many-input gate |
| CUDA graph capture of the denoiser | slower in this setup | graph overhead/constraints can outweigh launch savings |
| Merging campaign JSONs before preprocessing | 64-file directory gate moved only from 197.54 s to 195.70 s (`1.009x`) and worsened preprocessing failure granularity | the remaining e2e gap is not mainly per-file JSON preprocessing |
| Naive padded mixed-shape pairformer batching | valid-region differences accumulated over 48 blocks: FP32 `s/z` mean abs `0.012/0.006`, BF16 `0.236/0.168` for 245 -> 384 tokens | do not claim bitwise parity; the promoted path masks all token readers, sorts by length to reduce physical-shape drift/waste, and keeps exact mode available |

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
- `runner/campaign_inputs.py`, `runner/inference.py`,
  `runner/batch_inference.py`, and `configs/configs_inference.py`: public
  campaign inference batching.  The CLI length-sorts raw campaign records before
  featurization, the runner uses full-model batches for identical feature tensor
  trees, and the trunk path supports both same-token variable-atom records and
  masked padded-token records before writing one output per input.

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
- `scripts/perf/transition_output_epilogue_hotspot.py`: transition output-GEMM
  epilogue screen and candidate-injection harness for native CuTe/CUTLASS work.
- `scripts/perf/transition_output_epilogue_reference_candidate.py`: baseline
  candidate ABI smoke for the transition epilogue harness.
- `scripts/perf/tokyo_transition_epilogue_hotspot.sbatch`: Tokyo one-H100
  wrapper for the transition epilogue screen, including CUDA/CUTLASS env setup.
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
