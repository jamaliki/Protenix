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

The BF16 diffusion core and the BF16/Triton atom local-attention boundary are
now defaults on BF16-capable CUDA devices; they are shown explicitly above so a
benchmark log captures the full promoted stack.  Set
`PROTENIX_BF16_DIFFUSION_CORE=0`, `PROTENIX_BF16_ATOM_ATTENTION=0`, or
`PROTENIX_TRITON_LOCAL_ATTN=0` when running conservative numerical audits.

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
  atom-independent trunk is batched.  The diffusion tail can still batch its
  atom encoder/decoder by padding atoms and passing a real atom mask.
- If token counts differ, the CLI first length-sorts the raw JSON records, then
  pads only token-trunk tensors inside each batch.  A real `pair_mask` is passed
  through template, MSA, and pairformer blocks.  Diffusion then pads both token
  activations and atom activations at the transformer boundaries, with separate
  token and atom masks.

This fixes same-length mutation campaigns without letting fake atoms participate
in atom attention or confidence reductions, and it also supports the common
different-length design campaign case.  Padded batching is a controlled
numerical-change feature, not a bitwise singleton reproduction path: the masks
block padded content, but BF16/TF32 reductions can still differ slightly because
kernels see a different physical token/atom length.  Use
`--batch_mode exact --batch_size 1` for bitwise singleton debugging.

If the pairformer padded-token drift is unacceptable but you still want some
low-sample batching for different token counts, use:

```bash
protenix pred ... --batch_size 16 --batch_mode trunk_exact
```

This runs the input embedding and pairformer trunk once per record at each
record's real token length, then batches the diffusion tail using the same
masked token/atom boundaries as the high-throughput path.  It is deliberately
named `trunk_exact`, not `exact`: it protects the largest schedule-sensitive
pairformer boundary while preserving useful denoising throughput, but the
batched diffusion tail still uses padded masked kernels and therefore should be
validated against the tolerance required by the campaign.  Directory inputs are
handled as one campaign: all preprocessed JSON records are merged into a
short-lived, length-sorted file before inference so many one-record JSONs can
still fill batches.

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
that fake entries are masked before softmax/reduction.  The token transformer
uses a token key mask.  The atom transformer uses a separate atom key mask in
the local 32-query/128-key attention trunks, zeros fake atom states before and
after local attention, and excludes fake atoms from atom-to-token means.  Padding
is therefore a GPU scheduling tool, not extra biological input.

Mixed-token batched diffusion atom-transformer gate: commit `eea3464`, paired
jobs `95606` and `95607`
(`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile/runs/ptx_atom_{off,on}_n200_20260702_163818`)
ran the same 32-protein, 40-220-token, `N_sample=1`, `N_step=200` workload on
the same H100 node and commit.  The only difference was
`PROTENIX_BATCH_ATOM_TRANSFORMER=0` versus `1`.

| setting | job sec | predict sec sum | diffusion | atom encoder | diffusion transformer | atom decoder | warm records/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| atom batching off | 105.89 | 97.37 | 80.03 | 28.23 | 16.87 | 24.17 | 0.337 |
| atom batching on | 60.97 | 52.28 | 34.90 | 4.66 | 17.14 | 3.92 | 0.640 |

The paired end-to-end job speedup is `1.74x` over the immediately preceding
token-diffusion batching path.  The diffusion tail itself moved by `2.29x`, and
the atom encoder/decoder terms moved by about `6.1x` each.  Relative to the
original mixed-token profile at job `95439`, warm throughput improved from
`0.166` to `0.640 records/s` (`3.86x`), and summed predict time dropped from
`193.22s` to `52.28s` (`3.70x`).  Good logs now report
`token-trunk+diffusion-token-atom-batch`; setting
`PROTENIX_BATCH_ATOM_TRANSFORMER=0` leaves the older
`token-trunk+diffusion-transformer-batch` path available for bisects.

Mixed-token batched diffusion-conditioning gate: commit `bf00b4a`, paired job
`95624`
(`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile/runs/cond_batch_pair_b16_n200_20260702_165714`)
ran the same 32-protein, 40-220-token, `N_sample=1`, `N_step=200` workload at
`--batch_size 16`.  The only difference was
`PROTENIX_BATCH_DIFFUSION_CONDITIONING=0` versus `1`; token-transformer and atom
batching stayed enabled in both cases.

| setting | predict sec sum | records/s by predict | pairformer | diffusion | conditioning | atom encoder | diffusion transformer | atom decoder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditioning batching off | 37.17 | 0.861 | 14.72 | 20.39 | 3.28 | 2.45 | 8.82 | 1.89 |
| conditioning batching on | 33.93 | 0.943 | 14.67 | 17.22 | 0.33 | 2.32 | 8.84 | 1.91 |

The conditioning term moved by `9.9x` (`3.28s -> 0.33s`) and the representative
predict section moved by `1.10x` over the immediately preceding promoted path.
Relative to the original mixed-token profile at job `95439`, summed predict
time is now `193.22s -> 33.93s` (`5.70x`) and throughput is
`0.166 -> 0.943 records/s` (`5.69x`).  This gate also explains the current
optimization frontier: after conditioning batching, the remaining material
terms are the pairformer trunk (`14.67s`), the diffusion transformer (`8.84s`),
and the batched atom encoder/decoder (`4.22s` combined).  Further throughput
work should target those boundaries rather than the now-small conditioning
path.

Low-sample ragged diffusion follow-up: branch
`codex/ragged-multisample-diffusion` extends the mixed-token diffusion batching
boundary from `N_sample=1` to the practical campaign range `N_sample=2-5`.
The key shape rule is that `(record, sample)` is flattened only for token
activations and masks, preserving the rank-4 BF16/cuDNN attention path; the
sample-invariant pair tensor `z` stays at record grain and only the smaller
projected attention bias is repeated across sample lanes.  This avoids both
the old per-record diffusion tail and the wasteful `N_sample` repetition of
pair-bias layernorm/projection work.  The path is guarded by
`PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES` (default `5`) so large single-design
sampling jobs do not accidentally multiply memory by the protein batch.

Status: running, not promoted.  The first queued shape smoke/gate chain,
jobs `95868`/`95869`, was canceled before running because later diagnostics
found a real sample-axis shape bug and made the original gate untrustworthy.
The representative gate still uses the same 32 mixed 40-220-token proteins as
job `95624`, `N_sample=5`, `N_step=200`, `--batch_size 16`, confidence enabled,
no MSA/template, and compares the old low-sample boundary against flattened and
explicit-sample-axis variants.  Earlier queued jobs `95841` and `95848` were
canceled before running because their spooled scripts did not enable the fair
broadcast-conditioning baseline.  Parse the completed gate with:

```bash
python scripts/perf/parse_batch_gate_log.py \
  runs/sample_axis_b16s5_gate_20260702_215232_9eb2197 \
  --samples-per-input 5
```

While waiting for that gate, code inspection exposed a fusion-boundary issue in
the optional Triton atom-local-attention path for `N_sample>1`.  Batched atom
attention has activations shaped `[record, sample, ...]`, but the cached pair
bias is sample-invariant and therefore shaped `[record, 1, heads, trunks, 32,
128]`.  The old Triton wrapper tried to expand that bias to `[record, sample,
...]` and then flatten it as a view.  PyTorch cannot represent that particular
broadcast flatten without a copy, so the guard rejected the Triton kernel and
fell back to the rank-5 PyTorch local-attention path.  Commit `506a418` keeps
the compact bias and passes a `bias_sample_repeat` stride rule into the Triton
kernel instead.  This is a kernel-boundary fix, not a promoted throughput claim
yet.  The first isolated hotspot job, `95881`
(`runs/local_attn_bcast_b16s5_20260702_200825`), was canceled with the obsolete
pre-fix queue.  Re-run this exact `B=16`, `N_sample=5`, broadcast-bias
atom-attention shape only if the representative model gate shows atom-local
attention remains a material low-sample campaign bottleneck.

Commit `84f08ec` fixes the adjacent mask boundary.  Ragged atom batching also
passes an atom key mask shaped `[record, atom]`; the old mask add expanded that
mask to q/k/v's full `[record, sample, atom]` prefix before adding it to the
attention bias.  That was correct numerically but defeated the compact-bias
kernel path by materializing the sample axis.  The mask now expands to the
existing bias prefix, usually `[record, 1]`, so both pair bias and key mask stay
sample-invariant.  The corresponding masked hotspot job, `95891`
(`runs/local_attn_masked_bcast_b16s5_20260702_201338`), was also canceled with
the obsolete pre-fix queue; it remains the right isolated boundary if atom
attention is still worth attacking after the model gate.

The analogous token-attention question is whether we can avoid explicitly
repeating the projected pair bias across diffusion samples.  The current
ragged `N_sample>1` path flattens q/k/v to `[record * sample, heads, tokens,
head_dim]`, keeps pair-bias projection at record grain, then repeats the
smaller `[record, heads, tokens, tokens]` bias over sample lanes.  That repeat
is much cheaper than re-projecting `z`, but it is still HBM traffic every
diffusion block.  `scripts/perf/token_attention_bias_broadcast_probe.py`
therefore compares the current flattened SDPA call against natural rank-5
`[record, sample, heads, tokens, head_dim]` SDPA with stride-0 or broadcasted
sample-invariant pair and padding-key bias.  The `--valid-fraction` option
makes the mask case representative of sorted mixed-token batches instead of
only the all-valid ideal.  It also times a rank-5 FP32-upcast case because
Protenix's current generic attention helper conservatively upcasts rank-5 q/k/v
for atom local-attention safety.  If rank-5 BF16/cuDNN is fast, the model change
must also relax that policy only for full token attention.  If rank-5 itself
falls back to a slow path, the right next boundary is a custom token-attention
kernel that indexes bias by `record = flat_batch // N_sample` without
materializing repeated bias rows.

Masked probe status: superseded.  The queued job `95919`
(`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile/runs/token_attn_bias_broadcast_masked_20260702_203322`),
was canceled with the obsolete pre-fix dependency chain.  The same mechanism
should be revisited only after the representative `N_sample=5` gate proves the
model-level sample-axis path is worth promoting.  The intended probe uses
`--valid-fraction 0.8` at `tokens=124` and `220`, so it measures the
padding-mask case that actually matters for sorted mixed-token batches.

`scripts/perf/diffusion_transformer_sample_axis_probe.py` is the corresponding
one-block probe.  Raw SDPA can answer whether rank-5 broadcast attention is
available, but the model-level decision needs the q/k/v projections, attention
gate, residual, and conditioned transition too.  That script compares the
current flattened `[record * sample, token, channel]` block against a rank-5
candidate that keeps `a` as `[record, sample, token, channel]` and keeps the
diffusion conditioning `s` sample-invariant as `[record, 1, token, channel]`.
It also includes the current Protenix rank-5 FP32-upcast policy as a rejection
control.  Status: superseded.  The queued job `95925`
(`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile/runs/diffusion_transformer_sample_axis_20260702_203922`),
was canceled with the obsolete pre-fix dependency chain.

Implementation candidate `40e51f8` adds the corresponding model path behind
two opt-in flags:

```bash
export PROTENIX_DIFFUSION_TRANSFORMER_SAMPLE_AXIS=1
export PROTENIX_RANK5_FULL_ATTENTION_BF16=1
```

The first flag keeps the low-sample diffusion transformer activation as
`[record, sample, token, channel]` and passes the token key mask as
`[record, 1, token]`, so the projected pair and padding bias can be consumed as
`[record, 1, head, token, token]` instead of being materialized once per sample.
The second flag relaxes Protenix's conservative rank-5 FP32 attention policy
only for **full** token attention (`Q == K`); atom local attention still stays on
its original guarded path.  This is deliberately not default-on yet: it changes
the fusion boundary and needs a GPU gate before promotion.  Remote CPU checks in
the `env-boltz2` environment passed the sample-invariant and explicit
sample-axis parity tests plus the ragged-batch helper tests.
Follow-up commit `aa899d3` fixes the model integration boundary so the
conditioning tensor is passed as `[record, 1, token, channel]`; the original
implementation correctly tested the transformer module with that shape, but the
full diffusion batching path accidentally passed `[record, token, channel]`,
which would broadcast against the sample axis incorrectly.

Candidate gate status: fixed, validation pending.  The isolated checkout is
`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_sampleaxis_40e51f8`.
The first queued pair, jobs `95935`/`95936`, was canceled before running because
the script accidentally enabled the large-sample Triton elementwise/residual
flags that job `95635` had already rejected for mixed B16 campaigns.  Job
`95940` and then `95948` failed before useful model execution because a fresh
checkout tried to JIT-build the fast layernorm extension against an incomplete
CUDA runtime.  Commit `17f6fcd` added a safe torch-layernorm fallback and commit
`eafffad` added `PROTENIX_DISABLE_FAST_LAYER_NORM=1` so canaries can run even
when the extension build is unavailable.

The next fallback canary pair, jobs `96025`/`96026`, deliberately used
`PROTENIX_DISABLE_FAST_LAYER_NORM=1` and a run-local extension directory.
`96025` reached the real mixed-token path, but the batched forward failed and
fell back to singleton inference:

```text
Predicting 2 padded-token-trunk (auto) input(s): N_token 136-172, token_pad_eff 89.5%
Batched prediction ... failed; falling back to singleton inference.
Error: The size of tensor a (2) must match the size of tensor b (16)
```

The job still exited successfully because fallback is the production safety
path, so representative gate `96026` was canceled before it could measure
singleton fallback noise.  Commit `75ee498` made fallback failures auditable by
logging the full traceback and allowing `PROTENIX_RAISE_ON_BATCH_FALLBACK=1`.
The loud rerun `96040` localized the error to rank-5 diffusion-transformer SDPA.
A raw H100 SDPA probe (`96049`,
`runs/sdpa_rank5_shape_probe_20260702_214049`) showed that the intended shape
`q=[B, sample, head, token, dim]` with
`bias=[B, 1, head, token, token]` is accepted by PyTorch 2.11/cuDNN, so this
was not an inherent SDPA limitation.

Root cause: the model integration added an extra singleton to the conditioning
tensor.  Batched diffusion conditioning already returns
`s_batch=[record, 1, token, channel]`; the sample-axis branch passed
`s_batch[:, None, :, :]`, creating `[record, 1, 1, token, channel]`.  AdaLN
broadcast that into a spurious extra record axis, so SDPA saw the sample
dimension where the attention-head dimension should be.  Commit `6cf7a2e`
passes `s_batch` directly and documents the shape invariant.  Remote CPU/import
tests in `env-boltz2` pass for `tests.test_campaign_json_batching`,
`tests.test_ragged_multisample_diffusion`, and `tests.test_diffusion_transformer`.
Commit `cc5b7ee` also makes `PROTENIX_RAISE_ON_BATCH_FALLBACK=1` propagate past
the outer per-record error handler, so diagnostic smokes now fail the Slurm job
instead of writing an error file and exiting zero.  The corrected GPU smoke,
job `96061` (`runs/sample_axis_shape_fix_smoke_20260702_214538_cc5b7ee`),
completed successfully with `PROTENIX_RAISE_ON_BATCH_FALLBACK=1` and logged a
real two-record mixed-token batched forward:

```text
Batch model timing total (token-trunk+diffusion-token-atom-batch, 2 input(s), predict 9.47s)
Finished 2/2 input(s) ... in 9.56s
```

Because fallback escalation was enabled, that job would have failed if the
batch path silently fell back to singleton inference.

The representative paired gate completed as job `96081`
(`runs/sample_axis_b16s5_gate_20260702_215232_9eb2197`) on `gpu-canary`.  It
used the same 32 mixed 40-220-token proteins with `N_sample=5`, `N_step=200`,
`--batch_size 16`, confidence enabled, no MSA/template, and compared four cases
on one H100.  This canary used `PROTENIX_DISABLE_FAST_LAYER_NORM=1` because the
CUDA-13 fast-layernorm extension still fails to compile against the current
PyTorch headers, so treat this as a strong within-job relative gate rather than
a clean absolute comparison to older fast-layernorm runs.

1. `old_low_sample_boundary`: `PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES=1`,
   default full-attention FP32 policy.
2. `flat_sample_lanes_fp32attn`: current flattened low-sample path,
   default full-attention FP32 policy.
3. `flat_sample_lanes_bf16attn`: current flattened low-sample path with
   `PROTENIX_ATTENTION_FORCE_FP32=0`.
4. `explicit_sample_axis_bf16attn`: explicit sample-axis path with both
   `PROTENIX_ATTENTION_FORCE_FP32=0` and
   `PROTENIX_RANK5_FULL_ATTENTION_BF16=1`.

Results from the sample-axis gate, parsed with
`scripts/perf/parse_batch_gate_log.py --samples-per-input 5`, plus the later
promoted LayerNorm gate:

| case | predict sec | input records/s | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old low-sample boundary | 212.51 | 0.151 | 0.753 | 26.87 | 171.86 | 103.36 | 12.36 | baseline for this gate |
| flat sample lanes, FP32 attention | 76.81 | 0.417 | 2.083 | 23.44 | 43.49 | 26.59 | 8.32 | promoted mechanism |
| flat sample lanes, BF16 attention | 73.34 | 0.436 | 2.182 | 21.88 | 39.99 | 23.14 | 9.94 | best robust setting |
| explicit sample axis, BF16 attention | 72.50 | 0.441 | 2.207 | 22.07 | 44.26 | 27.45 | 4.58 | not promoted beyond opt-in |
| flat sample lanes, BF16 attention, forced Triton LayerNorm fallback | 58.34 | 0.549 | 2.743 | 16.45 | 37.15 | - | 2.50 | best paired low-precision fallback |
| flat sample lanes, BF16 attention, default Triton LayerNorm fallback | 61.41 | 0.521 | 2.605 | 16.62 | 40.81 | - | 2.55 | promoted default repeat at commit `95fb258` |

The large, trustworthy movement is the flattened low-sample diffusion boundary:
`212.51s -> 73.34s` predict (`2.90x`) and `0.753 -> 2.182` generated samples/s
for `N_sample=5`.  Allowing full token attention to remain BF16
(`PROTENIX_ATTENTION_FORCE_FP32=0`) adds a smaller `1.05x` over the flattened
FP32-attention path.  The explicit rank-5 sample-axis path is not a promoted
default despite the best total time (`72.50s`, another `1.2%`): the targeted
diffusion-transformer subrange got slower (`23.14s -> 27.45s`) and the total
movement is dominated by noisier confidence/pairformer differences.  Keep
`PROTENIX_DIFFUSION_TRANSFORMER_SAMPLE_AXIS=1` and
`PROTENIX_RANK5_FULL_ATTENTION_BF16=1` as diagnostic opt-ins until a custom
attention/bias kernel beats the flattened BF16 boundary on the transformer
itself.

The next representative promotion was the Triton LayerNorm fallback.  The final
`N_sample=5`, `N_step=200`, batch-16 gate was job `96155`
(`runs/layer_norm_model_gate_n200_20260702_225403_f95fc13`) on one H100 with
the same 32 mixed 40-220-token input.  Within the same job, forcing the fallback
off gave `76.17s` predict and `2.101` generated samples/s, while forcing it on
gave `58.34s` predict and `2.743` generated samples/s.  Relative to the old
low-sample boundary this moves the practical mixed-sequence `N_sample=5`
throughput from `0.753` to `2.743` generated samples/s (`3.64x`).  The movement
was concentrated where expected: pairformer fell `24.25s -> 16.45s`,
confidence fell `9.42s -> 2.50s`, and diffusion moved more modestly
`41.06s -> 37.15s`.  The implementation therefore defaults Triton LayerNorm on
only for FP16/BF16 no-grad CUDA inference when the C++ fast LayerNorm extension
is unavailable; FP32 stays opt-in because several FP32 isolated shapes were
slower.

After making that policy the default, job `96183`
(`runs/layer_norm_default_gate_n200_20260702_230327_95fb258`) reran the same
N200 gate at commit `95fb258` with `PROTENIX_TRITON_LAYER_NORM` unset for the
candidate case.  The CUDA unit test suite passed inside the allocation
(`5 tests OK`).  The paired model result was `75.30s -> 61.41s` predict and
`2.125 -> 2.605` generated samples/s versus forced-off LayerNorm.  This is
slower than the prior forced-on repeat because diffusion was noisier, but it is
the conservative current-default number: `212.51s -> 61.41s` (`3.46x`) and
`0.753 -> 2.605` generated samples/s (`3.46x`) over the old low-sample
boundary.

Job `96275`
(`runs/bf16_diffusion_core_gate_n200_20260703_003400_8b51580`) then tested the
next profile-driven precision change.  The post-LayerNorm trace showed the
batched diffusion transformer's largest GPU bucket was FP32/TF32 GEMM work, so
the gate compared FP32 versus BF16 diffusion-core activations while keeping the
same 32 mixed 40-220-token records, `N_sample=5`, `N_step=200`, batch size 16,
confidence enabled, and BF16 full attention.  It also crossed the old
BF16/FP16-only LayerNorm policy with the new shape-gated FP32 LayerNorm policy:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 LayerNorm policy, FP32 diffusion core | 61.78 | 2.590 | 16.53 | 41.23 | 24.28 | 2.47 | baseline repeat |
| BF16 LayerNorm policy, BF16 diffusion core | 51.40 | 3.113 | 16.62 | 30.83 | 13.99 | 2.43 | core precision win |
| shape-gated LayerNorm, FP32 diffusion core | 60.92 | 2.626 | 16.55 | 40.39 | 23.54 | 2.46 | LayerNorm shape gate alone is small here |
| shape-gated LayerNorm, BF16 diffusion core | 51.16 | 3.127 | 16.53 | 30.59 | 13.93 | 2.50 | promoted default |

This is the cleanest precision win so far: pairformer and confidence are flat,
while the diffusion transformer subrange falls from about `23.5-24.3s` to
`13.9-14.0s`.  The default policy now enables BF16 diffusion core on
BF16-capable CUDA devices; set `PROTENIX_BF16_DIFFUSION_CORE=0` to recover the
old conservative FP32 core.

Job `96301`
(`runs/atom_path_gate_n200_20260703_010037_51c2e8f`) then tested the atom
encoder/decoder boundary on the same mixed-sequence `N_sample=5`, `N_step=200`,
batch-size-16 workload:

| case | predict sec | generated samples/s | pairformer | diffusion | atom encoder | diffusion transformer | atom decoder | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current default after BF16 diffusion core | 51.23 | 3.123 | 16.59 | 30.65 | 7.50 | 13.88 | 7.33 | 2.47 | baseline repeat |
| BF16 atom attention only | 51.54 | 3.104 | 16.55 | 30.83 | 7.80 | 13.85 | 7.02 | 2.59 | rejected alone |
| BF16 atom attention + Triton local attention + fused bias | 44.36 | 3.607 | 16.12 | 23.72 | 2.94 | 14.08 | 2.13 | 2.45 | promoted |
| same plus local-attention gate fusion | 44.47 | 3.598 | 16.13 | 23.78 | 2.91 | 14.13 | 2.11 | 2.46 | rejected, no extra throughput |

The mechanism is specific: BF16 atom attention by itself is neutral because the
old local-attention/SDPA boundary still stores and reloads large FP32 tensors.
The Triton local-attention kernel consumes the profiled 32-query by 128-key atom
window directly, and the fused bias producer writes the layout that kernel
expects.  The atom encoder+decoder subtotal falls from `14.83s` to `5.06s`;
pairformer and confidence stay flat.  The current mixed-sequence `N_sample=5`
headline is therefore `212.51s -> 44.36s` predict and `0.753 -> 3.607`
generated samples/s, or `4.79x` over the old low-sample boundary.  The later
diffusion pair-bias cache gate below supersedes this headline with a `5.11x`
default by reducing the remaining diffusion-transformer subtotal.

Pairformer token-attention follow-up: job `96365`
(`runs/token_attention_gate_n200_20260703_015502_0dda88a`) tested an opt-in
reuse of the existing Triton `TriAttentionFunction` for dense pairformer token
attention.  The mechanism looked plausible in isolation: for one long padded
batch (`N=220`, `B=16`, `H=16`, `D=24`) pre-padded TriAttention was `6.4x`
faster than PyTorch math SDPA, and even with naive pad/copy overhead the
attention subrange improved `0.460 -> 0.403 ms`.  The shorter padded batch
(`N=124`) was neutral/slower, so the representative gate used a conservative
`N_token >= 160` guard.

The full mixed-sequence `N_sample=5`, `N_step=200`, batch-16 gate did not move
enough to promote.  Two alternating repeats gave default `44.63s` and `45.69s`
versus guarded TriAttention `44.81s` and `44.96s`; the averages are `45.16s`
and `44.89s`, only `0.6%` apart.  The candidate did reduce the long-batch
pairformer subtotal slightly, but pad/copy traffic plus ordinary diffusion and
atom-path variance erased the isolated win.  The opt-in code was removed rather
than kept as dormant complexity.  A future token-attention attempt should be a
true `D=24` dense-bias kernel that reads the existing layouts directly; reusing
the triangle-attention kernel through a padded dummy axis is not a meaningful
end-to-end optimization.

Diffusion-transformer compile follow-up: job `96407`
(`runs/diffusion_compile_probe_20260703_022616_47460d6`) screened
`torch.compile` on the exact flattened low-sample diffusion-token transformer
boundary.  The isolated steady-state result looked real: for the full 24-block
stack, `N=124` improved `24.29 -> 18.82 ms` (`1.29x`) and `N=220` improved
`48.35 -> 36.09 ms` (`1.34x`), with BF16-level output differences.  This was
the right next question because the post-atom profile shows thousands of
linears, SDPA launches, LayerNorms, sigmoid/gate kernels, residual adds, and
copy kernels rather than one bad attention implementation.

The actual model integration was rejected.  Commit `9679867` briefly added an
opt-in lazy compile hook, but the warmed representative gate job `96412`
(`runs/diffusion_compile_gate_n200_20260703_023303_9679867`) failed in the
compiled graph before timing: cuDNN frontend could not build a valid execution
plan for the SDPA call emitted by Inductor.  The diagnostic job was cancelled
after that failure point, and the model hook was removed.  Keep the isolated
probe as evidence that block-level fusion is attractive, but do not enable
`torch.compile` in the production path until the compiled SDPA backend is
controlled or replaced.

Narrow transition compile follow-up: job `96429`
(`runs/diffusion_transition_compile_probe_20260703_025244_ce590b5`) tested the
feed-forward transition boundary only, leaving SDPA eager:

```text
ConditionedTransitionBlock(attn_out, s) + attn_out
```

That avoids the invalid compiled-cuDNN-SDPA failure above.  The isolated H100
screen used the promoted/default BF16 path and the actual flattened
low-sample campaign shapes `[B * N_sample, N_token, C] = [80, N, 768]`:

| shape | eager boundary | compiled boundary | speedup | parity |
| --- | ---: | ---: | ---: | --- |
| `N_token=124`, mode `default` | 0.365 ms | 0.295 ms | 1.24x | max abs 0.03125 |
| `N_token=220`, mode `default` | 0.588 ms | 0.473 ms | 1.24x | max abs 0.03125 |
| `N_token=124`, mode `reduce-overhead` | 0.364 ms | 0.307 ms | 1.19x | max abs 0.03125 |
| `N_token=220`, mode `reduce-overhead` | 0.593 ms | 0.496 ms | 1.19x | max abs 0.015625 |

This is a real sub-block win, but the ceiling is modest because the transition
is only part of the 24-block diffusion transformer.  The follow-up model hook
was rejected by the representative gate, job `96434`
(`runs/transition_compile_gate_n200_20260703_030113_f6f982c`).  It deliberately
compiled only the feed-forward transition and left SDPA eager, but the full
mixed-sequence `N_sample=5`, `N_step=200`, batch-16 gate slowed down:

| case | samples/s | predict sec | diffusion transformer sec | decision |
| --- | ---: | ---: | ---: | --- |
| default | 3.590 | 44.570 | 14.266 | keep |
| compiled transition | 3.417 | 46.820 | 16.765 | reject |
| default repeat | 3.594 | 44.520 | 14.428 | keep |
| compiled transition repeat | 3.366 | 47.530 | 17.179 | reject |

The representative result moved the intended subtotal in the wrong direction,
so the model hook was removed.  Keep the isolated probe as evidence that the
transition boundary has launch/elementwise overhead, but do not use
`torch.compile` as the production mechanism for it.

Whole-transformer compile follow-up: the earlier 24-block isolated screen was
repeated with cuDNN SDPA disabled only inside the compiled diffusion-token
transformer, so Inductor could no longer choose the invalid cuDNN frontend
plan.  The isolated boundary still looked promising: job `96748`
(`runs/diffusion_compile_sdpa_backend_20260703_054221_6dabd86_retry`) measured
`1.29x` at `N_token=124` and `1.33x` at `N_token=220`, with BF16-level parity.

The representative gate did not promote it.  Job `96754`
(`runs/diffusion_compile_nocudnn_gate_n200_20260703_055004_f606ac0`) ran the
same 32 mixed 40-220-token records, `N_sample=5`, `N_step=200`, batch size 16,
confidence enabled:

| case | predict sec | generated samples/s | diffusion transformer sec | decision |
| --- | ---: | ---: | ---: | --- |
| default | 44.590 | 3.588 | 14.336 | keep |
| compiled transformer, cuDNN SDPA disabled | 44.920 | 3.562 | 14.439 | reject |
| default repeat | 43.870 | 3.647 | 13.993 | keep |
| compiled transformer, cuDNN SDPA disabled repeat | 43.950 | 3.641 | 14.014 | reject |

The compiled path is safe enough to run when cuDNN SDPA is disabled, but it
does not improve the actual throughput gate; even the intended transformer
subtotal was slightly slower.  The opt-in model hook was therefore removed.
Future work on this boundary should be explicit kernel/epilogue fusion, not a
generic `torch.compile` wrapper.

CUDA graph follow-up: job `96799`
(`runs/diffusion_transformer_cudagraph_20260703_063822_b9023d0`) tested a
narrower replay-only version of the same diffusion-token transformer boundary.
The goal was to separate real Python/launch overhead from the cost of copying
changing denoising-step inputs into graph-static buffers.  The probe captures
the 24-block transformer once, then measures both pure `graph.replay()` and
`copy_(a,s,z) + graph.replay()` against eager PyTorch in the same process.

| shape | eager | graph replay only | graph with changed-input copies | decision |
| --- | ---: | ---: | ---: | --- |
| `B=16`, `N_sample=5`, `N_token=124` | 33.45 ms | 31.21 ms (`1.072x`) | 31.28 ms (`1.070x`) | low-priority |
| `B=16`, `N_sample=5`, `N_token=220` | 73.73 ms | 71.85 ms (`1.026x`) | 72.01 ms (`1.024x`) | reject for now |

The graph path was exactly equal to eager for these deterministic inputs.  The
copies were not the limiter; the issue is ceiling.  The current representative
gate spends roughly `14s` of `44s` in the diffusion transformer, so even a
perfectly integrated transformer-only graph would be expected to save well
under 2% end to end on the long padded batches that dominate the campaign.  A
larger graph over conditioning, atom encode/decode, and the diffusion update
could be revisited, but the narrow transformer graph is not enough to justify
model-level graph machinery.

Diffusion pair-bias cache screen: commit `d43e0a0` adds
`scripts/perf/diffusion_pair_bias_cache_probe.py` for a more targeted
diffusion-transformer boundary.  The flattened low-sample path already keeps
the pair tensor `z` at record grain, but every diffusion transformer block still
projects `z` to `[B, heads, N, N]`, repeats it over flattened sample lanes, and
adds the token key mask on every denoising step.  `z` and the token mask are
step-invariant for the promoted unguided sampler, so this work could be cached
once per block before the `N_step=200` loop.

The screen compares current eager transformer execution with a manual block
loop that consumes precomputed final `[B * samples, heads, N, N]` biases.  It
also reports `cached_bias_rebuild_each_call` as a control: if rebuilding the
cache every call is flat but reusing it wins, the mechanism is specifically
cross-step reuse rather than a faster rewritten block.  The memory tradeoff is
explicit: at `B=16`, `N_sample=5`, `N=220`, storing 24 expanded BF16 biases is
roughly three GiB.

The H100 screen completed as canary job `97036`
(`runs/diffusion_pair_bias_cache_probe_canary_20260703_091034_dfc5ebd`) on the
two real sorted B16 bucket shapes:

| shape | eager transformer | cached reuse | rebuild cache each call | cache size | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `B=16`, `N_sample=5`, `N=124` | 26.28 ms | 20.89 ms (`1.26x`) | 23.31 ms (`1.13x`) | 0.90 GiB | advance to model gate |
| `B=16`, `N_sample=5`, `N=220` | 43.59 ms | 33.96 ms (`1.28x`) | 43.61 ms (`1.00x`) | 2.84 GiB | advance to model gate |

The `N=220` rebuild control is the important proof: rebuilding the cache every
call is flat, while reusing it wins.  The mechanism is not a better attention
block; it is deleting repeated, step-invariant projection/repeat/mask traffic
across the denoising loop.

The representative model gate then completed as job `97060`
(`runs/pair_bias_cache_gate_n200_20260703_091804_024d284`) with the same
32-record mixed 40-220-token campaign, `N_sample=5`, `N_step=200`,
`--batch_size 16`, confidence enabled, no MSA/template, and alternating
default/cache repeats:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | 45.13 | 3.545 | 15.77 | 24.70 | 14.75 | 2.55 | baseline |
| cached pair bias | 41.99 | 3.810 | 15.74 | 21.62 | 11.35 | 2.54 | promote |
| default repeat | 44.83 | 3.569 | 15.74 | 24.49 | 14.63 | 2.51 | baseline repeat |
| cached pair bias repeat | 41.23 | 3.881 | 15.71 | 20.91 | 11.18 | 2.53 | promote |

The paired result improves the intended diffusion-transformer subtotal by about
`1.30x` and the whole predict section by `1.08x` without moving pairformer or
confidence.  The current practical mixed-sequence `N_sample=5` headline is now
`212.51s -> 41.61s` predict on average and `0.753 -> 3.85` generated samples/s,
or `5.11x` over the old low-sample boundary.  The cache is now default-on for
multi-step batched diffusion and can be disabled with
`PROTENIX_CACHE_DIFFUSION_PAIR_BIAS=0`; `N_step=1` scout runs skip it because
there is no reuse window.

Post-cache launch profile: job `97110`
(`runs/post_cache_profile_n20_20260703_093234_3275f96`) captured a
representative `N_step=20`, `N_sample=5`, mixed-token, B16 trace after the pair
bias cache.  The model inference completed, then the job failed only in the
scoped aggregation command because `trace_multi_scope_aggregate.py` was invoked
without any `--scope` arguments.  The raw `1.8 GB` Chrome trace and global
aggregate were intact, so the scoped aggregate was rerun manually with exact
`user_annotation` ranges and written to
`profile_default_cache/profiles/rank0_scope_aggregate_user_exact.json`.

The batch timing now shows the intended bottleneck shift.  For the short
40-124-token bucket, pairformer was `7.93s`, diffusion `2.71s`, and confidence
`4.46s` inside a `25.66s` predict section.  For the long 136-220-token bucket,
pairformer was `11.43s`, diffusion `2.69s`, and confidence `6.41s` inside a
`29.08s` predict section.  The diffusion transformer subtotal is only
`0.73-0.82s` for `N_step=20`; the pairformer and confidence pairformer are now
the main remaining model costs.

Top global CUDA kernel classes in the post-cache trace were:

| kernel class | self CUDA ms | calls | interpretation |
| --- | ---: | ---: | --- |
| PyTorch/Triton BF16 LayerNorm kernels | 3517.5 | 13282 | normalization/layout-adjacent traffic dominates many pairformer paths |
| cuDNN full-token SDPA/FMHA | 1862.6 | 2400 | full attention remains material, but it is not one monolithic bad launch |
| BF16 direct-copy kernel | 1074.0 | 8078 | layout movement is still first-order |
| CUEQ fused gated dual GEMM | 1070.7 | 5760 | triangle-multiplication/transition fused MLP work is still material |
| BF16 elementwise multiply/add | 820.8 / 806.0 | 10964 / 12344 | residual/gate/activation traffic is launch-heavy |
| QKV-to-heads layout kernel | 665.6 | 2400 | attention producer layout remains visible |
| CUEQ/NVJET triangle GEMMs | 596.6, 578.5, 420.6 | 5760 / 4540 / 2960 | CUEQ is fast but still a large aggregate because it is called many times |

The corrected exact scope aggregate ranked pairformer subpaths by wall interval:

| scope | scope wall ms | calls | top kernel signal |
| --- | ---: | ---: | --- |
| triangle multiplication outgoing | 10892.6 | 1840 | cuDNN FMHA, LayerNorm, QKV layout, CUEQ fused GEMM/copy traffic |
| triangle attention starting-node | 6819.8 | 1840 | LayerNorm, elementwise multiply, cuDNN FMHA, NVJET GEMMs |
| token attention | 3868.2 | 1600 | cuDNN FMHA, LayerNorm, CUEQ fused GEMM, QKV layout |
| triangle multiplication incoming | 2969.4 | 1840 | LayerNorm, NVJET GEMMs, direct copies |
| triangle attention ending-node | 2530.2 | 1840 | LayerNorm, cuDNN FMHA, CUTLASS GEMMs, direct copies |
| atom encoder | 2358.0 | 40 | now smaller than the large pairformer scopes |
| diffusion token transformer | 1433.3 | 40 | no longer the leading target after pair-bias reuse |
| pair transition | 823.8 | 1840 | comparatively small after prior transition work |

Decision: keep the next experiments focused on pairformer kernels/layouts, but
do not promote a standalone replacement for pairformer token attention.  The
direct dense-bias D=24 token-attention screen tested the specific missing
boundary from the earlier failed padded TriAttention attempt: read the existing
`B x H x N x 24` q/k/v tensors plus non-contiguous dense pair bias directly,
without D=32 padding or copy overhead.  The isolated canary was encouraging:
the attention core was about `5.3-5.7x` faster and the full
`AttentionPairBias.attention` boundary was `1.6-2.3x` faster on the profiled
`N=124/220` buckets.

The representative model gate rejected the production hook.  Job `97256`
(`runs/d24_dense_attention_gate_n200_20260703_105346_e7ced70`) compared the
current promoted default with `PROTENIX_TRITON_D24_DENSE_ATTN=1` on the real
32-record mixed 40-220-token campaign, `N_sample=5`, `N_step=200`,
`--batch_size 16`, confidence enabled:

| case | predict sec | generated samples/s | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 41.78 | 3.830 | 16.18 | 20.95 | 2.54 | baseline |
| direct D24 attention | 41.96 | 3.813 | 15.96 | 21.34 | 2.57 | reject |
| default repeat | 40.97 | 3.905 | 16.03 | 20.39 | 2.46 | baseline repeat |
| direct D24 attention repeat | 41.87 | 3.821 | 15.93 | 21.37 | 2.49 | reject |

The mechanism moved the intended pairformer subtotal slightly
(`16.11s -> 15.95s` on average), but the whole predict section slowed
(`41.38s -> 41.92s`, `0.987x`).  The default-off runtime hook was removed
rather than carrying dead experimental code.  The probe remains as a useful
learning artifact: it shows why isolated kernel wins must be gated against the
actual campaign path before promotion.

Pre-cache promoted trace: job `96456`
(`runs/current_batched_profile_n20_20260703_031951_3d1adcd`) profiled the
actual `runner/batch_inference.py` campaign path after the atom-attention
promotion and transition-compile cleanup.  The workload was the same sorted
32-record mixed-token campaign, `N_sample=5`, `N_step=20`, `--batch_size 16`,
confidence enabled, no MSA/template, and the promoted default flags.  The
profile is for launch attribution only; representative promotion still uses
`N_step=200`.

Top global CUDA kernel classes in that trace were:

| kernel class | self CUDA ms | calls | avg us | interpretation |
| --- | ---: | ---: | ---: | --- |
| cuDNN full-token SDPA/FMHA | 1868.8 | 2400 | 778.7 | full attention remains the largest single vendor-kernel class |
| Triton LayerNorm fallback | 1680.4 | 11600 | 144.9 | promoted fallback is doing real work, not just overhead |
| CUEQ fused gated dual GEMM | 1045.3 | 5760 | 181.5 | pairformer fused MLP path remains material |
| BF16 direct-copy kernel | 1013.1 | 7268 | 139.4 | layout/dtype traffic is still a first-order cost |
| PyTorch FP32 LayerNorm | 892.7 | 9296 | 96.0 | conservative FP32 policy still leaves native LayerNorm launches |
| BF16 multiply | 820.1 | 10964 | 74.8 | generic elementwise launch floor |
| BF16 add | 805.6 | 12344 | 65.3 | generic elementwise launch floor |
| QKV-to-heads layout kernel | 666.0 | 2400 | 277.5 | attention-adjacent layout is material |
| CUEQ `layer_norm_transpose` | 581.7 | 5760 | 101.0 | CUEQ layout/normalization boundary remains material |
| BF16 SiLU | 449.5 | 3516 | 127.8 | transition/gating epilogues still launch-heavy |

The per-scope aggregates clarify where not to spend effort.  The batched token
diffusion transformer scope was `2.38s` in the N20 trace and was dominated by
direct-copy, SDPA, small/medium GEMMs, BF16 add/mul/sigmoid, and launch
overhead.  The atom encoder and decoder scopes were `2.45s` and `0.43s`; after
the promoted Triton local-attention path, the local-attention kernels themselves
were only tens of milliseconds.  The next atom win is therefore not another
local-attention kernel rewrite; it would need to remove many small launches or
copies around the atom blocks.

Roofline follow-up: Nsight Compute job `96769`
(`runs/ncu_roofline_next_bottleneck_20260703_060615_cecf4ae`) captured
filtered one-H100 launch ranges for the diffusion token transformer and one
CUEQ pairformer block.  It used NCU's `LaunchStats`, `Occupancy`,
`SpeedOfLight`, `SpeedOfLight_RooflineChart`, and `MemoryWorkloadAnalysis`
sections.  The absolute timings are replay-profiler timings, so use these rows
to classify kernels rather than as end-to-end latency.

| range/kernel | avg duration | memory pipe | DRAM | SM | achieved occupancy | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| diffusion `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | `250.9 us` | `70.1%` | `27.7%` | `54.2%` | `24.3%` | already a strong mixed compute/cache attention kernel; a naive attention replacement is unlikely to win |
| diffusion skinny NVJET GEMM `512x16...NNT` | `78.0 us` | `84.1%` | `84.1%` | `5.7%` | `13.9%` | HBM dominated skinny GEMM; reduce bytes or fuse the producer/epilogue, do not rewrite as a generic matmul |
| diffusion BF16 vectorized elementwise | `49.8 us` | `57.5%` | `56.9%` | `10.3%` | `68.2%` | memory-bound launch traffic; standalone Triton elementwise kernels already failed the full gate |
| pairformer CUEQ `layer_norm_transpose_forward_kernel` | `204.7 us` | `84.5%` | `59.9%` | `48.6%` | `48.4%` | CUEQ layout/normalization boundary is bandwidth limited and still attractive |
| pairformer CUEQ-adjacent NVJET GEMM `192x192...TNN` | `283.8 us` | `81.2%` | `81.2%` | `27.8%` | `14.8%` | triangle-multiplication wrapper work is HBM dominated |
| pairformer CUEQ-adjacent NVJET GEMM `128x256...TNN` | `139.2 us` | `80.4%` | `80.4%` | `18.9%` | `14.9%` | same: bytes and layout dominate more than tensor-core math |

The roofline result changes the kernel-design bar.  Broad `torch.compile`,
generic elementwise fusion, or a monolithic "mega-kernel" that reimplements
vendor attention/GEMM is not justified: the remaining attention mainloops are
not the obviously weak link.  The credible next experiments are narrower
producer/consumer boundaries that remove HBM trips while preserving cuDNN,
CUEQ, or cuBLAS-class math throughput.  One such screen is CUEQ 0.10's
`attention_pair_bias` primitive for diffusion token attention: it may fold
pair-bias projection, attention, gate, and output projection, but must be
screened because Protenix's diffusion head dimension is `48` and CUEQ documents
its fastest path for head dimensions that are multiples of `32`.

That CUEQ `attention_pair_bias` screen was rejected.  Probe commit `514ab88`
added `scripts/perf/diffusion_attention_pair_bias_cueq_probe.py`; follow-up
commits `a28fb39`, `da86115`, and `bfdfce7` made the screen robust enough to
isolate native CUEQ failures.  There are three useful lessons:

1. The production efficient-fusion convention matters.  `DiffusionModule`
   normalizes `z_pair` once before the 24-block transformer with a no-scale,
   no-bias LayerNorm, then each block's 1x1 `conv2d` applies only that block's
   learned per-head scale.  CUEQ's full `attention_pair_bias` primitive always
   performs LayerNorm inside the primitive, so replacing the full wrapper would
   move normalization back inside every block.
2. CUEQ 0.10's pair-bias Triton kernel cannot compile Protenix's no-offset
   LayerNorm call directly: weight is present but bias is `None`, and the
   kernel still reads the bias pointer.  Passing an explicit zero bias is
   mathematically equivalent and fixes that compilation issue in the probe.
3. After the zero-bias workaround, the CUEQ wrapper still fails its internal
   SDPA call on our H100/PyTorch 2.11/CUDA 13 stack with
   `cuDNN Frontend error: No valid execution plans built`.  Setting the probe's
   `--disable-cudnn-sdpa` flag did not help because the CUEQ wrapper opens its
   own `sdpa_kernel` context.  The failure appears at both actual Protenix
   head dimension `48` and a diagnostic head dimension `32`.

Representative isolated results:

| job/run | case | result | decision |
| --- | --- | --- | --- |
| `96780`, `runs/cueq_attention_pair_bias_isolated_20260703_062404_a28fb39` | forced PyTorch fallback CUEQ after q/k/v, `N=124` | `1.165 ms` after q/k/v versus baseline full `0.645 ms`; no native kernel | reject fallback |
| `96782`, `runs/cueq_attention_pair_bias_zerobias_20260703_062642_da86115` | zero-bias native CUEQ, `N=124/220`, head_dim `48` | pair-bias kernel compiles, then internal CUEQ SDPA fails with no cuDNN plan | reject wrapper |
| `96782` | zero-bias native CUEQ, diagnostic head_dim `32` | same internal SDPA no-plan failure | not a head_dim-48-only issue |
| `96787`, `runs/cueq_attention_pair_bias_no_cudnn_20260703_062842_bfdfce7` | zero-bias CUEQ with global cuDNN SDPA disabled | same CUEQ internal no-plan failure | no model integration |

Do not add a production CUEQ `attention_pair_bias` path unless the package
exposes a way to call only the fused pair-bias projection while using
Protenix's existing SDPA fallback policy, or unless the CUEQ wrapper itself
learns to handle cuDNN frontend plan failures.

A later scoped-pairformer trace, job `96710`
(`runs/profile_b16_mixed_n20_d9b8cab_20260703_051515`), added
`record_function` ranges around the pairformer block internals.  It confirmed
that CUEQ triangle multiplication is the right place to look inside the
pairformer: in the N20 profiling run, `triangle_mul_out` accounted for
`11.93s` and `triangle_mul_in` for `4.30s` of scoped time, with repeated
`cuequivariance::fused_gated_dual_gemm`, `cuequivariance::layer_norm_transpose`,
`aten::cat`, `aten::to`, and copy traffic around the fused CUEQ kernels.  That
suggested a narrow inference-only experiment: cache the packed/cast CUEQ
projection weights so each triangle-multiplication call no longer rebuilds the
same concatenated BF16 weight tensors.

The hotspot screen rejected that experiment before an end-to-end gate.  Job
`96737`
(`runs/cueq_weight_cache_hotspot_20260703_053331_fb33367`) alternated baseline
and cached-weight runs in one H100 allocation for the two mixed-batch shapes
that matter (`B=16`, `N_token=124` and `220`, BF16, CUEQ triangle attention and
triangle multiplication):

| shape | block ms baseline | block ms cached | triangle-mul ms baseline | triangle-mul ms cached | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `B=16, N=124` | 4.923 | 4.818 | 1.431 | 1.311 | reject: only `2.1%` block win |
| `B=16, N=220` | 14.896 | 14.830 | 3.521 | 3.456 | reject: only `0.45%` block win |

This is useful negative evidence.  The CUEQ kernels are fast enough that
removing the obvious weight-pack/cast staging has little total-block ceiling,
especially in the longer-token batch where triangle attention and pair
transition dominate.  The default-off cached-weight code path was removed
rather than carried as another unpromoted environment flag.  The remaining
pairformer opportunity is not Python-side weight packing; it is a larger
kernel-boundary change that removes work around CUEQ triangle attention or
triangle multiplication itself without adding new layout traffic.

The obvious transpose-removal variant was also screened and rejected.  Job
`96743` (`runs/end_triangle_transpose_probe_20260703_053931_1e4c462`) compared
the current ending-triangle-attention boundary,
`transpose(...).contiguous() -> starting-style CUEQ attention -> transpose back`,
against an algebraically equivalent `TriangleAttention(starting=False)` call on
the original pair tensor.  The hope was to avoid two full `[B, N, N, C]`
copies.  In practice, the strided orientation made the LayerNorm/QKV/CUEQ
producer boundary much worse:

| shape | explicit-transpose boundary | logical ending-node boundary | decision |
| --- | ---: | ---: | --- |
| `B=16, N=124` | 1.113 ms | 1.615 ms | reject: `45%` slower |
| `B=16, N=220` | 3.884 ms | 5.329 ms | reject: `37%` slower |

This explains why the current explicit contiguous transposes remain in the
inference block despite looking wasteful in isolation: they pay a predictable
HBM copy so the larger normalization, projection, and CUEQ kernels see the
layout they were optimized for.  A useful future change would have to fuse the
transpose into those producers or teach the kernels the alternate strides,
not merely pass a strided view through the existing path.

This profile motivated one narrow follow-up instead of re-enabling the rejected
broad fusion flag.  Commit `b5c8c3e` adds
`PROTENIX_TRITON_FUSED_ELEMENTWISE_MIN_ELEMENTS`, which keeps the default-off
generic Triton elementwise fusions away from tiny tensors.  The representative
gate was job `96600`
(`runs/elementwise_threshold_gate_n200_20260703_034213_b5c8c3e`), comparing the
current default with thresholds of `1,000,000` and `8,000,000` output elements
on the same `N_sample=5`, `N_step=200`, batch-16 campaign.

| case | predict sec | generated samples/s | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 44.020 | 3.635 | 15.747 | 23.774 | 2.461 | baseline |
| threshold 1M | 43.600 | 3.670 | 15.325 | 23.722 | 2.517 | reject: noise-level |
| threshold 8M | 44.030 | 3.634 | 15.482 | 23.955 | 2.527 | reject: flat |
| default repeat | 44.620 | 3.586 | 15.755 | 24.369 | 2.459 | baseline repeat |
| threshold 1M repeat | 44.090 | 3.629 | 15.380 | 24.063 | 2.577 | reject: noise-level |
| threshold 8M repeat | 44.220 | 3.618 | 15.412 | 24.133 | 2.603 | reject: flat |

The apparent best case is about `1.1%` faster by generated samples/sec, but the
movement is small compared with same-job variance and the intended subtotals
do not move decisively.  Keep the minimum-size guard only as a safety valve for
manual experiments with the already default-off global elementwise flag; do not
use it in the promoted recipe.

A larger attention-boundary follow-up tested the algebraic idea that won in
triangular CUEQ attention: when ordinary self-attention uses the same tensor
for q, k, and v, concatenate the three projection weights and run one wider
GEMM instead of three independent GEMMs.  The hotspot screen, job `96656`
(`runs/self_qkv_hotspot_20260703_035814_c67cde2`), showed the intended local
movement:

| case | block ms off | block ms on | qkv+sdpa+gate+out off | qkv+sdpa+gate+out on |
| --- | ---: | ---: | ---: | ---: |
| diffusion `N_token=124` | 0.910 | 0.881 | 0.336 | 0.294 |
| diffusion `N_token=220` | 1.429 | 1.427 | 0.654 | 0.595 |
| pair `N_token=124` | 1.004 | 0.965 | 0.367 | 0.321 |
| pair `N_token=220` | 1.184 | 1.182 | 0.454 | 0.441 |

Because the local subrange moved, it advanced to the full mixed-sequence
`N_sample=5`, `N_step=200`, batch-16 gate, job `96662`
(`runs/self_qkv_gate_n200_20260703_043101_22ec28b`).  The real gate rejected
promotion:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | 44.970 | 3.558 | 15.765 | 24.563 | 14.612 | 2.578 | baseline |
| fused self-QKV | 44.480 | 3.597 | 15.833 | 24.135 | 14.336 | 2.458 | repeat required |
| default repeat | 44.410 | 3.603 | 15.815 | 24.075 | 14.403 | 2.467 | baseline repeat |
| fused self-QKV repeat | 45.230 | 3.537 | 15.772 | 24.824 | 14.802 | 2.556 | reject: regressed |

The average is effectively flat/slower (`44.69s` default vs `44.86s`
candidate).  This is a useful negative result: reducing q/k/v launch count
does not help enough once the wider GEMM and downstream layout/cache effects
are embedded in the real diffusion/confidence/pairformer workload.  The
default-off code path was removed instead of leaving another unpromoted knob.

An adjacent producer-fusion screen tried concatenating the two
`AdaptiveLayerNorm` conditioning projections.  Hotspot job `96659`
(`runs/adaln_projection_hotspot_20260703_042917_22ec28b`) rejected it before a
full gate: the AdaLN subrange sometimes shrank by only `0.006-0.009 ms`, while
full diffusion blocks were flat or slower (`1.416 -> 1.494 ms` at
`N_token=220`).  This path was also removed.

Launch-level follow-up: job `96110`
(`runs/flat_bf16_batch_profile_20260702_221601_c8a532d`) proved the profiler
wrapper needed to catch `SystemExit` before exporting artifacts.  Corrected job
`96111`
(`runs/flat_bf16_batch_profile_fixed_20260702_222402_b4a1f78`) completed on
`gpu-canary`.  It profiled the robust flattened BF16 setting with `N_sample=5`,
`N_step=20`, `--batch_size 16`, confidence enabled, no MSA/template, and
fast-LayerNorm disabled to match the current CUDA 13/PyTorch 2.11 runtime.  The
job wrote a 2.0 GB Chrome trace, `rank0_batch_top_cuda.txt`, and
`trace_aggregate.json`.

Top raw CUDA kernel classes in that short trace:

| kernel class | self CUDA ms | calls | avg us | interpretation |
| --- | ---: | ---: | ---: | --- |
| PyTorch `vectorized_layer_norm_kernel` | 5692.5 | 21192 | 268.6 | largest raw kernel class after the C++ fast LayerNorm extension fails to build |
| cuDNN SDPA FMHA | 1861.3 | 2400 | 775.5 | vendor attention is material but already strong |
| BF16 copy/cast kernel | 1665.7 | 65366 | 25.5 | dtype/layout traffic remains significant |
| direct copy/cast kernel | 1099.9 | 35924 | 30.6 | more small traffic rather than one obvious math kernel |
| fused gated dual-GEMM epilogue kernel | 1042.3 | 5760 | 181.0 | existing CUEQ-adjacent fused path is still material |

Decision: do not replace CUEQ wholesale from this trace.  The first
bottleneck-backed candidate was a narrow inference-only Triton LayerNorm
fallback for the case where the shipped CUDA LayerNorm extension cannot compile.
Canary job `96143`
(`runs/layer_norm_hotspot_20260702_224156_3b3629e`) showed mixed isolated
microbenchmarks: BF16 large-row cases won up to `1.32x`, while some FP32 cases
regressed.  Short model gate `96151`
(`runs/layer_norm_model_gate_n20_20260702_225032_f95fc13`) then showed
`40.09s -> 26.23s` predict (`1.53x`) for the N20 version of the mixed campaign.
Representative N200 gate `96155` confirmed the end-to-end win above.  The
candidate is promoted as an automatic BF16/FP16 inference fallback, with
training, CPU, missing Triton, non-contiguous tensors, unsupported feature
dimensions, and default FP32 all falling back to PyTorch.

Post-promotion profile: job `96188`
(`runs/post_ln_default_profile_20260702_230930_88afbbe`) repeated the N20
mixed-campaign profiler at commit `88afbbe` with
`PROTENIX_TRITON_LAYER_NORM` unset.  It wrote a 1.8 GB trace and confirmed the
component bottleneck shifted: summed predict was `37.82s`, with pairformer
`19.51s` (52%), confidence `10.44s` (28%), and diffusion `5.70s` (15%).  The
largest raw CUDA classes were now:

| post-LN kernel class | self CUDA ms | calls | avg us | interpretation |
| --- | ---: | ---: | ---: | --- |
| GEMM via `aten::mm` | 2893 | 53862 | 53.7 | many small/medium matmuls are now the broad floor |
| CUDA command-buffer full | 2747 | 23853 | 115.2 | launch/command-buffer overhead remains material |
| CUEQ triangle attention | 1929 | 3200 | 778.6 | pairformer attention is a real remaining target |
| cuDNN SDPA FMHA | 1868 | 2400 | 778.4 | vendor token attention is also material |
| PyTorch `vectorized_layer_norm_kernel` | 1776 | 13832 | 128.4 | remaining native LayerNorm is mostly FP32/policy-disabled |
| Triton `_layer_norm_kernel` | 1605 | 7400 | 216.9 | promoted BF16/FP16 LayerNorm fallback now carries its share |
| CUEQ fused gated dual GEMM | 1051 | 5760 | 182.4 | CUEQ-adjacent fused MLP path remains nontrivial |

Mining the trace metadata showed the remaining `aten::native_layer_norm` calls
are not unsupported BF16 shapes; they are FP32 diffusion-token shapes such as
`[80, 124, 768]`, `[80, 124, 384]`, `[80, 220, 768]`, and `[80, 220, 384]`
with last-dimension-contiguous strides.  That explains why the conservative
default (`auto`) can be slower than the earlier `PROTENIX_TRITON_LAYER_NORM=1`
forced run: `auto` leaves FP32 on PyTorch.

An initial paired N200 gate, job `96214`
(`runs/layer_norm_force_fp32_gate_n200_20260702_235207_88afbbe`), compared
default `auto` against forced-all Triton LayerNorm in one job:

| case | predict sec | generated samples/s | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `auto` first, fresh Triton cache | 72.22 | 2.215 | 20.05 | 41.30 | 9.45 | not a fair baseline; likely paid first-use Triton compile/load overhead |
| `force_all` second | 58.53 | 2.734 | 16.10 | 37.62 | 2.56 | promising but order/cache-confounded |

This result is not enough to promote FP32 LayerNorm by default because the
faster case ran second after the low-precision Triton kernels had already been
compiled/loaded.  Warmed-cache gates `96218` on `gpu-canary` and `96219` on the
main `gpu` partition were submitted to prewarm both policies, then measure
`force_all` first and `auto` second.  Do not promote FP32 LayerNorm by default
unless one of those warmed gates beats the conservative default cleanly; the
isolated FP32 microbenchmarks were mixed.

The warmed-cache gate did show that FP32 LayerNorm still had a useful but
shape-dependent signal.  Job `96218`
(`runs/layer_norm_force_fp32_warm_gate_n200_20260703_000012_40e1c7e`) measured
forced-all Triton LayerNorm before auto on the same 32-record, 40-220-token,
`N_sample=5`, `N_step=200` campaign:

| case | predict sec | generated samples/s | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| forced-all Triton LayerNorm | 57.08 | 2.803 | 16.47 | 36.21 | 2.51 | useful signal |
| conservative auto | 61.67 | 2.594 | 16.52 | 41.15 | 2.52 | slower diffusion |

Commit `a271f5e` promoted the narrow version of that finding instead of the
force-all switch: in auto mode, Triton LayerNorm remains broad for BF16/FP16,
but FP32 uses Triton only for large flattened diffusion-token shapes with
feature widths `384` or `768` and at least `8192` rows.  This keeps the
diffusion win without changing every small FP32 normalization in the model.
The shape policy is covered by `tests.test_triton_layer_norm` and is already
included in all later current-stack gates.

The existing experimental Triton elementwise/residual/transition-input flags are
not a shortcut for this mixed-campaign workload.  Job `95635`
(`/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile/runs/fusion_flags_pair_b16_n200_20260702_170343`)
ran the same B16 gate with
`PROTENIX_TRITON_FUSED_ELEMENTWISE=1`,
`PROTENIX_TRITON_FUSED_ATTENTION_RESIDUAL=1`,
`PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL=1`, and
`PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION=1`.  On the same node/job,
`fusion_off` was already noisy (`41.64s` predict versus the promoted `33.93s`
on the earlier node), but the paired comparison is decisive:

| setting | predict sec sum | records/s by predict | pairformer | diffusion | diffusion transformer | confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fusion flags off | 41.64 | 0.768 | 18.00 | 19.41 | 10.48 | 1.82 |
| fusion flags on | 69.32 | 0.462 | 17.13 | 30.37 | 12.59 | 13.86 |

Decision: reject these flags for mixed-length B16 campaign inference.  They
slightly reduced pairformer time on this noisy node, but lost far more in
diffusion and confidence.  The README mixed-campaign command intentionally uses
the default-off settings.

A narrower follow-up branch
`codex/pair-transition-scoped-fusion` tried to apply only the pairformer
transition elementwise fusion, without enabling the global elementwise flag that
poisoned diffusion and confidence.  The isolated one-block screen looked
plausible: at the actual B20/N148 and B12-16/N220 campaign shapes, scoped
fusion reduced the `pair_transition` subrange by about 18-19% and the full
single-block time by about 3-4%.  The representative gate rejected the change:

| setting | predict sec sum | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| scoped pair-transition off | 41.74 | 15.26 | 17.51 | 7.79 | control |
| scoped elementwise only | 40.46 | 15.40 | 17.46 | 6.48 | reject: pairformer did not improve |
| scoped elementwise + input projection | 41.19 | 15.48 | 17.45 | 7.14 | reject: pairformer slower, more memory |

The apparent end-to-end movement in that noisy job came from unrelated
confidence variation, not from the intended pairformer mechanism.  Decision:
keep the branch as an audit trail, but do not merge it into `main`.  The lesson
is the same as above: a local subrange win is only a hypothesis until the
measured full-workload subtotal moves in the expected direction.

The current stack was retested after the later triangle-attention and BF16
diffusion promotions because the pair-transition boundary is still about one
fifth of a B16/N220 Pairformer block.  Job `96871`
(`runs/pair_transition_current_hotspot_20260703_073500_cec3101`) reproduced the
isolated transition win at the actual B16 padded buckets:

| shape | default transition | transition input fusion | local result |
| --- | ---: | ---: | --- |
| `B=16`, `N=124` | 1.053 ms | 0.859 ms | 1.23x transition |
| `B=16`, `N=220` | 3.218 ms | 2.617 ms | 1.23x transition |

The paired `PairformerBlock` screen, job `96884`
(`runs/pair_transition_block_flag_gate_20260703_074144_cec3101`), showed the
expected small block-level ceiling:

| shape | default block | transition flags block | default transition | flagged transition |
| --- | ---: | ---: | ---: | ---: |
| `B=16`, `N=124` | 4.91-4.93 ms | 4.71 ms | 0.996-0.997 ms | 0.806 ms |
| `B=16`, `N=220` | 14.91-14.95 ms | 14.27-14.36 ms | 3.062 ms | 2.464 ms |

That was still not enough to promote.  A representative full mixed-token gate,
job `96889`
(`runs/transition_flags_trunkexact_gate_n200_20260703_074435_cec3101`), used
the 32-record, 40-220-token, `N_sample=5`, `N_step=200`, `batch_size=16`
campaign with confidence enabled.  To avoid repeating the old broad-elementwise
failure, the candidate enabled transition input fusion but set
`PROTENIX_TRITON_FUSED_ELEMENTWISE_MIN_ELEMENTS=100000000`, so only the huge
pair-transition elementwise consumer should pass the size guard:

| setting | predict sec | generated samples/s | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default auto | 44.880 | 3.565 | 15.713 | 24.486 | 2.531 | control |
| transition flags | 44.710 | 3.579 | 15.338 | 24.727 | 2.545 | flat |
| default auto repeat | 44.190 | 3.621 | 15.685 | 23.914 | 2.504 | control |
| transition flags repeat | 44.770 | 3.574 | 15.362 | 24.797 | 2.528 | reject |

Decision: reject pair-transition input fusion as a default for the practical
mixed-token gate.  It consistently moves the intended pairformer subrange, but
the full run gives that back in diffusion and scheduling noise; mean predict
time was `44.54s` for default auto versus `44.74s` with transition flags.  A
future transition kernel would need to fuse the output/residual boundary or
otherwise reduce more than the input-projection epilogue to matter end to end.

Another narrow epilogue branch, `codex/transition-addmm-residual`, tried to
fold the pairformer transition residual add into the final transition GEMM with
`torch.addmm(beta=1)`.  This was the right kind of small boundary experiment:
it kept the library GEMM mainloop and asked the GEMM epilogue to consume the
caller residual, rather than replacing the matmul with a custom kernel.  The
actual-shape one-block gate rejected it:

| shape | baseline block ms | addmm-residual block ms | speedup | peak reserved | parity |
| --- | ---: | ---: | ---: | --- | --- |
| B20, N=148 | 8.056 | 8.077 | 0.997x | 1570 -> 1670 MiB | exact |
| B12, N=220 | 10.514 | 10.522 | 0.999x | 2412 -> 2412 MiB | exact |
| B16, N=220 | 13.825 | 13.858 | 0.998x | 3032 -> 3072 MiB | exact |

Job `95730`
(`runs/transition_addmm_residual_exact_20260702_181131`, commit `b22b74f`)
therefore rejects the branch.  The likely mechanism is that `torch.addmm` with
nonzero beta does not beat the existing `F.linear` plus simple in-place residual
add for these shapes, even though it is algebraically equivalent.

A follow-up branch, `codex/triangle-output-epilogue-hotspot`, tried the same
discipline on the triangle-attention output boundary.  CUEQ returns heads in a
heads-first layout, then the inference path applies `sigmoid(gate)`, flattens
heads, runs `linear_o`, and adds the block residual.  The prototype fused the
gate/layout/flatten/output-projection/residual boundary in one Triton matmul
kernel while leaving the CUEQ attention kernel itself unchanged.  The isolated
boundary looked real:

| shape | existing epilogue boundary | fused Triton boundary | isolated speedup | parity |
| --- | ---: | ---: | ---: | --- |
| B20, N=148 | 0.310 ms | 0.240 ms | 1.29x | max abs 0.03125 |
| B12, N=220 | 0.406 ms | 0.315 ms | 1.29x | max abs 0.03125 |
| B16, N=220 | 0.538 ms | 0.419 ms | 1.29x | max abs 0.03125 |

But the integrated `PairformerBlock` screen was only about `1.02x` at the
actual mixed-campaign shapes, and the representative full gate did not clear
the promotion bar.  Job `95758`
(`runs/tri_output_epilogue_full_b16_n200_20260702_183509`, commit `8d5eafb`)
ran the same 32-protein, 40-220-token, `N_sample=1`, `N_step=200` B16 workload
with the candidate off and on in one Slurm allocation:

| setting | predict sec sum | records/s by predict | pairformer | diffusion | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| output epilogue off | 37.20 | 0.860 | 15.59 | 18.00 | 2.48 | control |
| output epilogue on | 35.92 | 0.891 | 15.50 | 17.07 | 2.23 | reject |

The paired total moved by about `1.04x`, but that run was slower than the
promoted `33.93s` main gate and the movement was not cleanly concentrated in
the intended pairformer subtotal.  Decision: keep the branch as an audit trail
only.  Do not merge the default-off model path; fusing a true GEMM boundary is
not worth carrying unless the representative gate moves for the intended
reason.

A batch-size follow-up on commit `7de6bdf` screened fixed B16/B20/B24/B32 with
conditioning batching enabled (job `95646`,
`runs/batchsize_screen_cond_b16_32_20260702_171044`).  The node was slower than
the promoted conditioning gate, but the within-job shape tradeoff was clear:

| batch shape | predict sec sum | pairformer | diffusion | note |
| --- | ---: | ---: | ---: | --- |
| B16 + B16 | 39.81 | 17.75 | 18.52 | cold first measured case on this node |
| B20 + B12 | 34.95 | 16.73 | 16.39 | best fixed batch in this noisy screen |
| B24 + B8 | 36.33 | 17.11 | 17.40 | more first-batch padding; small second batch |
| B32 | 39.62 | 21.69 | 15.88 | diffusion improves, but pairformer padding dominates |

This motivated a guarded prototype that used `--batch_size 32` for diffusion
while chunking the pairformer trunk internally into B16 groups
(`PROTENIX_TOKEN_TRUNK_BATCH_SIZE=16`, commit `6ac0d70`).  The representative
gate (job `95654`, `runs/chunked_trunk_gate_20260702_172016`) produced
`33.67s` for B32-with-B16-trunk, bracketed by B20 controls at `35.00s` and
`33.81s`.  That is inside run noise and not a legitimate complexity trade, so
the prototype was reverted in commit `adf02e5`.

Actual-shape pairformer hotspot screens on current `main` (`11a7d9d`) confirm
that the remaining trunk work is a kernel problem, not another batching problem.
Job `95663`
(`runs/pairformer_block_actual_shapes_20260702_172840`) replayed one
`PairformerBlock` at the B20/B12 token buckets seen in the mixed campaign:

| shape | block ms | triangle attention | pair transition | triangle multiplication | token/single |
| --- | ---: | ---: | ---: | ---: | ---: |
| B20, N=148 | 8.16 | 40.4% | 19.3% | 29.3% | 7.0% |
| B12, N=220 | 10.50 | 43.7% | 19.7% | 26.5% | 6.1% |
| B16, N=220 | 13.71 | 44.3% | 20.1% | 25.8% | 5.8% |

Job `95666`
(`runs/triangle_attention_actual_shapes_20260702_173005`) then split the
triangle-attention module at those same shapes.  The promoted Triton q/k/v
layout producer and triangle-attention gate epilogue were active.  The main
CUEQ attention kernel is still the largest single launch family, but only about
half of the module time:

| shape/module | module forward ms | CUEQ attention | layernorm | qkv projection | gate/flatten/output family |
| --- | ---: | ---: | ---: | ---: | ---: |
| B20, N=148, starting | 1.53 | 1.14 ms | 0.15 ms | 0.20 ms | ~0.60 ms |
| B20, N=148, ending | 1.70 | 1.15 ms | 0.29 ms | 0.18 ms | ~0.61 ms |
| B12, N=220, starting | 2.15 | 1.64 ms | 0.18 ms | 0.23 ms | ~0.73 ms |
| B12, N=220, ending | 2.36 | 1.63 ms | 0.37 ms | 0.23 ms | ~0.74 ms |

Interpretation: the next legitimate pairformer kernel target has to either
improve the CUEQ attention schedule itself or fuse the surrounding layernorm,
projection, gate, flatten, and output-projection work while preserving the
current CUEQ attention throughput.  A slower "mega-kernel" attention mainloop
would lose, as the earlier CUTLASS transition-output experiment did.

Nsight Compute job `95692`
(`runs/ncu_triangle_actual_shapes_20260702_180029_ccfix`, commit `60d9c2d`)
captured one warmed `TriangleAttention.forward` launch range at the actual
mixed-campaign shapes.  The important result is that CUEQ's core is not one
opaque slow kernel: it delegates the attention mainloop to a strong cuDNN SDPA
launch, then pays several layout/wrapper/epilogue launches around it.

| shape/module | cuDNN SDPA | CUEQ/nvjet transforms | Triton q/k/v layout | Triton gate/flatten | LayerNorm | output GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B20, N=148, starting | 0.640 ms | 0.313 ms | 0.233 ms | 0.110 ms | 0.097 ms | 0.067 ms |
| B12, N=220, starting | 0.980 ms | 0.413 ms | 0.310 ms | 0.144 ms | 0.126 ms | 0.086 ms |

NCU's speed-of-light counters classify the attention mainloop as mostly
compute/L1 work (`~59%` SM throughput, only `~19-22%` DRAM throughput on these
captures), while q/k/v layout and gate/flatten are DRAM/L2-heavy (`~84-89%`
DRAM throughput).  That is a useful kernel-design constraint: replacing the
cuDNN/CUEQ attention mainloop is unlikely to win unless the replacement is
vendor-quality, but further wrapper or epilogue fusion can still remove real
HBM traffic if it does not slow the mainloop.

The existing non-CUEQ backends are not a shortcut.  Job `95693`
(`runs/triatt_backend_actual_shapes_20260702_180427`) compared one full
`PairformerBlock` with the same CUEQ triangle-multiplication path and only the
triangle-attention backend changed:

| shape | CUEQ block ms | TriAttention block ms | torch block ms | decision |
| --- | ---: | ---: | ---: | --- |
| B20, N=148 | 8.23 | 9.16 | 15.84 | keep CUEQ |
| B12, N=220 | 10.55 | 12.12 | 23.70 | keep CUEQ |
| B16, N=220 | 13.76 | 15.81 | 31.22 | keep CUEQ |

So the attractive boundary is not "swap CUEQ for another available attention
backend".  It is a narrower producer/consumer boundary around CUEQ: fewer q/k/v
layout passes, fewer gate/flatten/residual passes, or a native extension that
keeps cuDNN/CUEQ-class attention throughput while deleting wrapper traffic.

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

For the actual mixed-token 32-record `N_sample=5`, `N_step=200` campaign,
larger batches are worse even after the promoted batching and BF16 changes.
Job `96673`
(`runs/batch_size_16_32_gate_n200_20260703_044457_cd17527`) compared the
current stack at `--batch_size 16` and `--batch_size 32`:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B16 | 43.950 | 3.641 | 15.801 | 23.644 | 14.002 | 2.440 | keep |
| B32 | 54.500 | 2.936 | 22.426 | 27.565 | 18.319 | 2.443 | reject |
| B16 repeat | 43.980 | 3.638 | 15.734 | 23.706 | 14.056 | 2.469 | keep |
| B32 repeat | 54.590 | 2.931 | 22.541 | 27.480 | 18.191 | 2.494 | reject |

The mechanism is clear: B32 saves one Python/model call, but it also mixes the
entire 40-220-token range into one padded physical shape.  Pairformer grows
from about `15.8s` to `22.5s`, and the diffusion transformer grows from about
`14.0s` to `18.2s`, so per-GPU generated-sample throughput drops by about
`19%`.  For mixed-length low-sample campaigns, prefer `--batch_size 16` unless
a future bucketing rule can preserve tight token ranges at larger effective
batches.

The opposite direction was also screened in job `96680`
(`runs/batch_size_8_16_gate_n200_20260703_045758_6941bd0`).  B8 tightens the
token range inside each padded trunk batch and trims the pairformer subtotal a
little, but it doubles the number of diffusion/atom batches.  That extra
batch-boundary overhead and smaller GEMMs dominate:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | confidence | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B8 | 51.410 | 3.112 | 15.248 | 31.708 | 18.860 | 2.432 | reject |
| B16 | 44.980 | 3.557 | 15.893 | 24.404 | 14.381 | 2.578 | keep |
| B8 repeat | 53.750 | 2.977 | 15.669 | 33.457 | 19.760 | 2.554 | reject |
| B16 repeat | 44.830 | 3.569 | 15.805 | 24.529 | 14.621 | 2.454 | keep |

Together, the B8/B16/B32 gates show a local operational optimum rather than a
monotonic batch-size rule.  On this 32-record, 40-220-token, `N_sample=5`
campaign, B16 is the best fixed batch size because it keeps trunk padding
tolerable without fragmenting the diffusion and atom work into too many small
launch groups.  Better batching now means smarter token bucketing, not simply
raising or lowering `--batch_size`.

The older noisy B20 scout was rechecked after the current stack to make sure
the B16 recommendation was not just missing a better split point.  Job `96811`
(`runs/batch_size_16_20_gate_n200_20260703_064520_3e40e9f`) alternated
`--batch_size 16` and `20` on the same current branch and input:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B16 | 44.07 | 3.631 | 15.71 | 23.80 | 14.07 | keep |
| B20 | 44.77 | 3.574 | 16.02 | 24.21 | 14.47 | reject |
| B16 repeat | 43.98 | 3.638 | 15.74 | 23.68 | 14.03 | keep |
| B20 repeat | 44.64 | 3.584 | 16.18 | 23.96 | 14.29 | reject |

B20 is about `1.5%` slower on average.  Its first batch carries more padded
pairformer work, and the smaller long-token second batch does not recover
enough diffusion throughput to compensate.  Keep B16 as the documented fixed
batch-size default for this mixed-length N5 campaign.

The token-bucket scheduler was also gated after this fixed-batch sweep.  The
hypothesis was reasonable: keep `--batch_size 16` as the maximum launch group,
but split the length-sorted queue into approximate token-width buckets so the
pairformer sees less padded `O(B * N_max^2)` work.  Job `96904`
(`runs/token_bucket_gate_n200_20260703_080414_8cb1e6d`) rejected that idea for
the representative mixed-token campaign:

| case | queue buckets | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sorted default | 2 | 43.850 | 3.649 | 15.794 | 23.604 | 14.010 | keep |
| bucket96 | 3 | 48.930 | 3.270 | 16.283 | 28.164 | 15.733 | reject |
| bucket64 | 4 | 52.370 | 3.055 | 15.367 | 32.384 | 18.880 | reject |
| bucket48 | 5 | 62.670 | 2.553 | 19.151 | 39.679 | 22.925 | reject |
| bucket32 | 6 | 69.350 | 2.307 | 18.787 | 46.040 | 27.456 | reject |

Bucket64 is the clearest mechanism check: pairformer improves slightly, but
diffusion grows by `8.78s` because the token transformer, atom encoder, and atom
decoder run as four smaller groups instead of two fuller groups.  Narrower
buckets make this worse.  So the next batching win is not a queue heuristic; it
needs a ragged or segmented kernel/schedule that preserves large diffusion
launch groups while avoiding padded pairformer work.

Commit `2b8201b` tested that exact scheduler split: keep the outer sorted B16
diffusion batches intact, but internally sub-batch only the pairformer trunk by
token bucket.  The implementation added a temporary `--trunk_bucket_size`
knob, ran the trunk on smaller padded token groups, restored output order, and
then called the existing batched diffusion sampler on the original 16 records.
Remote unit tests passed (`23` tests, `2` skipped), but the representative full
gate rejected the idea, so the temporary production knob was removed rather
than left as default-off complexity.  Job `96951`
(`runs/trunk_bucket_gate_n200_20260703_083611_2b8201b`) produced:

| case | outer batches | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sorted default | 2 | 44.350 | 3.608 | 15.736 | 24.098 | 14.446 | keep |
| trunk96 | 2 | 45.250 | 3.536 | 16.430 | 24.256 | 14.351 | reject |
| trunk64 | 2 | 44.760 | 3.575 | 15.461 | 24.689 | 14.689 | reject |
| trunk48 | 2 | 49.440 | 3.236 | 20.619 | 24.176 | 14.230 | reject |
| trunk32 | 2 | 47.970 | 3.335 | 18.703 | 24.722 | 14.578 | reject |

The best mechanism check is `trunk64`: pairformer moved in the intended
direction by `0.28s`, but the extra trunk calls and changed memory/cache state
made diffusion `0.59s` slower, so total throughput fell.  More aggressive
trunk-only bucketing made the trunk itself worse.  This closes the cheap
scheduler option: avoiding padded pairformer work without losing throughput
needs a true segmented/ragged pairformer kernel or schedule, not Python-level
sub-batching.

Ending-node triangle attention looked like a more attractive layout boundary in
isolation.  Job `96838`
(`runs/triangle_attention_current_breakdown_20260703_070352_8c3409e`) first
updated `scripts/perf/triangle_attention_breakdown.py` so its manual path times
the same fused CUEQ epilogue as the production module.  At `B=16`, `N=220`, the
starting-node module took `3.12 ms`, while the ending-node module took
`5.01 ms`.  The split timer attributed most of the ending penalty to applying
LayerNorm after the logical pair-axis transpose: `layer_norm` was `1.79 ms` in
ending attention versus `0.52 ms` in starting attention.

Because channel-wise LayerNorm commutes with swapping the two residue axes, the
next screen tested `LN(x).transpose(...)` for ending-node attention instead of
`LN(x.transpose(...))`.  The diagnostic-only path in commit `fede984` was a
strong isolated win:

| shape | default ending manual ms | norm-first manual ms | module gate off | module gate on | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `B=16`, `N=220` | 5.46-5.48 | 4.39-4.41 | 4.99 ms | 3.94 ms | `~21%` module-local win |
| `B=16`, `N=148` | 2.50 | 2.02 | 2.23 ms | 1.75 ms | `~22%` module-local win |
| `B=16`, `N=96` | 2.18-2.19 | 2.04-2.08 | 1.97 ms | 1.76 ms | smaller but still positive |

The representative mixed-sequence gate rejected it.  Job `96851`
(`runs/ending_norm_e2e_gate_n200_20260703_071455_b8e9e12`) alternated the
production path off/on on the same 32-record, 40-220-token, `N_sample=5`,
`N_step=200`, batch-16 campaign:

| case | predict sec | generated samples/s | pairformer | diffusion | diffusion transformer | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| norm-first off | 44.49 | 3.596 | 15.78 | 24.13 | 14.44 | baseline |
| norm-first on | 44.23 | 3.617 | 15.75 | 23.86 | 14.08 | within noise |
| norm-first off repeat | 44.06 | 3.631 | 15.67 | 23.82 | 14.09 | baseline repeat |
| norm-first on repeat | 44.40 | 3.604 | 15.75 | 24.06 | 14.28 | repeat regressed |

Decision: reject the production path and remove it.  The local mechanism is
real, but in the full workload the pairformer subtotal did not move
consistently and total throughput averaged flat.  If revisiting ending-node
attention, do not stop at commuting LayerNorm; the likely larger boundary is a
contiguous ending-node producer that writes CUEQ's swapped q/k/v and bias
layouts directly while preserving the current CUEQ attention throughput.

A more ambitious CUEQ-adjacent producer fusion was then screened without
touching production.  Commit `7806ec0` added
`scripts/perf/triangle_attention_packed_projection_probe.py`, which replaces
the separate q/k/v, gate, and triangle-bias projections with one wider packed
projection, then unpacks q/k/v, gate, and FP32 CUEQ bias in one Triton layout
kernel.  This preserves the CUEQ attention mainloop, but tests whether the
three projection launches and repeated reads of the normalized pair tensor are
worth fusing.  Job `96936`
(`runs/triangle_packed_projection_probe_20260703_082814_7806ec0`) rejected it:

| shape | module forward ms | packed producer ms | speedup | parity |
| --- | ---: | ---: | ---: | --- |
| `B=16`, `N=124`, start | 0.8545 | 1.0807 | 0.7907x | exact |
| `B=16`, `N=124`, end | 1.4751 | 2.9951 | 0.4925x | max abs `0.00195` |
| `B=16`, `N=220`, start | 3.1363 | 3.7570 | 0.8348x | exact |
| `B=16`, `N=220`, end | 5.0170 | 9.7878 | 0.5126x | max abs `0.00195` |

Interpretation: CUEQ's surrounding projections are already better served as
separate cuBLAS calls plus narrow layout kernels than as one oversized producer
followed by a custom scatter.  For triangle attention, the remaining credible
kernel work must either fuse LayerNorm with the projection mainloop in a
vendor-quality schedule or attack ragged/padded work directly; simply widening
the projection boundary is the wrong shape.

A refreshed one-block profile on the current promoted path explains why that
one-line boundary was too small.  Job `96864`
(`runs/pairformer_block_current_b16_20260703_073024_a03319e`) timed a single
`PairformerBlock` at the two actual B16 padded buckets from the sorted mixed
campaign:

| shape | block ms | triangle attention | pair transition | triangle multiplication | token/single/transpose |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=16`, `N=124` | 4.91 | 37.6% | 20.3% | 28.9% | 13.2% |
| `B=16`, `N=220` | 14.89 | 44.9% | 20.6% | 23.6% | 10.9% |

For `N=220`, starting and ending triangle attention are now essentially equal
inside the full block (`3.34 ms` each), while pair transition is another
`3.06 ms`.  The next credible trunk win therefore needs to cover a broader
boundary than one ending-node LayerNorm layout: either a CUEQ-quality triangle
attention producer/consumer rewrite that moves both start and end, or a
vendor-quality transition epilogue/mainloop that preserves cuBLAS throughput.

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
singleton/parity debugging.  The intermediate `--batch_mode trunk_exact
--batch_size 16` mode was measured in job `96889` on the same 32-record,
40-220-token, `N_sample=5`, `N_step=200` gate: it preserves the pairformer
trunk's real physical token length for each record and still batches the
diffusion tail, but predict time was `100.77s` (`1.588` generated samples/s)
versus `44.19-44.88s` (`3.57-3.62` generated samples/s) for default padded
`auto`.  That makes `trunk_exact` a correctness/audit escape hatch, not the
throughput default.  A true exact-and-fast mixed-token solution needs a ragged
pairformer schedule or custom kernels that reduce over each sequence's real
length while sharing launches across records.

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

Decision: for the same-output high-`N_sample` path this gate-fusion boundary is
a useful kernel-learning result and memory reduction, but not a promoted
throughput win until repeated full gates clear the noise floor.  For the
many-independent-sequence `N_sample=5` path, the underlying Triton local
attention plus fused bias producer is promoted by job `96301` above; only the
extra gate-fusion flag remains opt-in.

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

The diffusion core now defaults to BF16 on BF16-capable CUDA devices after the
mixed-sequence gate showed a clear end-to-end win concentrated in the diffusion
transformer.  Set `PROTENIX_BF16_DIFFUSION_CORE=0` for conservative numerical
audits.  BF16 atom attention is also default-on for BF16-capable CUDA, but only
became a promoted win once the atom local-attention consumer was moved to the
guarded Triton path with fused bias production.  Set
`PROTENIX_BF16_ATOM_ATTENTION=0` and `PROTENIX_TRITON_LOCAL_ATTN=0` to recover
the old atom path.  Full token attention is allowed to stay BF16 with
`PROTENIX_ATTENTION_FORCE_FP32=0`.

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

Pairformer transition ragged-layout screen: commit `6dee304` adds
`scripts/perf/pair_transition_ragged_hotspot.py` to test the first safe
segmented-pairformer boundary.  Pair transition is row-wise over `(i, j)` pair
features, so a compact valid-pair layout is mathematically straightforward:
run the transition MLP on only `sum(length_i^2)` real pairs instead of the
padded `batch * max_length^2` pairs.  The screen deliberately reports three
variants:

- ideal compact compute, assuming later pairformer kernels can consume compact
  rows directly;
- compact compute plus gathering valid rows from today's padded tensor;
- full gather/compute/scatter, which is the cost of inserting only this one
  compact island into the current padded pairformer.

This distinction matters for kernel design.  A win only in the first row says
"rewrite a larger ragged pairformer boundary"; a win after scatter says "this
one local MLP is worth integrating now."  The completed H100 canary was job
`97035`
(`runs/pair_transition_ragged_hotspot_canary_20260703_091034_dfc5ebd`) for the
two real sorted B16 token groups (`40-124` and `136-220`):

| token group | valid-pair efficiency | padded transition | compact compute only | gather + compute | gather + compute + scatter | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `40-124` | 48.6% | 0.954 ms | 0.482 ms (`1.98x`) | 0.559 ms (`1.71x`) | 0.634 ms (`1.50x`) | useful isolated signal |
| `136-220` | 67.0% | 2.905 ms | 1.944 ms (`1.49x`) | 2.270 ms (`1.28x`) | 2.548 ms (`1.14x`) | too small alone |

All variants matched the padded result on valid pairs.  The result is positive,
but not a direct 10x lever: this transition MLP is only a fraction of each
pairformer block, and inserting one compact island would add gather/scatter
traffic around kernels that still consume padded pair tensors.  Keep this as
evidence for a future larger ragged pairformer boundary, not as a promoted
single-module optimization.

The next screen widened the question from one row-wise transition MLP to a
whole `PairformerBlock`.  Commit `134cbaa` added
`scripts/perf/pairformer_ragged_islands_hotspot.py`, which compares today's
single padded B16 block with lower-bound "islands": two smaller sorted buckets,
exact-length groups, and singletons.  This is deliberately benchmark-only; it
does not preserve the cross-block pair layout and it pays extra launches.  A
large win here would justify a real single-kernel segmented/CuTe design.  A
flat or negative result says the existing CUEQ/CUDA kernels are already too
good at the padded shapes for Python-level islanding to matter.

The one-H100 canary was job `97271`
(`runs/ragged_pairformer_islands_20260703_110850_134cbaa`).  The job failed
only after both JSON results were written, because the ad hoc Slurm summary
used an unquoted heredoc around Python f-strings.  A salvaged summary was
written to `summary_salvaged.md`:

| token group | variant | valid-pair efficiency | groups | block ms | speedup vs padded |
| --- | --- | ---: | ---: | ---: | ---: |
| `40-124` | padded all | 0.486 | 1 | 4.777 | 1.000 |
| `40-124` | two split buckets | 0.707 | 2 | 7.256 | 0.658 |
| `40-124` | exact-length groups | 1.000 | 8 | 25.792 | 0.185 |
| `40-124` | singletons | 1.000 | 16 | 50.353 | 0.095 |
| `136-220` | padded all | 0.670 | 1 | 15.227 | 1.000 |
| `136-220` | two split buckets | 0.832 | 2 | 13.198 | 1.154 |
| `136-220` | exact-length groups | 1.000 | 8 | 30.548 | 0.498 |
| `136-220` | singletons | 1.000 | 16 | 59.933 | 0.254 |

Decision: reject Python-level ragged islands and do not expect small
pairformer sub-bucketing to close the gap.  The short bucket is already so fast
that extra launches dominate even when valid-pair efficiency improves from
`0.486` to `0.707`; the long bucket has a modest `1.15x` block-level island
signal, but the combined short+long block total is essentially flat/slower.
A true ragged pairformer kernel would need to keep the one-launch/vendor-kernel
quality of CUEQ while avoiding padded work.  That remains a large kernel design
project, not a cleanup or batching knob.

A small mask-format hypothesis was also screened before changing model code.
Triangle attention already feeds CUEQ a boolean mask, but triangle
multiplication still receives the normal BF16 pair mask.  Commit `53bf637`
added a benchmark-only `--pair-mask-dtype` knob to
`scripts/perf/pairformer_block_breakdown.py`; job `97278`
(`runs/pairformer_bool_mask_screen_20260703_111545_53bf637`) compared BF16 and
bool masks on the two B16 shapes:

| shape | BF16 mask block ms | bool mask block ms | decision |
| --- | ---: | ---: | --- |
| `B=16, N=124` | 4.904 | 4.910 | flat |
| `B=16, N=220` | 14.909 | 14.840 | +0.5%, below threshold |

Decision: do not plumb bool masks into triangle multiplication.  The result is
within hotspot noise and far too small to justify changing pair-mask dtype
contracts across the model.

### Same-GPU concurrency / MPS gate

Status: promote as an **operational campaign mode**, not as a model-kernel
change.

Hypothesis: the optimized mixed-sequence path still launches many medium and
small kernels.  A single `--batch_size 16` campaign does not fully occupy an
H100, so several independent warm Protenix workers on the same GPU may improve
per-GPU throughput.  CUDA MPS should help the separate CUDA contexts overlap
more effectively than ordinary process concurrency.

Benchmark harness: commit `184a494` adds
`scripts/perf/concurrent_batch_inference_gate.py`.  The harness loads persistent
workers, runs a short warmup, writes warmup and every timed repeat to distinct
dump directories, synchronizes workers at the timed boundary, and fails the
summary if parsed model timing logs do not cover the expected records.  This
matters: the first draft reused warmup output directories and could report
bogus few-second runs; the hardened gate caught that failure mode.

Representative workload:

- one H100 80 GB on Sandpit Tokyo;
- 32 mixed 40-220-token records per worker;
- `N_sample=5`, `N_step=200`, `--batch_size 16`, BF16, CUEQ triangle kernels;
- confidence head and normal CIF/JSON dumping enabled;
- branch `codex/ragged-multisample-diffusion`, commit `184a494`.

Non-MPS two-worker gate:

- job `97296`;
- run dir
  `runs/concurrency_gate_20260703_113119_184a494`;
- sequential two-campaign wall baseline: `320` generated samples in `86.08s`,
  or `3.717` generated samples/s;
- two ordinary processes on one GPU: `320` generated samples in `79.38s`, or
  `4.031` generated samples/s.

Width and MPS gate:

- job `97306`;
- run dir
  `runs/concurrency_width_gate_20260703_114157_184a494`;
- CUDA MPS pipe/log directories must live on a node-local filesystem such as
  `/tmp`.  Putting `CUDA_MPS_PIPE_DIRECTORY` on Lustre created control logs but
  no live socket and should not be used.

| mode | workers | generated samples | wall s | wall samples/s | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ordinary processes | 2 | 320 | 79.38 | 4.031 | small win |
| ordinary processes | 3 | 480 | 115.86 | 4.143 | flattening |
| ordinary processes | 4 | 640 | 152.56 | 4.195 | flattening |
| CUDA MPS, `/tmp` socket | 2 | 320 | 66.36 | 4.822 | useful |
| CUDA MPS, `/tmp` socket | 3 | 480 | 94.32 | 5.089 | near knee |
| CUDA MPS, `/tmp` socket | 4 | 640 | 124.58 | 5.137 | best measured |

Interpretation:

- Ordinary process concurrency proves there is some underfill, but context
  isolation and contention cap the win quickly.
- CUDA MPS materially improves overlap across independent workers.  The best
  measured point, four workers, is only slightly above three workers, so the
  practical recommendation is **three to four MPS workers per H100** for large
  campaign shards.
- Compared with the prior promoted single-process low-sample rate (`3.85`
  generated samples/s), `5.137` generated samples/s is a `1.33x` operational
  gain.  Compared with the original low-sample branch boundary (`0.753`
  generated samples/s), it is about `6.8x` per-GPU throughput.
- This is not a substitute for kernel work: each worker individually slows down
  as width increases, and the aggregate curve is already flattening.  The
  remaining route to the requested `10x` target still needs a larger true
  kernel/layout win, not just more same-GPU processes.

Decision: document and recommend CUDA MPS concurrency for throughput-oriented
design campaigns with enough independent shards.  Keep the production model
code unchanged; the benchmark harness remains the reproducibility tool.

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
| Raising mixed-token `N_sample=5` batch size from 16 to 32 | current full gate slowed from `3.64` to `2.93` generated samples/s because pairformer and diffusion transformer ran on a much larger padded physical shape | bigger batches only help when bucketing keeps token ranges tight; memory headroom alone is not a throughput argument |
| Lowering mixed-token `N_sample=5` batch size from 16 to 8 | current full gate slowed from `3.56` to `3.0-3.1` generated samples/s because diffusion/atom work split into four smaller launch groups | tighter token ranges are useful only if the saved padding beats the lost batching efficiency |
| Explicit mixed-token bucket sizes `96`, `64`, `48`, and `32` at B16 | all slowed the representative N200 gate; best explicit bucket fell from `3.649` to `3.270` generated samples/s and narrower buckets reached `2.307` generated samples/s | queue bucketing saves some padded pairformer work only by fragmenting diffusion/atom batches; the real solution needs ragged kernels or segmented schedules, not more launch groups |
| Internal pairformer-only trunk buckets while keeping diffusion B16 | best case `trunk64` improved pairformer `15.736 -> 15.461s`, but diffusion worsened `24.098 -> 24.689s` and throughput fell `3.608 -> 3.575` generated samples/s; narrower trunk buckets were much worse | Python-level sub-batching changes launch/cache balance and is not the segmented-pairformer solution; remove the knob and target true ragged kernels/schedules |
| Full-block ragged pairformer islands | two split buckets improved the long `136-220` block `1.15x`, but slowed the short `40-124` block to `0.66x`; exact-length groups and singletons were much slower | launch count and lost batching eat the padding savings unless the ragged schedule is a real one-launch/vendor-quality kernel |
| Bool masks for CUEQ triangle multiplication | block screen was flat at `N=124` and only +0.5% at `N=220` | mask dtype plumbing is not a meaningful bottleneck after triangle-attention bool-mask cleanup |
| Materializing pairformer token-attention bias as contiguous `[B, H, Q, K]` | traced pairformer SDPA shape `[16, 16, 124, 24]` moved from math-dispatch attempt to a cuDNN frontend "no valid execution plans" error; fallback timing then segfaulted in the isolated screen | the non-contiguous bias layout is not the only blocker; head_dim 24 plus dense additive bias lacks a safe cuDNN plan in this runtime, so do not pay a bias copy hoping for a backend flip |
| CUEQ `attention_pair_bias` for diffusion token attention | native pair-bias projection needed an explicit zero LayerNorm bias, then CUEQ's internal SDPA failed with `No valid execution plans built`; forced PyTorch fallback was slower than the current full conv2d+SDPA baseline | CUEQ's wrapper is not a drop-in win here; use it only if we can call a narrower fused pair-bias producer while keeping Protenix's SDPA fallback policy |
| Reusing Triton `TriAttentionFunction` for pairformer token attention | isolated `N=220` hotspot improved, but the full mixed-sequence gate averaged only `45.16s -> 44.89s` predict (`+0.6%`) and one repeat regressed | padding `D=24` to `D=32` and copying layouts is too much integration overhead; if revisited, write a direct dense-bias `D=24` kernel instead |
| Direct dense-bias D24 Triton token attention | isolated core improved about `5.3-5.7x` and the full attention boundary `1.6-2.3x`, but the representative mixed `N_sample=5`, `N_step=200` gate slowed from `41.38s` to `41.92s` on average | even a good narrow attention kernel is too small after pair-bias caching; remove rejected default-off runtime hooks and target larger pairformer/triangle boundaries |
| `torch.compile` around the diffusion-token transformer | isolated 24-block screens improved `1.29-1.34x`; the first model gate failed in a cuDNN SDPA plan, and the safer no-cuDNN-SDPA gate was flat/slower (`44.59 -> 44.92s`, repeat `43.87 -> 43.95s`) | block-level fusion remains attractive, but a generic Inductor wrapper does not move the real campaign path |
| Fused self-attention `q/k/v/g` projection | exact parity but trunk block slowed from about 14.1 ms to 14.9 ms | fewer launches are not enough if the wider GEMM/cache/layout is worse |
| Fused ordinary self-attention q/k/v projection | hotspot moved the intended `qkv+sdpa+gate+out` subrange, but the full mixed `N_sample=5`, `N_step=200` gate averaged flat/slower (`44.69s -> 44.86s`) and one repeat regressed | producer fusion must still move the full workload; local launch savings can disappear in wider GEMM/layout/cache effects |
| Ending-node triangle attention `LN(x).transpose(...)` | ending-attention module screens improved about `21-22%`, but the full mixed `N_sample=5`, `N_step=200` gate was flat/noisy (`44.49 -> 44.23s`, repeat `44.06 -> 44.40s`) and pairformer did not consistently move | channel-wise LayerNorm commutes with the pair-axis transpose, but moving only this boundary is not enough; revisit only as part of a larger contiguous ending-node q/k/v+bias producer |
| Packed CUEQ triangle-attention q/k/v+gate+bias producer | exact or near-exact isolated screen, but slower on every actual shape: start attention `0.79-0.83x`, ending attention `0.49-0.51x` | one oversized projection plus a custom unpack/scatter loses to separate cuBLAS projections plus narrow Triton layout kernels; do not widen this producer boundary without a vendor-quality fused LayerNorm+GEMM schedule |
| Fused `AdaptiveLayerNorm` conditioning projections | AdaLN subrange shrank slightly in some hotspot shapes, but full diffusion blocks were flat/slower, e.g. `1.416 -> 1.494 ms` at `N_token=220` | do not carry default-off concatenated-GEMM knobs unless the enclosing block moves |
| Fused transition `a1/a2` projection without split-aware consumer | transition slowed from about 5.3 ms to 7.0 ms | fusing the producer can break the consumer by creating non-contiguous halves |
| Custom Triton transition input GEMM+SiLU | synthetic +12%, but real transition input path slowed from 3.09 ms to 3.54 ms | custom GEMM screens must use real module weights/autocast; cuBLAS efficiency is the bottleneck to match |
| Unguarded pairformer transition input fusion at B96 | failed before producing model output | the concatenated pair-transition activation is too large; keep the shape guard |
| Scoped pairformer transition elementwise fusion | isolated pair-transition subrange improved 18-19%, but the full representative gate showed pairformer flat/slower | subrange wins must be rejected when the measured full-workload subtotal does not move |
| Pairformer transition residual via `torch.addmm(beta=1)` | exact parity, but actual-shape blocks were neutral/slower (`0.997-0.999x`) and sometimes used more memory | preserving the library GEMM is necessary but not sufficient; the epilogue path still has to beat `F.linear` plus a simple add |
| Triangle-attention output projection epilogue | isolated boundary improved about `1.29x` and block screens about `1.02x`, but the full B16 mixed-token gate was noisy and did not cleanly move pairformer or beat the promoted main gate | do not carry default-off fused GEMM paths unless the representative subtotal moves for the intended mechanism |
| Custom Triton transition output GEMM+gate/residual | best tile 2.00 ms versus 1.51 ms for cuBLAS plus fused gate/residual | the output epilogue is still attractive, but only if implemented as a vendor-quality CuTe/CUTLASS epilogue |
| Naive fused Triton triangle attention | correct at N64, but B16/N245 slowed from 4.24-4.26 ms to 6.22-7.78 ms | the right boundary still needs a vendor-quality CUEQ/CuTe schedule |
| Replacing CUEQ triangle attention with existing `triattention` or torch backends | actual-shape one-block screens were slower: TriAttention `1.11-1.15x` slower, torch about `1.9-2.3x` slower | keep CUEQ; target its layout/wrapper/epilogue boundaries instead |
| CUEQ triangle-multiplication ONDEMAND tuning | total B64 block +0.34%, below noise | the shipped/default schedule is close enough for this shape |
| CUEQ q/k/v layout block-size retuning | block `256` beat the current `1024` by 3.7-4.2% in isolation, but only saves `0.04-0.12` ms per layout launch | do not promote tiny tile retunes without a full-gate signal |
| Logical ending-node triangle attention without explicit pair transposes | bitwise exact one-block screen but only `1.0076-1.0089x` at B32/B64/B96 | explicit pair-tensor orientation copies are measurable but not the dominant pairformer boundary |
| Default BF16 full-token attention after the promoted stack | isolated trunk-attention hotspot improved, but full exact-shape gates moved only +0.16% at B64 and +0.27% at B96 | do not promote dtype-policy changes unless the representative throughput gate moves, not just the local kernel |
| Inference DataLoader workers for exact-shape batches | B8 had a small screen win, but B32 hit tensor-sharing failures unless switched to slower file-system sharing; clean B32 no-worker batching was better | do not add host-pipeline complexity unless it survives the actual many-input gate |
| CUDA graph replay of the diffusion-token transformer | exact and mildly faster in isolation (`1.07x` at `N=124`, `1.02x` at `N=220`), but the long-shape ceiling is under 2% e2e | launch overhead is not the meat of the remaining transformer bottleneck; revisit only as part of a larger full-denoising graph |
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
- `protenix/model/modules/transformer.py`: token/atom key-mask plumbing for
  padded mixed-length batching, masked atom-to-token aggregation, local-attention
  bias fusion, cached step-invariant diffusion pair-bias projection, and
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
- `protenix/model/layer_norm/layer_norm.py`: dispatches to the Triton/PyTorch
  LayerNorm fallback when the C++ fast LayerNorm extension is unavailable.
- `protenix/model/layer_norm/layer_norm_triton.py`: low-precision
  inference-only Triton LayerNorm fallback with conservative guards.  It is
  automatic for FP16/BF16 no-grad CUDA tensors, opt-in for FP32 with
  `PROTENIX_TRITON_LAYER_NORM=1`, and disabled with
  `PROTENIX_TRITON_LAYER_NORM=0`.
- `protenix/model/triangular/qkv_layout_triton.py`: inference-only CUEQ q/k/v
  layout producer for triangle attention; keeps the fused projection GEMM and
  CUEQ attention kernel unchanged while removing hidden contiguous-copy traffic.
- `protenix/model/modules/confidence.py` and `protenix/model/protenix.py`:
  confidence-logit chunk iteration, streaming summary consumption, and the
  mixed-token diffusion sampler that pads/masks conditioning, token-transformer,
  and atom-transformer activations across records while reusing diffusion
  transformer pair biases across denoising steps.
- `runner/campaign_inputs.py`, `runner/inference.py`,
  `runner/batch_inference.py`, and `configs/configs_inference.py`: public
  campaign inference batching.  The CLI length-sorts raw campaign records before
  featurization, the runner uses full-model batches for identical feature tensor
  trees, and the trunk path supports both same-token variable-atom records and
  masked padded-token/padded-atom records before writing one output per input.

Profiling and reproducibility helpers:

- `scripts/perf/protenix_inference_benchmark.py`: full-model throughput gate.
- `scripts/perf/concurrent_batch_inference_gate.py`: same-GPU multi-process
  campaign gate; measures ordinary process concurrency and CUDA MPS throughput
  with persistent warm workers and isolated output directories per timed repeat.
- `scripts/perf/full_inference_precision_audit.py`: full-run matmul precision
  audit; records forced-FP32 modules and non-module einsums/matmuls.
- `scripts/perf/atom_transformer_hotspot.py`: atom block hotspot screen.
- `scripts/perf/local_attention_hotspot.py`: local-attention kernel screen.
- `scripts/perf/local_attention_bias_hotspot.py`: bias-projection screen.
- `scripts/perf/trunk_attention_hotspot.py`: trunk attention block screen.
- `scripts/perf/dense_bias_attention_d24_probe.py`: direct Triton
  dense-pair-bias token-attention screen for the pairformer `D=24` head shape.
- `scripts/perf/triangle_attention_breakdown.py`: per-launch/subrange screen
  for CUEQ triangle attention and its layout/epilogue boundaries.
- `scripts/perf/diffusion_pair_bias_cache_probe.py`: tests whether caching the
  step-invariant projected diffusion pair bias for each transformer block can
  remove repeated work across the denoising loop.
- `scripts/perf/transition_output_epilogue_hotspot.py`: transition output-GEMM
  epilogue screen and candidate-injection harness for native CuTe/CUTLASS work.
- `scripts/perf/transition_output_epilogue_reference_candidate.py`: baseline
  candidate ABI smoke for the transition epilogue harness.
- `scripts/perf/pair_transition_ragged_hotspot.py`: compact valid-pair
  pairformer-transition screen; separates ideal compact compute from
  gather/scatter costs before attempting a larger ragged pairformer rewrite.
- `scripts/perf/pairformer_ragged_islands_hotspot.py`: lower-bound whole-block
  screen for ragged pairformer islands; compares one padded B16 block with
  smaller split buckets, exact-length groups, and singletons.
- `scripts/perf/tokyo_transition_epilogue_hotspot.sbatch`: Tokyo one-H100
  wrapper for the transition epilogue screen, including CUDA/CUTLASS env setup.
- `scripts/perf/confidence_head_hotspot.py`: confidence pairformer screen.
- `scripts/perf/confidence_precision_screen.py`: confidence precision/parity
  screen with real checkpoint weights and matmul dtype tracing.
- `scripts/perf/layer_norm_hotspot.py`: isolated H100 screen for the Triton
  LayerNorm fallback versus the PyTorch fallback.
- `scripts/perf/trace_aggregate.py`: compact profiler trace summarizer.
- `scripts/perf/trace_multi_scope_aggregate.py`: multi-scope Chrome-trace
  summarizer used for pairformer range attribution.

## If continuing from here

Do not expect another large gain from config changes.  The next proper
same-output campaign should be one of:

1. a deliberate atom/trunk kernel-layout rewrite, especially around producers
   feeding attention kernels;
2. a confidence-pairformer precision/tiling campaign with explicit numerical
   acceptance criteria;
3. vendor-quality fused epilogues for specific matmul-adjacent patterns.

The larger kernel-engineering projects should start with a
fresh profile, an isolated hotspot screen, and a full same-output N1280/N2560
gate before promotion.
