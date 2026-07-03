# Protenix: Protein + X

<div align="center" style="margin: 20px 0;">
  <span style="margin: 0 10px;">⚡ <a href="https://protenix-server.com">Protenix Web Server</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="docs/PTX_V1_Technical_Report_202602042356.pdf">Protenix-v1</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="docs/PX2.pdf">Protenix-v2</a></span>
</div>

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=x)](https://x.com/ai4s_protenix)
[![Slack](https://img.shields.io/badge/Slack-Join-yellow?logo=slack)](https://join.slack.com/t/protenixworkspace/shared_invite/zt-3drypwagk-zRnDF2VtOQhpWJqMrIveMw)
[![Wechat](https://img.shields.io/badge/Wechat-Join-brightgreen?logo=wechat)](https://github.com/bytedance/Protenix/issues/52)
[![Email](https://img.shields.io/badge/Email-Contact-lightgrey?logo=gmail)](#contact-us)
</div>

We’re excited to introduce **Protenix** — Toward High-Accuracy Open-Source Biomolecular Structure Prediction.

Protenix is built for high-accuracy structure prediction. It serves as an initial step in our journey toward advancing accessible and extensible research tools for the computational biology community.

<img src="assets/protenix_predictions.gif" style="width: 100%; height: auto;" alt="Protenix predictions">

## 🌟 Related Projects
- **[PXDesign](https://protenix.github.io/pxdesign/)** is a model suite for de novo protein-binder design built on the Protenix foundation model. PXDesign achieves 20–73% experimental success rates across multiple targets — 2–6× higher than prior SOTA methods such as AlphaProteo and RFdiffusion. The framework is freely accessible via the Protenix Server.

- **[PXMeter](https://github.com/bytedance/PXMeter/)** is an open-source toolkit designed for reproducible evaluation of structure prediction models, released with high-quality benchmark dataset that has been manually reviewed to remove experimental artifacts and non-biological interactions. The associated study presents an in-depth comparative analysis of state-of-the-art models, drawing insights from extensive metric data and detailed case studies. The evaluation of Protenix is based on PXMeter.

- **[Protenix-Dock](https://github.com/bytedance/Protenix-Dock)**: Our implementation of a classical protein-ligand docking framework that leverages empirical scoring functions. Without using deep neural networks, Protenix-Dock delivers competitive performance in rigid docking tasks.

## 🎉 Latest Updates
- **2026-04-08: Protenix-v2 Released** 💪💪 [[Protenix-v2 Technical Report](docs/PX2.pdf)]
  - Protenix-v2 shows clear gains on antibody-antigen structure prediction, together with an additional update in ligand-related plausibility.
- **2026-02-05: Protenix-v1 Released** 💪 [[Protenix-v1 Technical Report](docs/PTX_V1_Technical_Report_202602042356.pdf)]
  - Supported Template/RNA MSA features and improved training dynamics, along with further Inference-time model performance enhancements.
- **2025-11-05: Protenix-v0.7.0 Released** 🚀
  - Introduced advanced diffusion inference optimizations: Shared variable caching, efficient kernel fusion, and TF32 acceleration. See our [performance analysis](./assets/inference_time_vs_ntoken.png).
- **2025-07-17: Protenix-Mini & Constraint Features**
  - Released lightweight model variants ([Protenix-Mini](https://arxiv.org/abs/2507.11839)) that drastically reduce inference costs with minimal accuracy loss.
  - Added support for [atom-level contact and pocket constraints](docs/infer_json_format.md#constraint), enhancing prediction accuracy through physical priors.
- **2025-01-16: Pipeline Enhancements**
  - Open-sourced the full [training data pipeline](./docs/prepare_training_data.md) and [MSA pipeline](./docs/msa_template_pipeline.md).
  - Integrated local [ColabFold-compatible search](./docs/colabfold_compatible_msa.md) for streamlined MSA generation.


## 🚀 Getting Started

### 🛠 Quick Installation

```bash
pip install --upgrade protenix --index-url https://pypi.org/simple
```

If your package mirror lags behind the latest GitHub release, use the official PyPI index above to make sure the installed CLI matches the commands shown in this README.

### 🧬 Quick Prediction

```bash
# Predict structure using a JSON input
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
```

#### Key Model Descriptions
| Model Name | MSA | RNA MSA | Template | Params | Training Data Cutoff | Model Release Date |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `protenix-v2` | ✅ | ✅ | ✅ | 464 M | 2021-09-30 | 2026-04-08 |
| `protenix_base_default_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2021-09-30 | 2026-02-05 |
| `protenix_base_20250630_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2025-06-30 | 2026-02-05 |
| `protenix_base_default_v0.5.0` | ✅ | ❌ | ❌ | 368 M | 2021-09-30 | 2025-05-30 |

- **protenix-v2**: An enhanced-capacity version of the base model, featuring increased representation dimensionality and expanded parameter space (~464M), along with substantial training and optimization improvements.
- **protenix_base_default_v1.0.0**: Base model, trained with a data cutoff aligned with AlphaFold3 (2021-09-30). The total parameter count of protenix_base_default_v1.0.0 is close to that of AlphaFold3.
- **protenix_base_20250630_v1.0.0**: Applied model, trained with an updated data cutoff (2025-06-30) for better practical performance. This model can be used for practical application scenarios.
- **protenix_base_default_v0.5.0**: Previous version of the model, maintained primarily for backward compatibility with users who developed based on v0.5.0.

For a complete list of supported models, please refer to [Supported Models](docs/supported_models.md).

For detailed instructions on installation, data preprocessing, inference, and training, please refer to the [Training and Inference Instructions](docs/training_inference_instructions.md). We recommend users refer to [inference_demo.sh](inference_demo.sh) for detailed inference methods and input explanations.

### ⚡ H100 inference throughput guide for this branch

This branch is optimized for **per-GPU inference throughput**, not minimum
single-job latency.  The measured fast paths keep the normal confidence outputs
enabled.  Use this source checkout, or an editable install from it; an older
`protenix` package from PyPI will not include these branch-specific changes.

```bash
git clone git@github.com:jamaliki/Protenix.git
cd Protenix
pip install -e .
```

Pick the path that matches your workload:

| workload | best way to run it | what provides the speedup |
| --- | --- | --- |
| Many independent designs, usually `N_sample=1-5` | Put JSONs in one input directory, or combine records into one top-level JSON list.  Use `--batch_size 16` for mixed lengths, or `32-64` for tightly bucketed/same-shape 245-token-like cases | Same-shape records use full-model batches; ragged atom shapes and nearby different token lengths share the pairformer trunk plus the diffusion token, atom, and conditioning work |
| Many samples for one design, e.g. `N_sample=1280-2560` | Use one input with a large `-e/--sample` value and the large-sample flags below | Diffusion, atom attention, elementwise gates, and confidence summary avoid repeated work and excess HBM traffic |

#### Recommended path: many independent designs

This is the protein-design campaign path.  Do **not** run a shell loop that
invokes `protenix pred` once per JSON.  Put the one-record JSON files in a
directory and let the runner fill GPU batches:

```bash
protenix pred \
  -i ./campaign_jsons \
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
  --batch_size 16
```

Use `--batch_size 16` as the mixed-length H100 default.  On the profiled
40-220-token campaign it was faster than `8` and `32` because it gives the
diffusion kernels enough work without padding the pairformer trunk too much.
For tightly bucketed or same-shape 245-token `7r6r`-like inputs, `32-64` is the
practical knee: larger batches consume much more memory but add little
throughput.  If you hit OOM, try `--batch_size 8`.

The `5-6x` public CLI speedup below is the exact-shape case: every feature
tensor, including atom-shaped tensors, has the same shape.  Real design
campaigns often have equal token counts but different atom counts.  Those still
batch, but the log line will read
`token-trunk+diffusion-token-atom-batch`, not `exact-model-batch`; the shared
pairformer trunk then becomes the main cost, so the ceiling is lower than the
repeated-`7r6r` headline unless the records are also atom-shape identical.

For the common low-sample campaign range (`N_sample=1-5`), leave
`PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES` at its default `5` so the diffusion tail
can batch records and samples together.  For maximum throughput on H100, also
set `PROTENIX_ATTENTION_FORCE_FP32=0`; this lets full token attention stay in
BF16/cuDNN instead of upcasting q/k/v to FP32.  The diffusion transformer core
now defaults to BF16 on BF16-capable CUDA GPUs because the representative mixed
gate showed its FP32/TF32 GEMMs were the largest remaining diffusion bottleneck;
set `PROTENIX_BF16_DIFFUSION_CORE=0` only for conservative numerical audits.
The atom encoder/decoder now also default to the measured fast boundary on
BF16-capable CUDA: BF16 atom attention, narrow Triton local attention, and fused
local-attention bias production.  This boundary is guarded by dtype, shape,
CUDA, and no-grad checks, so unsupported cases fall back to the original PyTorch
path; set `PROTENIX_BF16_ATOM_ATTENTION=0` and `PROTENIX_TRITON_LOCAL_ATTN=0`
when you need the old conservative atom path.  Do not enable
`PROTENIX_DIFFUSION_TRANSFORMER_SAMPLE_AXIS=1` by default yet: it is useful for
kernel experiments, but the full gate did not show a robust win over the
flattened BF16 path.

The CUEQ pairformer path also defaults to a contiguous ending-attention
producer (`PROTENIX_CUEQ_ENDING_CONTIGUOUS_PRODUCER=1`).  This is a small
throughput win, not a headline feature: it avoids two large pair-tensor copies
around ending-node triangle attention and gave about `+1.5%` on the pairformer
subtotal, or `+0.6%` on the full B32 same-token variable-atom predict gate.  Set
it to `0` only when bisecting a numerical or shape-specific issue.

For normal multi-step inference, the diffusion transformer also caches the
step-invariant token pair-attention bias once per block.  This spends a few GiB
on long mixed batches, but removes repeated pair-bias projection and sample-lane
repeat work from the `N_step=200` denoising loop.  Set
`PROTENIX_CACHE_DIFFUSION_PAIR_BIAS=0` if you are memory-constrained or auditing
the older path; one-step scout runs skip the cache automatically because there
is no reuse window.

Do not set the experimental Triton elementwise/residual/transition flags for
this mixed-campaign path unless you are running a new benchmark.  On the B16
mixed-length gate they slowed the representative predict section from `41.6s`
to `69.3s`; the default branch settings are the measured fast path.

The default `--batch_mode auto` is deliberately conservative but usable for real
sequence-design campaigns.  If all featurized tensor shapes match, the whole
model runs as one exact-shape batch.  If atom counts differ, only the expensive
pairformer trunk is batched.  If token counts also differ, the runner pads
token-shaped tensors inside a length-sorted batch, passes real masks through
template/MSA/pairformer blocks and the token-level diffusion transformer, and
uses a separate atom mask when the diffusion atom encoder/decoder are padded to
the largest atom count in the batch.  This is the practical mixed-sequence path
for protein design campaigns: fake tokens are masked before token attention can
read them, fake atoms are masked before local atom attention can read them as
keys, and fake atoms are excluded from atom-to-token means before coordinates
are cropped back to the original atom counts.  It is not bitwise identical to
singleton inference because BF16/TF32 reductions run at a different physical
length, but padded values are masked from model attention/reduction paths.  Use
`--batch_mode exact --batch_size 1` when exact singleton reproducibility
matters.  See `docs/perf/inference_throughput.md` for the padding deep dive and
reproduction commands.

If most of your concern is the pairformer trunk drift from padding different
token counts, use `--batch_mode trunk_exact --batch_size 16`.  This keeps the
pairformer trunk at each record's real token length and then batches the
diffusion tail.  It is slower than `auto`, but it is the right compromise when
you want mixed-length batching without changing the largest trunk reduction
boundary.  On the 32-record, 40-220-token, `N_sample=5`, `N_step=200` H100 gate,
`trunk_exact` was about `100.8s` predict time versus `41-42s` for default
padded `auto`, so treat it as a numerical-audit or strict-trunk mode rather
than the maximum-throughput setting.

For mixed token lengths, input order matters unless records are packed by
length.  This branch automatically writes a transient length-sorted campaign
JSON before featurization when `--batch_size > 1` and `--batch_mode auto`,
`padded`, or `trunk_exact` are used.  Good logs show high `token_pad_eff` and
tight ranges such as
`Predicting 8 padded-token-trunk (auto) input(s): N_token 184-196,
token_pad_eff 96.9%`.  The optional `--token_bucket_size N` knob additionally
splits the streaming queue into approximate token buckets for advanced callers;
leave it at the default `0` for the normal CLI path, which already length-sorts
the raw campaign records.  On the representative 32-record, 40-220-token,
`N_sample=5`, `N_step=200` H100 gate, explicit buckets of `96`, `64`, `48`,
and `32` tokens were all slower than the sorted default because they fragmented
the diffusion/atom work into too many small launch groups.

Directory inputs are campaign-aware in this branch:

- `-i ./campaign_jsons` is resolved as one campaign, not as many independent
  CLI calls.
- Compatible records are merged into transient campaign JSONs for inference,
  while outputs are still written per design.
- Generated preprocessing siblings such as `foo-update-msa.json` are ignored
  when `foo.json` is present, so rerunning a directory does not silently infer
  both the source and generated JSON.
- Triton-backed CUEQ q/k/v layout conversion and triangle-attention epilogue
  fusion are enabled by default when Triton is installed.  They preserve the
  high-quality GEMM/CUEQ kernels and remove measured layout-copy traffic around
  them.  If Triton is missing, the code falls back safely but runs slower.
- When the shipped CUDA LayerNorm extension cannot build, the BF16/FP16
  inference fallback uses a narrow Triton LayerNorm by default.  FP32,
  training, CPU, non-contiguous tensors, and unsupported feature widths still
  fall back to PyTorch; set `PROTENIX_TRITON_LAYER_NORM=0` to disable it.

Measured one-H100 gates, `N_step=200`, confidence enabled, and normal CIF/JSON
dumping:

| input layout | old behavior | optimized behavior | speedup |
| --- | ---: | ---: | ---: |
| 32 one-record JSON files in a directory | 342.7 s, 0.093 seq/s | 121.5 s, 0.263 seq/s | 2.82x end to end |
| One combined JSON with 32 same-shape records | 361.8 s runner time | 69.3 s runner time | 5.2x after initialization |
| 32 same-token, variable-atom 251-token records | 301.9 s summed predict at `--batch_size 1` | 44.4 s predict, 42.6 s model-forward at `--batch_size 32`; pairformer is 30.1 s | 6.8x predict vs current unbatched path |
| 64 shuffled variable-length proteins, 40-220 tokens, `N_sample=1`, `N_step=1` scout gate | 32.94 s batch-section time, 1.94 records/s | 12.15 s, 5.27 records/s after automatic length sort | 2.71x batch-section throughput |
| 32 variable-length proteins, 40-220 tokens, `N_sample=1`, `N_step=200` | 193.2 s summed predict, 0.166 warm records/s | 33.1 s wall, 29.9 s summed predict, 0.968 records/s at `--batch_size 16` with batched diffusion token+atom+conditioning path | 5.83x single-process throughput |
| 32 variable-length proteins, 40-220 tokens, `N_sample=5`, `N_step=200` | 212.5 s summed predict, 0.753 generated samples/s with the old low-sample boundary | 41.6 s, 3.85 generated samples/s at `--batch_size 16` with flattened sample lanes, BF16 full attention, BF16 diffusion core, BF16 atom attention, Triton local atom attention, default Triton LayerNorm fallback, and cached diffusion pair bias | 5.11x over the old branch boundary |
| Same mixed-token `N_sample=1` workload, many campaign shards on one H100 | 0.166 records/s original low-sample boundary | 1.77 records/s with five `--batch_size 16` workers under CUDA MPS using `/tmp` MPS sockets and `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` | 10.7x per-GPU operational throughput |
| Same mixed-token workload, many campaign shards on one H100 | 0.753 generated samples/s original low-sample boundary | 5.53 generated samples/s with five `--batch_size 16` workers under CUDA MPS using `/tmp` MPS sockets and `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` | 7.3x per-GPU operational throughput |

For same-token, variable-atom campaigns, compare the `predict` or
model-forward section rather than the outer Slurm/script wall time.  Current
code measured closer to `6-7x` for the comparable `B32, N_token=251,
N_sample=1` predict path, while end-to-end wrapper speedups can look much
smaller if the run includes model initialization, cold caches, file dumping, an
older container, or a checkpoint variant that has not been re-benchmarked.
Seeing `token-trunk+diffusion-token-atom-batch` in the log means batching is
working; it does not mean the run is using the exact-shape `7r6r` path.

For the mixed-token, low-sample workload, keep `--batch_size 16` as the current
fixed-batch default: B8 loses diffusion/atom batching efficiency, while B32
loses more to padded trunk and diffusion-transformer shapes.

#### Optional: fill one H100 with multiple campaign workers

For large design campaigns, throughput can improve further by running several
independent `protenix pred` workers on the same allocated H100 under CUDA MPS.
This does not change model math; it overlaps independent campaigns to fill gaps
left by many small launches.  Use this only when you have enough independent
JSON shards and care about aggregate samples/sec more than one shard's latency.

Important details:

- Keep each worker on the same optimized settings above, especially
  `--batch_size 16` for mixed 40-220-token proteins.
- Start CUDA MPS with pipe and log directories on a node-local filesystem such
  as `/tmp`.  Do not put `CUDA_MPS_PIPE_DIRECTORY` on Lustre; the MPS control
  socket may fail there.
- In the current H100 gates, five MPS workers with
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` are the practical knee.  On the mixed
  `N_sample=1`, `N_step=200` workload they reached `1.77` records/s, which is
  about `10.7x` over the original low-sample branch boundary (`0.166`
  records/s).  On the mixed `N_sample=5`, `N_step=200` workload they reached
  `5.53` generated samples/s, `1.44x` over the prior single-process promoted
  rate (`3.85` generated samples/s) and about `7.3x` over the original
  low-sample branch boundary (`0.753` generated samples/s).  Seven workers at
  `14%` reached `5.55` generated samples/s once for `N_sample=5`, but that is
  within noise while adding more memory pressure and per-shard latency.
- No-MPS multi-process concurrency helped only modestly (`4.19` generated
  samples/s at four workers), so use MPS if you choose this operational mode.

Minimal Slurm-job sketch after acquiring one H100:

```bash
export PROTENIX_ATTENTION_FORCE_FP32=0
export PROTENIX_GPU_RANDOM_AUGMENT=1
export PROTENIX_BROADCAST_DIFFUSION_S=1

export CUDA_MPS_PIPE_DIRECTORY="/tmp/protenix_mps_${SLURM_JOB_ID}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/protenix_mps_${SLURM_JOB_ID}/log"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# Launch five shards per H100; run another wave after wait returns.
for shard in shard_0.json shard_1.json shard_2.json shard_3.json shard_4.json; do
  protenix pred -i "$shard" -o "out/${shard%.json}" \
    -s 101 -c 10 -p 200 -e 5 -d bf16 \
    --trimul_kernel cuequivariance \
    --triatt_kernel cuequivariance \
    --enable_cache true \
    --enable_fusion true \
    --enable_tf32 true \
    --batch_size 16 &
done
wait

echo quit | nvidia-cuda-mps-control
```

Quick sanity check: a good campaign run should log messages like
`Predicting 32 same-shape (auto) input(s)` or
`Predicting 32 same-token-trunk (auto) input(s)`.  For mixed sequence lengths,
look for `Predicting ... padded-token-trunk (auto)` plus a high
`token_pad_eff`.  The model timing source should read
`token-trunk+diffusion-token-atom-batch` for the newest mixed-token fast path
or `token-trunk+diffusion-transformer-batch` if atom batching has been disabled.
If it only reads `token-trunk-batch`, check whether
`PROTENIX_BATCH_DIFFUSION_TRANSFORMER=0`,
`N_sample > PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES`, or guidance disabled the
diffusion sharing path.  If atom timings stay large, check
`PROTENIX_BATCH_ATOM_TRANSFORMER=0` or cache-disabled runs.  If you only see
singleton predictions, the inputs are not sharing a compatible trunk boundary or
you are not using this branch's runner.

#### Fast path: many samples from one design

For inference-time scaling on one input, use a large sample count and the
large-sample fast-path flags:

```bash
export PROTENIX_GPU_RANDOM_AUGMENT=1
export PROTENIX_BF16_DIFFUSION_CORE=1
export PROTENIX_ATTENTION_FORCE_FP32=0
# These atom/local-attention flags are defaults on BF16-capable CUDA in this
# branch; exporting them is useful when you want a fully explicit benchmark env.
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

protenix pred \
  -i examples/input.json \
  -o ./output_many_samples \
  -s 101 \
  -c 10 \
  -p 200 \
  -e 1280 \
  -d bf16 \
  --trimul_kernel cuequivariance \
  --triatt_kernel cuequivariance \
  --enable_cache true \
  --enable_fusion true \
  --enable_tf32 true
```

On the representative one-H100 `7r6r` benchmark, the optimized large-sample
configuration measured about `9.18-9.21` samples/sec per GPU at
`N_sample=2560`, compared with `0.528` samples/sec for the original default
`N_sample=5` setting.  Start with `-e 1280` if memory is tight, then try
`-e 2560`.

#### Practical rules for maximum speedup

- Keep `-d bf16`, `--trimul_kernel cuequivariance`,
  `--triatt_kernel cuequivariance`, `--enable_cache true`,
  `--enable_fusion true`, and `--enable_tf32 true` unless you are debugging.
- Leave the LayerNorm policy at its default `auto` setting for BF16/FP16
  inference.  It uses Triton only when the faster C++ extension is unavailable
  and only for the low-precision path that moved the end-to-end H100 gate.
- Throughput comes from filling the GPU.  For campaign inference, prioritize
  length-sorted compatible trunk batches over more Python processes; `auto` will
  still use the faster full-model batch when atom shapes happen to match exactly.
- Do not chase very large `--batch_size` values.  For low-sample campaigns the
  pairformer trunk becomes the bottleneck.  Mixed-length campaigns should start
  at `16`; tightly bucketed 245-token-like campaigns can try `32-64`.
- Keep confidence enabled when comparing against the numbers above.  Disabling
  confidence changes the workload and makes the throughput numbers
  non-comparable.
- The first run may include Triton/JIT and model-loading overhead.  Compare
  sustained campaign throughput or use the benchmark harness when measuring.

For reproduction commands, profiling scripts, implementation notes, roofline
and bottleneck analysis, and negative results that shaped these defaults, see
[docs/perf/inference_throughput.md](docs/perf/inference_throughput.md).  The
benchmark and hotspot tools live under [scripts/perf](scripts/perf/).


### 📊 Benchmark

#### Protenix-v2

Protenix-v2 (refers to the `protenix-v2` model) shows clear gains on antibody-antigen structure prediction, together with an additional update in ligand-related plausibility. Compared to baselines and the earlier Protenix-v1, Protenix-v2 demonstrates a substantial improvement trend. At the DockQ > 0.23 threshold, Protenix-v2 achieves absolute success rate gains of 9 to 13 percentage points over Protenix-v1 across three collections. Remarkably, Protenix-v2 at only 5 seeds already exceeds the performance of Protenix-v1 at 1000 seeds, indicating a clear gain in efficiency.

<img src="./assets/protenix-v2.png" style="width: 100%; height: auto;" alt="Protenix-v2 model Metrics">


#### Protenix-v1

Protenix-v1 (refers to the `protenix_base_default_v1.0.0` model), the first fully open-source model that outperforms AlphaFold3 across diverse benchmark sets while adhering to the same training data cutoff, model scale, and inference budget as AlphaFold3. For challenging targets, such as antigen-antibody complexes, the prediction accuracy of Protenix-v1 can be further enhanced through inference-time scaling – increasing the sampling budget from several to hundreds of candidates leads to consistent log-linear gains.

<img src="./assets/protenix_base_default_v1.0.0_metrics.png" style="width: 100%; height: auto;" alt="protenix-v1 model Metrics">

<img src="./assets/protenix_base_default_v1.0.0_metrics2.png" style="width: 100%; height: auto;" alt="protenix-v1 model Metrics 2">

For detailed benchmark metrics on each dataset, please refer to [docs/model_1.0.0_benchmark.md](docs/model_1.0.0_benchmark.md).

## Citing Protenix

If you use Protenix in your research, please cite the following:

```
@article {Zhang2026.04.10.717613,
	author = {Zhang, Yuxuan and Gong, Chengyue and Sun, Jinyuan and Guan, Jiaqi and Ren, Milong and Xue, Song and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Xiao, Wenzhi},
	title = {Protenix-v2: Broadening the Reach of Structure Prediction and Biomolecular Design},
	elocation-id = {2026.04.10.717613},
	year = {2026},
	doi = {10.64898/2026.04.10.717613},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/04/11/2026.04.10.717613},
	eprint = {https://www.biorxiv.org/content/early/2026/04/11/2026.04.10.717613.full.pdf},
	journal = {bioRxiv}
}

@article {Zhang2026.02.05.703733,
	author = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Guan, Jiaqi and Wang, Lan and Yang, Yanping and Xia, Yu and Xiao, Wenzhi},
	title = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
	elocation-id = {2026.02.05.703733},
	year = {2026},
	doi = {10.64898/2026.02.05.703733},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1},
	eprint = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1.full.pdf},
	journal = {bioRxiv}
}

@article {2025.01.08.631967,
	author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
	title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
	elocation-id = {2025.01.08.631967},
	year = {2025},
	doi = {10.1101/2025.01.08.631967},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967},
	eprint = {https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967.full.pdf},
	journal = {bioRxiv}
}
```

### 📚 Citing Related Work
Protenix is built upon and inspired by several influential projects. If you use Protenix in your research, we also encourage citing the following foundational works where appropriate:
```
@article{abramson2024accurate,
  title={Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick, Joshua and others},
  journal={Nature},
  volume={630},
  number={8016},
  pages={493--500},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@article{ahdritz2024openfold,
  title={OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
  author={Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O’Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccol{\`o} and others},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1514--1524},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  volume={19},
  number={6},
  pages={679--682},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
```

## Contributing to Protenix

We welcome contributions from the community to help improve Protenix!

📄 Check out the [Contributing Guide](CONTRIBUTING.md) to get started.

✅ Code Quality: 
We use `pre-commit` hooks to ensure consistency and code quality. Please install them before making commits:

```bash
pip install pre-commit
pre-commit install
```

🐞 Found a bug or have a feature request? [Open an issue](https://github.com/bytedance/Protenix/issues).



## Acknowledgements


The implementation of LayerNorm operators refers to both [OneFlow](https://github.com/Oneflow-Inc/oneflow) and [FastFold](https://github.com/hpcaitech/FastFold).
We also adopted several [module](protenix/openfold_local/) implementations from [OpenFold](https://github.com/aqlaboratory/openfold), except for [`LayerNorm`](protenix/model/layer_norm/), which is implemented independently.


## Code of Conduct

We are committed to fostering a welcoming and inclusive environment.
Please review our [Code of Conduct](CODE_OF_CONDUCT.md) for guidelines on how to participate respectfully.


## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

The Protenix project including both code and model parameters is released under the [Apache 2.0 License](./LICENSE). It is free for both academic research and commercial use.

## Contact Us

We welcome inquiries and collaboration opportunities for advanced applications of our model, such as developing new features, fine-tuning for specific use cases, and more. Please feel free to contact us at anewbt_mind@bytedance.com.
