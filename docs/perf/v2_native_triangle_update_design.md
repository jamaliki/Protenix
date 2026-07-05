# Protenix-v2 native triangle-update design

This note is the handoff point from wrapper cleanup to a real native kernel
rewrite for the Protenix-v2 pairformer.  The target workload is Sam-style
protein-design inference on one H100: many independent sequences, `N_sample=1-5`,
mixed token lengths, confidence enabled, BF16, `c_z=256`, and
`hidden_scale_up=True`.

The current best evidence says Python/Triton wrapper cleanup has reached
diminishing returns.  The direct producer-owned benchmark is now positive, but
the long bucket that dominates real v2 throughput only moves one full block by
about four percent.  A production win now needs a native update boundary that
owns the exact-length layout through producer, contraction, output gate, and
residual store.

## Current evidence

Representative sorted v2 buckets:

| bucket | lengths | pair efficiency | cubic efficiency | why it matters |
| --- | --- | ---: | ---: | --- |
| short | `40-124`, two records per length | `48.6%` | `38.7%` | heavily padded; easy to make isolated screens look good |
| long | `136-220`, two records per length | `73.4%` | `64.8%` | controls Sam-style v2 pairformer throughput |

Important measured results:

| boundary | short result | long result | decision |
| --- | ---: | ---: | --- |
| Compact segmented update with custom Triton contraction | full block `0.37-0.38x` | full block `0.21x` | reject; custom contraction is much slower than vendor kernels |
| Exact-group `torch.bmm` contraction only, producer-owned layout assumed | `1.35-1.48x` vs dense contraction | `2.05-2.25x` vs dense contraction | useful ceiling; layout matters |
| Stock CUTLASS/CuTe wrappers | mostly slower; best long contraction `1.09-1.17x` | too small/asymmetric | reject stock adapters |
| Direct producer-owned update after tiled assembly and row-stat tiling | isolated update `~2.1x` | isolated update `~1.25x` | useful benchmark boundary |
| Direct producer-owned full Pairformer block | `1.31x` | repeated `1.043x` | too small for production default |
| Fused output finish benchmark | isolated short update `~3.0x` | isolated long update `~1.26x`, full long block `1.045x` | opt-in learning path only |
| Native cuBLAS contraction + native assembly with the current producer | full update `0.97-0.98x` vs current direct | full update `0.89-0.91x` vs current direct | reject as a deployable path; useful boundary evidence only |
| Fusing update assembly with output row statistics | not tested after long-bucket rejection | long update `0.97x` incoming, `0.93x` outgoing vs fused output | reject; extra assembly-kernel work outweighed the saved stats pass |

The takeaway is narrow but important: exact-length work reduction is real, but
not when implemented as a dense-ABI bridge.  The next kernel must avoid:

- building compact row-major `ab` and then restacking into exact groups;
- per-record or per-feature launch fan-out;
- PyTorch `reshape`/`clone`/`copy_` around `[record * D, N, N]` results;
- dense `masked_fill` over all padded pair rows;
- replacing CUEQ/cuBLAS tensor-core mainloops with scalar or low-occupancy
  custom math.

### First native cuBLAS slice result

Commit `802b1ce` added
`scripts/perf/triangle_update_native_cublas_probe.py` and
`scripts/perf/triangle_update_native_cublas_probe.cu` as the first executable
native slice.  It consumes the direct producer's exact-length groups, calls
`cublasGemmStridedBatchedEx` per exact-length group, and assembles compact
row-major update rows in CUDA.  Job `110868` on one H100
(`runs/native_cublas_triangle_corrected_802b1ce_20260704_235245`) compiled and
passed BF16-valid parity, but the full path still using the current producer
lost to the current direct update:

| bucket | direction | padded CUEQ | current direct | native full with current producer | decision |
| --- | --- | ---: | ---: | ---: | --- |
| short `B16/N124` | incoming | `2.437 ms` | `1.144 ms` | `1.182 ms` (`0.968x` vs direct) | reject full path |
| short `B16/N124` | outgoing | `2.448 ms` | `1.170 ms` | `1.200 ms` (`0.975x` vs direct) | reject full path |
| long `B16/N220` | incoming | `4.121 ms` | `3.327 ms` | `3.718 ms` (`0.895x` vs direct) | reject full path |
| long `B16/N220` | outgoing | `4.232 ms` | `3.403 ms` | `3.727 ms` (`0.913x` vs direct) | reject full path |

The post-producer native boundary is still useful diagnostic evidence:
`native_contract_assemble_only` was `~0.24-0.25 ms` on the short bucket and
`~1.09-1.12 ms` on the long bucket, but once the current producer and output
finish are included it does not beat the tiled direct path.  Do not spend more
time on a standalone cuBLAS wrapper.  The next attempt must own a larger
boundary, especially the producer and output/residual store, or it will keep
paying the same wrapper tax.

### Assembly plus output-stat fusion result

Commit `a525163` briefly added a benchmark-only `fused_stats_output` finish
that fused exact-group update assembly with output LayerNorm row-statistics.
The hypothesis was memory-bound: the existing `fused_output` path first copies
`[record * channel, i, j]` contraction results into row-major compact update
rows, then rereads those rows to compute mean/rstd for the output projection.
Job `110899` on `gpu-canary-0`
(`runs/fused_stats_output_a525163_20260704_235941`) showed the fused assembly
kernel was slower than the saved row-stat pass:

| long `B16/N220` direction | current finish | fused output | fused assembly+stats output | decision |
| --- | ---: | ---: | ---: | --- |
| incoming full update | `3.317 ms` | `3.292 ms` | `3.351 ms` | reject |
| outgoing full update | `3.377 ms` | `3.324 ms` | `3.467 ms` | reject |
| incoming finish-only | `1.791 ms` | `1.717 ms` | `1.817 ms` | reject |
| outgoing finish-only | `1.818 ms` | `1.757 ms` | `1.890 ms` | reject |

The failed path was removed after the screen.  This is useful evidence for the
next larger native boundary: fusing more source-level operations is not enough
if it makes the layout-copy kernel heavier.  The row-stat reduction is cheap
when it is a simple tiled pass; the real target remains the producer and
output/residual boundary around a vendor-quality contraction.

## Boundary to own

The production triangle update computes:

```text
z -> input LayerNorm
  -> gated projections A and B
  -> triangular contraction update[d, i, j] = sum_k A[d, i, k] * B[d, j, k]
  -> output LayerNorm over d
  -> output gate/value projections
  -> residual add into dense z, with invalid padded rows zeroed by contract
```

The next native boundary should own this entire update for one direction
(`outgoing` first, then `incoming`) and one exact-length group:

```text
dense z rows -> producer-owned exact group layout
              -> exact-length contraction
              -> row statistics / output projection / residual store
              -> dense z output
```

It is acceptable for an early screen to use multiple internal kernels if the
interface avoids wrapper copies and the launch count is bounded by exact-length
groups, not by rows or features.  It is not acceptable to expose intermediate
layouts back to PyTorch and then claim a native boundary.

## Layout contract

Group records by exact token length.  For one group of `R` records, length `N`,
and channel count `D=256`, use a producer-owned layout:

```text
lhs: [R, D, N, N] or [R * D, N, N]
rhs: [R, D, N, N] or [R * D, N, N]
```

The exact physical layout is a kernel choice, but it must satisfy:

- the input producer writes it directly from dense `z` and `dense_offsets`;
- the contraction consumes it without extra packing;
- the output path either writes dense `z` directly or keeps update rows in a
  layout that the output LayerNorm/gate can consume without a PyTorch materialize;
- the scatter/residual step writes only valid rows plus explicitly zeroes
  invalid rows needed by downstream dense consumers.

For shuffled campaigns, the compact row order may be grouped by length, but the
final dense store must map back to original record order through `dense_offsets`.

## Candidate implementation ladder

### Slice 1: native grouped update skeleton

Build a CUDA/CuTe extension callable from a benchmark script with this shape:

```python
native_triangle_update(
    z,                # dense [B, Nmax, Nmax, 256] BF16
    dense_offsets,    # grouped valid dense rows
    group_descriptors,# start, count, length, original offsets
    weights,          # LN and projection weights
    direction,        # outgoing first
) -> dense_out
```

Initial constraints:

- BF16 CUDA only.
- Inference/no-grad only.
- `D=256`.
- H100 `sm_90a`.
- sorted/grouped exact lengths matching the current short/long screens.

The first executable version may use separate kernels for:

1. input row statistics;
2. producer into exact group layout;
3. contraction;
4. output row statistics + output projection + dense residual store.

But it must not call PyTorch/CUEQ between those stages.  If any stage needs a
temporary tensor, allocate it inside the benchmark harness and keep it in the
native layout.

### Slice 2: preserve tensor-core contraction speed

The contraction is the dangerous part.  Previous direct Triton contraction
kernels lost by an order of magnitude, and stock CUTLASS adapters did not match
`torch.bmm`.  The first native contraction screen should therefore compare:

- current direct benchmark using exact-group `torch.bmm`;
- native contraction in the same producer-owned layout;
- optional native full update if contraction is not already rejected.

Promotion condition for this slice:

```text
long bucket native contraction >= 0.8x of exact-group torch.bmm
and no extra copy/materialize kernels appear in torch.profiler
```

This is deliberately not enough for production, but it prevents wasting time on
a full fused update whose contraction mainloop is already too slow.

### Slice 3: fuse enough to beat the block

Only after the native contraction is near `torch.bmm` speed should the output
boundary be fused.  The current opt-in fused output finish proves the math is
fusible, but the long full-block movement is only about `0.10 ms`.  A real
native update needs to recover a much larger part of:

- producer row stats and input projections (`~1.55 ms/update` after tiling);
- contraction and assembly (`~0.5-0.8 ms/update`, depending on shape);
- output LayerNorm/gate/scatter (`~0.4-1.8 ms/update`, depending on finish);
- invalid-row dense safety.

The target is not "fewer launches" in isolation.  The target is a full
`PairformerBlock` win on both buckets, especially long `B16/N220`.

## Acceptance gates

Run every candidate on one H100 with the existing benchmark harnesses or direct
successors:

```bash
scripts/perf/triangle_update_producer_owned_direct_probe.py \
  --preset long --direction both --warmup 5 --iters 30

scripts/perf/pairformer_compact_segmented_triangle_update_probe.py \
  --lengths 136,136,160,160,172,172,184,184,196,196,208,208,216,216,220,220 \
  --producer direct --warmup 5 --iters 30
```

Minimum promotion bar for the first production candidate:

| gate | required result |
| --- | --- |
| isolated direct update, short bucket | positive versus padded CUEQ, BF16-valid parity |
| isolated direct update, long bucket | clearly positive versus padded CUEQ, not just within noise |
| full Pairformer block, short bucket | no regression from current direct path |
| full Pairformer block, long bucket | at least double-digit block speedup versus padded CUEQ before end-to-end integration |
| numerical | valid-region max/mean drift comparable to existing BF16 CUEQ/direct screens, no NaNs |
| memory | no material increase in peak reserved memory for B16/N220 |
| maintainability | guarded by dtype/device/shape, default-off until full e2e gates pass |

If the long block only moves by `~1-4%`, do not promote it.  That scale has
already been achieved by wrapper cleanup and is too small to explain a large
Sam-style v2 throughput gain.

## End-to-end gates before defaulting

After a block-level candidate passes, run real Protenix-v2 mixed-token gates:

- 32 records, token lengths `40..220` repeated twice;
- `--batch_size 16`;
- `N_sample=1` and `N_sample=5`;
- `N_step=200`, confidence enabled;
- staged `protenix-v2.pt`;
- compare against current `main`, not an old branch.

Report:

- wall/predict/model-forward seconds;
- records/s and generated samples/s per GPU;
- pairformer, diffusion, confidence subtotals;
- peak allocated/reserved memory;
- output correctness/finite checks.

Only then should a native triangle update become default-on.

## Build environment on Tokyo

Known-good environment shape:

```bash
PY_ENV=/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2
CUDA_HOME=/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/kaveh
DEV_BIN=/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/dev/bin
CUTLASS_INCLUDE_DIR=/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-cutlass/include
export PATH="$DEV_BIN:$PY_ENV/bin:$CUDA_HOME/bin:$PATH"
export CC="$DEV_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$DEV_BIN/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST=9.0a
export PROTENIX_DISABLE_FAST_LAYER_NORM=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

Use one-GPU Slurm jobs with:

```text
--gres=gpu:1 --cpus-per-task=14 --mem=128G
```

## Non-goals

- Do not add another stock CUTLASS grouped wrapper unless it expresses a layout
  not already tested.
- Do not promote short-bucket-only wins for Sam-style v2.
- Do not replace CUEQ/cuBLAS tensor-core contraction with scalar Triton/CUDA
  math and hope fusion compensates.
- Do not claim a per-GPU throughput win from more GPUs.
- Do not drop the confidence head.

## Next concrete coding step

The first native cuBLAS contraction/update-emission slice has now been tested
and rejected as a deployable path because the actual full update with the
current producer is slower than the tiled direct update.  The next coding step
should therefore skip more standalone GEMM wrappers and prototype a larger
native boundary for one direction:

1. input row statistics and producer into exact-length layout;
2. tensor-core contraction in that layout;
3. output row statistics, gate/value projection, residual store, and invalid-row
   handling without returning intermediate update tensors to PyTorch.

The acceptance target is still the long bucket.  If this larger native boundary
does not beat the current direct update on `B16/N220`, reject it before running
full Pairformer gates.
