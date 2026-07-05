#!/bin/bash
#SBATCH --job-name=prot_diff_ncu
#SBATCH --partition=gpu-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Focused Nsight Compute capture for the diffusion token transformer.
# The Python harness starts CUDA profiling only around one warmed transformer
# variant.  That keeps import time, model construction, pair-bias precompute,
# and Triton/cuDNN setup out of the report, so the launch table answers the
# optimization question: after pair-bias caching, which kernels still own the
# B4/S5 low-sample diffusion-transformer boundary?

if [[ -z "${PROTENIX_REPO:-}" || -z "${RUN_DIR:-}" ]]; then
  ENV_FILE=${NCU_JOB_ENV:-${SLURM_SUBMIT_DIR:-$PWD}/ncu_job_env.sh}
  if [[ -f "$ENV_FILE" ]]; then
    # Avoid sbatch --export on Tokyo: it has previously held jobs before the
    # script ran.  A small run-local env file keeps submission reproducible.
    # shellcheck source=/dev/null
    source "$ENV_FILE"
  fi
fi

PROTENIX_REPO=${PROTENIX_REPO:?set PROTENIX_REPO to the Protenix checkout}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a writable profiling run directory}
PY=${PY:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2/bin/python}
NCU=${NCU:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/bin/ncu}
DEV_BIN=${DEV_BIN:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/dev/bin}
BATCH=${BATCH:-4}
SAMPLES=${SAMPLES:-5}
TOKENS=${TOKENS:-220}
VALID_FRACTION=${VALID_FRACTION:-0.8}
BLOCKS=${BLOCKS:-24}
HEADS=${HEADS:-16}
C_A=${C_A:-768}
C_S=${C_S:-384}
C_Z=${C_Z:-128}
DTYPE=${DTYPE:-bf16}
MODULE_DTYPE=${MODULE_DTYPE:-fp32}
PROTENIX_ATTENTION_FORCE_FP32=${PROTENIX_ATTENTION_FORCE_FP32:-0}
CAPTURE=${CAPTURE:-cached_bias_reuse}
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}
DEFAULT_NCU_METRICS="gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active"
NCU_METRICS=${NCU_METRICS-$DEFAULT_NCU_METRICS}

cd "$PROTENIX_REPO"
mkdir -p "$RUN_DIR" "$RUN_DIR/torchinductor_cache" "$RUN_DIR/triton_cache"

PY_BIN_DIR=${PY%/*}
PY_PREFIX=${PY_BIN_DIR%/*}
NCU_BIN_DIR=${NCU%/*}
export PATH="$DEV_BIN:$NCU_BIN_DIR:$PY_BIN_DIR:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export LD_LIBRARY_PATH="$PY_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CC=${CC:-"$DEV_BIN/x86_64-conda-linux-gnu-gcc"}
export CXX=${CXX:-"$DEV_BIN/x86_64-conda-linux-gnu-g++"}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PROTENIX_ATTENTION_FORCE_FP32
export PROTENIX_DISABLE_FAST_LAYER_NORM=${PROTENIX_DISABLE_FAST_LAYER_NORM:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor_cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"

# Match the current promoted low-sample path.  These broad experimental fusion
# switches were rejected in full gates, so leave them off while profiling the
# real baseline we still need to beat.
unset PROTENIX_TRITON_FUSED_ELEMENTWISE
unset PROTENIX_TRITON_FUSED_ATTENTION_RESIDUAL
unset PROTENIX_TRITON_FUSED_TRANSITION_RESIDUAL
unset PROTENIX_FUSED_TRANSITION_INPUT_PROJECTION

REPORT_SUFFIX=$NCU_SET
if [[ -n "$NCU_METRICS" ]]; then
  REPORT_SUFFIX=${NCU_LABEL:-duration_sol}
fi
REPORT="$RUN_DIR/diffusion_transformer_${CAPTURE}_B${BATCH}_S${SAMPLES}_N${TOKENS}_${REPORT_SUFFIX}"

echo "job_id=${SLURM_JOB_ID:-none} host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"
echo "repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
echo "python=$("$PY" --version 2>&1)"
echo "cc=$CC"
echo "ncu=$("$NCU" --version | head -5 | tr '\n' '; ')"
echo "capture=$CAPTURE batch=$BATCH samples=$SAMPLES tokens=$TOKENS dtype=$DTYPE module_dtype=$MODULE_DTYPE attention_force_fp32=$PROTENIX_ATTENTION_FORCE_FP32 ncu_set=$NCU_SET ncu_metrics=${NCU_METRICS:-<set>}"
nvidia-smi

NCU_ARGS=(
  --profile-from-start off \
  --target-processes all \
  --force-overwrite \
  --export "$REPORT"
)
if [[ -n "$NCU_METRICS" ]]; then
  NCU_ARGS+=(--metrics "$NCU_METRICS")
else
  NCU_ARGS+=(--set "$NCU_SET")
fi

"$NCU" \
  "${NCU_ARGS[@]}" \
  "$PY" scripts/perf/diffusion_pair_bias_cache_probe.py \
    --batch "$BATCH" \
    --samples "$SAMPLES" \
    --tokens "$TOKENS" \
    --valid-fraction "$VALID_FRACTION" \
    --blocks "$BLOCKS" \
    --heads "$HEADS" \
    --c-a "$C_A" \
    --c-s "$C_S" \
    --c-z "$C_Z" \
    --dtype "$DTYPE" \
    --module-dtype "$MODULE_DTYPE" \
    --warmup "$WARMUP" \
    --ncu-capture "$CAPTURE" \
    --ncu-iters "$NCU_ITERS" \
  > "$RUN_DIR/ncu_stdout.txt" \
  2> "$RUN_DIR/ncu_stderr.txt"

"$NCU" \
  --import "$REPORT.ncu-rep" \
  --page summary \
  > "$RUN_DIR/ncu_summary.txt" 2>&1 || true

"$NCU" \
  --import "$REPORT.ncu-rep" \
  --csv --page raw \
  > "$RUN_DIR/ncu_raw.csv" 2> "$RUN_DIR/ncu_raw.err" || true

"$PY" scripts/perf/summarize_ncu_raw.py "$RUN_DIR/ncu_raw.csv" --top 30 \
  > "$RUN_DIR/ncu_kernel_summary.md" 2> "$RUN_DIR/ncu_kernel_summary.err" || true

echo "run_dir=$RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
