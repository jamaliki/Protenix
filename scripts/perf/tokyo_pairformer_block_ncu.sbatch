#!/bin/bash
#SBATCH --job-name=prot_pair_ncu
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

# This job profiles exactly one warmed PairformerBlock, not model startup or
# featurization.  The Python harness calls cudaProfilerStart/Stop around the
# block; Nsight Compute is launched with --profile-from-start off so the report
# answers the kernel-boundary question we care about: which launches dominate
# B32/N251 many-sequence throughput after the diffusion path has been optimized?

if [[ -z "${PROTENIX_REPO:-}" || -z "${RUN_DIR:-}" ]]; then
  ENV_FILE=${NCU_JOB_ENV:-${SLURM_SUBMIT_DIR:-$PWD}/ncu_job_env.sh}
  if [[ -f "$ENV_FILE" ]]; then
    # Slurm on Tokyo has held jobs submitted with explicit --export=... before
    # they reached the batch script.  Sourcing a tiny run-local env file keeps
    # submission boring while still recording the exact parameters.
    # shellcheck source=/dev/null
    source "$ENV_FILE"
  fi
fi

PROTENIX_REPO=${PROTENIX_REPO:?set PROTENIX_REPO to the Protenix checkout}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a writable profiling run directory}
PY=${PY:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2/bin/python}
NCU=${NCU:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/bin/ncu}
DEV_BIN=${DEV_BIN:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/dev/bin}
TOKENS=${TOKENS:-251}
BATCH=${BATCH:-32}
LENGTHS=${LENGTHS:-}
C_S=${C_S:-384}
C_Z=${C_Z:-128}
N_HEADS=${N_HEADS:-16}
HIDDEN_SCALE_UP=${HIDDEN_SCALE_UP:-false}
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}

if [[ -n "$LENGTHS" ]]; then
  IFS=',' read -ra LENGTH_ITEMS <<< "$LENGTHS"
  BATCH=${#LENGTH_ITEMS[@]}
  TOKENS=0
  for ITEM in "${LENGTH_ITEMS[@]}"; do
    ITEM=${ITEM//[[:space:]]/}
    if (( ITEM > TOKENS )); then
      TOKENS=$ITEM
    fi
  done
fi

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
export PROTENIX_DISABLE_FAST_LAYER_NORM=${PROTENIX_DISABLE_FAST_LAYER_NORM:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor_cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"

# Keep the pairformer profiling boundary aligned with the promoted inference
# defaults.  These flags preserve vendor CUEQ/cuDNN kernels and only remove
# measured layout/epilogue traffic around them.
export PROTENIX_ATTENTION_FORCE_FP32=${PROTENIX_ATTENTION_FORCE_FP32:-0}
export PROTENIX_TRITON_CUEQ_QKV_LAYOUT=${PROTENIX_TRITON_CUEQ_QKV_LAYOUT:-1}
export PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE=${PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE:-1}
export PROTENIX_TRITON_TRANSITION_DUAL_GEMM=${PROTENIX_TRITON_TRANSITION_DUAL_GEMM:-1}
export PROTENIX_TRITON_TRIANGLE_DIRECT_RECOMPUTE_GATE=${PROTENIX_TRITON_TRIANGLE_DIRECT_RECOMPUTE_GATE:-0}

PY_SHAPE_ARGS=(--batch "$BATCH" --tokens "$TOKENS")
if [[ -n "$LENGTHS" ]]; then
  # Mixed-token v2 profiling must use the real padding mask.  Dense BxNxN
  # profiling disables the direct-recompute path and can point optimization at
  # the wrong launches.
  PY_SHAPE_ARGS=(--lengths "$LENGTHS")
fi

REPORT="$RUN_DIR/pairformer_block_B${BATCH}_N${TOKENS}_Cz${C_Z}_hsu${HIDDEN_SCALE_UP}_${NCU_SET}"

echo "job_id=${SLURM_JOB_ID:-none} host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"
echo "repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
echo "python=$("$PY" --version 2>&1)"
echo "cc=$CC"
echo "ncu=$("$NCU" --version | head -5 | tr '\n' '; ')"
echo "shape=batch:$BATCH tokens:$TOKENS lengths:${LENGTHS:-dense} c_z:$C_Z hidden_scale_up:$HIDDEN_SCALE_UP direct_recompute:$PROTENIX_TRITON_TRIANGLE_DIRECT_RECOMPUTE_GATE"
nvidia-smi

"$NCU" \
  --profile-from-start off \
  --set "$NCU_SET" \
  --target-processes all \
  --force-overwrite \
  --export "$REPORT" \
  "$PY" scripts/perf/pairformer_block_breakdown.py \
    "${PY_SHAPE_ARGS[@]}" \
    --c-s "$C_S" \
    --c-z "$C_Z" \
    --n-heads "$N_HEADS" \
    --hidden-scale-up "$HIDDEN_SCALE_UP" \
    --input-dtype bfloat16 \
    --compute-dtype bfloat16 \
    --triangle-attention cuequivariance \
    --triangle-multiplicative cuequivariance \
    --warmup "$WARMUP" \
    --ncu-capture \
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

echo "run_dir=$RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
