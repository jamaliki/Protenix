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

PROTENIX_REPO=${PROTENIX_REPO:?set PROTENIX_REPO to the Protenix checkout}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a writable profiling run directory}
PY=${PY:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2/bin/python}
NCU=${NCU:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-nsight/bin/ncu}
TOKENS=${TOKENS:-251}
BATCH=${BATCH:-32}
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}

cd "$PROTENIX_REPO"
mkdir -p "$RUN_DIR" "$RUN_DIR/torchinductor_cache" "$RUN_DIR/triton_cache"

PY_BIN_DIR=${PY%/*}
PY_PREFIX=${PY_BIN_DIR%/*}
NCU_BIN_DIR=${NCU%/*}
export PATH="$NCU_BIN_DIR:$PY_BIN_DIR:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export LD_LIBRARY_PATH="$PY_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor_cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"

# Keep the pairformer profiling boundary aligned with the promoted inference
# defaults.  These flags preserve vendor CUEQ/cuDNN kernels and only remove
# measured layout/epilogue traffic around them.
export PROTENIX_ATTENTION_FORCE_FP32=${PROTENIX_ATTENTION_FORCE_FP32:-0}
export PROTENIX_TRITON_CUEQ_QKV_LAYOUT=${PROTENIX_TRITON_CUEQ_QKV_LAYOUT:-1}
export PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE=${PROTENIX_TRITON_TRIANGLE_ATTENTION_EPILOGUE:-1}

echo "job_id=${SLURM_JOB_ID:-none} host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"
echo "repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
echo "python=$("$PY" --version 2>&1)"
echo "ncu=$("$NCU" --version | head -5 | tr '\n' '; ')"
nvidia-smi

"$NCU" \
  --profile-from-start off \
  --set "$NCU_SET" \
  --target-processes all \
  --force-overwrite \
  --export "$RUN_DIR/pairformer_block_B${BATCH}_N${TOKENS}_${NCU_SET}" \
  "$PY" scripts/perf/pairformer_block_breakdown.py \
    --batch "$BATCH" \
    --tokens "$TOKENS" \
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
  --import "$RUN_DIR/pairformer_block_B${BATCH}_N${TOKENS}_${NCU_SET}.ncu-rep" \
  --page summary \
  > "$RUN_DIR/ncu_summary.txt" 2>&1 || true

"$NCU" \
  --import "$RUN_DIR/pairformer_block_B${BATCH}_N${TOKENS}_${NCU_SET}.ncu-rep" \
  --csv --page raw \
  > "$RUN_DIR/ncu_raw.csv" 2> "$RUN_DIR/ncu_raw.err" || true

echo "run_dir=$RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
