#!/bin/bash
#SBATCH --job-name=ptx_epilogue
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/slurm_logs/%x-%j.err

set -euo pipefail

REPO=${PROTENIX_REPO:-/mnt/lustre/users/kiarash-eitgbi/code/protenix_src_main_profile}
RUN_ROOT=${PROTENIX_RUN_ROOT:-/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs}
RUN_NAME=${PROTENIX_RUN_NAME:-transition_epilogue_hotspot_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR="$RUN_ROOT/$RUN_NAME"

PROTENIX_PYTHON=${PROTENIX_PYTHON:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2/bin/python}
DEV_BIN=${DEV_BIN:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/dev/bin}
CUDA_HOME=${CUDA_HOME:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/kaveh}
CUTLASS_INCLUDE_DIR=${CUTLASS_INCLUDE_DIR:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-cutlass/include}

mkdir -p "$RUN_DIR" "$RUN_ROOT/slurm_logs" "$RUN_DIR/torch_extensions" \
  "$RUN_DIR/triton_cache" "$RUN_DIR/torchinductor_cache"

cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PATH="$DEV_BIN:$CUDA_HOME/bin:$(dirname "$PROTENIX_PYTHON"):$PATH"
export CC="$DEV_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$DEV_BIN/x86_64-conda-linux-gnu-g++"
export CUDA_HOME
export CUTLASS_INCLUDE_DIR
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export LAYERNORM_TYPE=normal
export PROTENIX_TRITON_FUSED_ELEMENTWISE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor_cache"

echo "job_id=$SLURM_JOB_ID host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"
echo "repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
echo "run_dir=$RUN_DIR"
echo "python=$($PROTENIX_PYTHON --version 2>&1)"
echo "cuda_home=$CUDA_HOME"
echo "cutlass_include_dir=$CUTLASS_INCLUDE_DIR"
nvidia-smi

ARGS=(
  --samples "${SAMPLES:-1280}"
  --tokens "${TOKENS:-245}"
  --gate-batch "${GATE_BATCH:-1}"
  --c-a "${C_A:-768}"
  --c-s "${C_S:-384}"
  --factor "${FACTOR:-2}"
  --input-dtype "${INPUT_DTYPE:-bfloat16}"
  --compute-dtype "${COMPUTE_DTYPE:-bfloat16}"
  --warmup "${WARMUP:-10}"
  --iters "${ITERS:-30}"
  --seed "${SEED:-123}"
)

if [[ -n "${CANDIDATE:-}" ]]; then
  ARGS+=(--candidate "$CANDIDATE")
fi

printf 'command:'
printf ' %q' "$PROTENIX_PYTHON" scripts/perf/transition_output_epilogue_hotspot.py "${ARGS[@]}"
printf '\n'

"$PROTENIX_PYTHON" scripts/perf/transition_output_epilogue_hotspot.py "${ARGS[@]}" \
  | tee "$RUN_DIR/result.json"
