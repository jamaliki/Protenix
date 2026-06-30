#!/bin/bash
#SBATCH --job-name=ptx_bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/lustre/users/kiarash-eitgbi/code/protenix/runs/slurm_logs/%x-%j.err

set -euo pipefail

REPO=${PROTENIX_REPO:-/mnt/lustre/users/kiarash-eitgbi/code/protenix}
RUN_ROOT=${PROTENIX_RUN_ROOT:-$REPO/runs}
RUN_NAME=${PROTENIX_RUN_NAME:-baseline_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR="$RUN_ROOT/$RUN_NAME"
PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/mnt/lustre/users/kiarash-eitgbi/protenix_data}
PROTENIX_PYTHON=${PROTENIX_PYTHON:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/env-boltz2/bin/python}
DEV_BIN=${DEV_BIN:-/mnt/lustre/users/kiarash-eitgbi/micromamba/envs/dev/bin}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

mkdir -p "$RUN_DIR" "$RUN_ROOT/slurm_logs" "$RUN_DIR/torch_extensions" \
  "$RUN_DIR/triton_cache" "$RUN_DIR/torchinductor_cache"

cd "$REPO"

export PROTENIX_ROOT_DIR
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PATH="$DEV_BIN:$(dirname "$PROTENIX_PYTHON"):$PATH"
export CC="$DEV_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$DEV_BIN/x86_64-conda-linux-gnu-g++"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor_cache"

echo "job_id=$SLURM_JOB_ID host=$(hostname) date_utc=$(date -u --iso-8601=seconds)"
echo "repo=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
echo "run_dir=$RUN_DIR"
echo "python=$($PROTENIX_PYTHON --version 2>&1)"
"$PROTENIX_PYTHON" - <<'PY'
import importlib.util, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ["cuequivariance", "rdkit", "ml_collections", "Bio", "triton"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
nvidia-smi

ARGS=(
  --input-json "${INPUT_JSON:-examples/example.json}"
  --input-indices "${INPUT_INDICES:-}"
  --dump-dir "$RUN_DIR"
  --model-name "${MODEL_NAME:-protenix_base_default_v1.0.0}"
  --seeds "${SEEDS:-101}"
  --repeat "${REPEAT:-1}"
  --min-items-per-rank "${MIN_ITEMS_PER_RANK:-1}"
  --warmup-items "${WARMUP_ITEMS:-0}"
  --n-sample "${N_SAMPLE:-5}"
  --use-default-params "${USE_DEFAULT_PARAMS:-true}"
  --dtype "${DTYPE:-bf16}"
  --use-msa "${USE_MSA:-true}"
  --use-template "${USE_TEMPLATE:-false}"
  --use-rna-msa "${USE_RNA_MSA:-false}"
  --triangle-attention "${TRIANGLE_ATTENTION:-cuequivariance}"
  --triangle-multiplicative "${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
  --enable-cache "${ENABLE_CACHE:-true}"
  --enable-fusion "${ENABLE_FUSION:-true}"
  --enable-tf32 "${ENABLE_TF32:-true}"
  --num-workers "${NUM_WORKERS:-0}"
  --chunk-size "${CHUNK_SIZE:-none}"
  --sample-diffusion-chunk-size "${SAMPLE_DIFFUSION_CHUNK_SIZE:-5}"
  --dump "${DUMP_OUTPUTS:-true}"
  --torch-profile "${TORCH_PROFILE:-false}"
)

if [[ -n "${N_STEP:-}" ]]; then
  ARGS+=(--n-step "$N_STEP" --use-default-params false)
fi
if [[ -n "${N_CYCLE:-}" ]]; then
  ARGS+=(--n-cycle "$N_CYCLE" --use-default-params false)
fi
if [[ -n "${NUM_SEEDS:-}" ]]; then
  ARGS+=(--num-seeds "$NUM_SEEDS")
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  "$PROTENIX_PYTHON" -m torch.distributed.run --standalone \
    --nproc-per-node="$NPROC_PER_NODE" \
    scripts/perf/protenix_inference_benchmark.py "${ARGS[@]}"
else
  "$PROTENIX_PYTHON" scripts/perf/protenix_inference_benchmark.py "${ARGS[@]}"
fi
