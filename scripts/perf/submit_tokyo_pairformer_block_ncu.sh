#!/usr/bin/env bash
set -euo pipefail

# Submit a focused PairformerBlock Nsight Compute capture on Sandpit Tokyo.
# Run this from the remote Protenix checkout.  It creates a timestamped run
# directory, submits the sbatch script with explicit one-GPU resources, and
# prints the job id plus paths needed for monitoring/parsing.

REPO=${PROTENIX_REPO:-$(git rev-parse --show-toplevel)}
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
STAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-"$REPO/runs/pairformer_block_ncu_${STAMP}_${COMMIT}"}

TOKENS=${TOKENS:-251}
BATCH=${BATCH:-32}
C_S=${C_S:-384}
C_Z=${C_Z:-128}
N_HEADS=${N_HEADS:-16}
HIDDEN_SCALE_UP=${HIDDEN_SCALE_UP:-false}
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}

mkdir -p "$RUN_DIR/slurm_logs"

cat > "$RUN_DIR/ncu_job_env.sh" <<EOF
export PROTENIX_REPO=$REPO
export RUN_DIR=$RUN_DIR
export TOKENS=$TOKENS
export BATCH=$BATCH
export C_S=$C_S
export C_Z=$C_Z
export N_HEADS=$N_HEADS
export HIDDEN_SCALE_UP=$HIDDEN_SCALE_UP
export WARMUP=$WARMUP
export NCU_ITERS=$NCU_ITERS
export NCU_SET=$NCU_SET
EOF

JOB_ID=$(
  cd "$RUN_DIR"
  sbatch --parsable "$REPO/scripts/perf/tokyo_pairformer_block_ncu.sbatch"
)

echo "job_id=$JOB_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/slurm_logs/prot_pair_ncu-$JOB_ID.out"
echo "err=$RUN_DIR/slurm_logs/prot_pair_ncu-$JOB_ID.err"
squeue -j "$JOB_ID" -o "%.18i %.28j %.10T %.12M %.10l %.12R %.20N"
