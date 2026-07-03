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
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}

mkdir -p "$RUN_DIR/slurm_logs"

EXPORT_VARS="PROTENIX_REPO=$REPO,RUN_DIR=$RUN_DIR,TOKENS=$TOKENS,BATCH=$BATCH,WARMUP=$WARMUP,NCU_ITERS=$NCU_ITERS,NCU_SET=$NCU_SET,HOME=${HOME:-},USER=${USER:-}"

JOB_ID=$(
  sbatch --parsable \
    --chdir="$RUN_DIR" \
    --export="$EXPORT_VARS" \
    "$REPO/scripts/perf/tokyo_pairformer_block_ncu.sbatch"
)

echo "job_id=$JOB_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/slurm_logs/prot_pair_ncu-$JOB_ID.out"
echo "err=$RUN_DIR/slurm_logs/prot_pair_ncu-$JOB_ID.err"
squeue -j "$JOB_ID" -o "%.18i %.28j %.10T %.12M %.10l %.12R %.20N"
