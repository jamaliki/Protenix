#!/usr/bin/env bash
set -euo pipefail

# Submit a focused triangle-attention Nsight Compute capture on Sandpit Tokyo.
# Run this from the remote Protenix checkout.  It writes the tiny environment
# file consumed by the sbatch wrapper, then submits from the run directory so
# relative Slurm log paths stay contained.

REPO=${PROTENIX_REPO:-$(git rev-parse --show-toplevel)}
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
STAMP=$(date -u +%Y%m%d_%H%M%S)

TOKENS=${TOKENS:-251}
BATCH=${BATCH:-32}
ENDING=${ENDING:-false}
CAPTURE=${CAPTURE:-module}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-20}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}

SIDE=start
if [[ "$ENDING" == "1" || "$ENDING" == "true" || "$ENDING" == "True" ]]; then
  SIDE=end
fi

RUN_DIR=${RUN_DIR:-"$REPO/runs/triangle_attention_ncu_${CAPTURE}_${SIDE}_${STAMP}_${COMMIT}"}
mkdir -p "$RUN_DIR/slurm_logs"

cat > "$RUN_DIR/ncu_job_env.sh" <<EOF
export PROTENIX_REPO=$REPO
export RUN_DIR=$RUN_DIR
export TOKENS=$TOKENS
export BATCH=$BATCH
export ENDING=$ENDING
export CAPTURE=$CAPTURE
export WARMUP=$WARMUP
export ITERS=$ITERS
export NCU_ITERS=$NCU_ITERS
export NCU_SET=$NCU_SET
EOF

JOB_ID=$(
  cd "$RUN_DIR"
  sbatch --parsable "$REPO/scripts/perf/tokyo_triangle_attention_ncu.sbatch"
)

echo "job_id=$JOB_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/slurm_logs/prot_tri_ncu-$JOB_ID.out"
echo "err=$RUN_DIR/slurm_logs/prot_tri_ncu-$JOB_ID.err"
squeue -j "$JOB_ID" -o "%.18i %.28j %.10T %.12M %.10l %.12R %.20N"
