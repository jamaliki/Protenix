#!/usr/bin/env bash
set -euo pipefail

# Submit a focused diffusion-transformer Nsight Compute capture on Sandpit
# Tokyo.  Run this from the remote Protenix checkout.  It writes the tiny
# environment file consumed by the sbatch wrapper, then submits from the run
# directory so relative Slurm log paths stay contained.

REPO=${PROTENIX_REPO:-$(git rev-parse --show-toplevel)}
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
STAMP=$(date -u +%Y%m%d_%H%M%S)

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
CAPTURE=${CAPTURE:-cached_bias_reuse}
WARMUP=${WARMUP:-3}
NCU_ITERS=${NCU_ITERS:-1}
NCU_SET=${NCU_SET:-full}
DEFAULT_NCU_METRICS="gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
NCU_METRICS=${NCU_METRICS-$DEFAULT_NCU_METRICS}

RUN_DIR=${RUN_DIR:-"$REPO/runs/diffusion_transformer_ncu_${CAPTURE}_B${BATCH}_S${SAMPLES}_N${TOKENS}_${STAMP}_${COMMIT}"}
mkdir -p "$RUN_DIR/slurm_logs"

cat > "$RUN_DIR/ncu_job_env.sh" <<EOF
export PROTENIX_REPO=$REPO
export RUN_DIR=$RUN_DIR
export BATCH=$BATCH
export SAMPLES=$SAMPLES
export TOKENS=$TOKENS
export VALID_FRACTION=$VALID_FRACTION
export BLOCKS=$BLOCKS
export HEADS=$HEADS
export C_A=$C_A
export C_S=$C_S
export C_Z=$C_Z
export DTYPE=$DTYPE
export CAPTURE=$CAPTURE
export WARMUP=$WARMUP
export NCU_ITERS=$NCU_ITERS
export NCU_SET=$NCU_SET
export NCU_METRICS=$NCU_METRICS
EOF

JOB_ID=$(
  cd "$RUN_DIR"
  sbatch --parsable "$REPO/scripts/perf/tokyo_diffusion_transformer_ncu.sbatch"
)

echo "job_id=$JOB_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/slurm_logs/prot_diff_ncu-$JOB_ID.out"
echo "err=$RUN_DIR/slurm_logs/prot_diff_ncu-$JOB_ID.err"
squeue -j "$JOB_ID" -o "%.18i %.28j %.10T %.12M %.10l %.12R %.20N"
