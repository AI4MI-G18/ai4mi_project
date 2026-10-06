#!/bin/bash
# Verify scripts/00-run-pipeline.sh on four combinations at once, on A100s.
#
#     sbatch --array=0-3 job-verify-a100.sh        # needs the data built first,
#                                                  # see submit.sh
#
# One per array task, all at EPOCHS=25:
#
#     0   DATASET=SEGTHOR          ./scripts/00-run-pipeline.sh ENet CE
#     1   DATASET=SEGTHOR          ./scripts/00-run-pipeline.sh ENet DiceCE
#     2   DATASET=SEGTHOR_PREPROC  ./scripts/00-run-pipeline.sh ENet CE
#     3   DATASET=SEGTHOR_PREPROC  ./scripts/00-run-pipeline.sh ENet DiceCE
#
#SBATCH --job-name=pipeline-verify
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=./jobs/logs/%x_%a_%j.out
set -euo pipefail

WORKTREE=$(realpath ./)
cd "$WORKTREE"

echo "== $(date -Is)  job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "== tree    $WORKTREE"
echo "== branch  $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "== no GPU"

# One line per array index. Written out rather than computed, because four cases
# read more clearly as a list than as arithmetic on the index.
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) DATASET=SEGTHOR         ; LOSS=CE     ;;
    1) DATASET=SEGTHOR         ; LOSS=DiceCE ;;
    2) DATASET=SEGTHOR_PREPROC ; LOSS=CE     ;;
    3) DATASET=SEGTHOR_PREPROC ; LOSS=DiceCE ;;
    *) echo "array index $SLURM_ARRAY_TASK_ID is not one of 0-3" >&2; exit 1 ;;
esac

export DATASET EPOCHS=25
echo "== task ${SLURM_ARRAY_TASK_ID:-0}: DATASET=$DATASET  ENet $LOSS  $EPOCHS epochs"
time ./scripts/00-run-pipeline.sh ENet "$LOSS"

echo "== wrote ${RESULTS_ROOT:-results}/preproc/$DATASET/ENet-$LOSS"
