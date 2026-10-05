#!/bin/bash
# The 2.5D U-Net, without and with deep supervision (the _DS network: auxiliary
# losses on the decoder outputs at 1/2 and 1/4 resolution), with Adam and with
# AdamW, on H100s. The 3D U-Net has its own job, job-deep-3d-h100.sh.
#
#     ./jobs/submit.sh                             # builds the data first
#
# One per array task, all on SEGTHOR_PREPROC with DiceCE at EPOCHS=25:
#
#     0   OPT=Adam   ./scripts/00-run-pipeline.sh UNet25D    DiceCE
#     1   OPT=Adam   ./scripts/00-run-pipeline.sh UNet25D_DS DiceCE
#     2   OPT=AdamW  ./scripts/00-run-pipeline.sh UNet25D    DiceCE
#     3   OPT=AdamW  ./scripts/00-run-pipeline.sh UNet25D_DS DiceCE
#
# 2h30: nobody has timed these on an H100. On an A100, without deep supervision, 25
# epochs took 1h54, the stitching and the evaluation included. A fifth more for the
# deep supervision makes 2h17, should the H100 turn out to be no faster at all.
# 16 cores are one GPU's share of an H100 node (64 cores, 4 GPUs).
#
# DATA_ROOT, RESULTS_ROOT and VOLUMES_ROOT say where the slices are read from and
# where the runs and the stitched volumes go (default: this tree). submit.sh names
# them with --export, as sbatch here does not hand the shell's variables on.
#
#SBATCH --job-name=pipeline-deep-25d
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --output=./jobs/logs/%x_%a_%j.out
set -euo pipefail

WORKTREE=$(realpath ./)
cd "$WORKTREE"

echo "== $(date -Is)  job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "== tree    $WORKTREE"
echo "== branch  $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "== no GPU"

# One line per array index, as in job-verify-a100.sh
DATASET=SEGTHOR_PREPROC
LOSS=DiceCE
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) ARCH=UNet25D    ; OPT=Adam  ;;
    1) ARCH=UNet25D_DS ; OPT=Adam  ;;
    2) ARCH=UNet25D    ; OPT=AdamW ;;
    3) ARCH=UNet25D_DS ; OPT=AdamW ;;
    *) echo "array index $SLURM_ARRAY_TASK_ID is not one of 0-3" >&2; exit 1 ;;
esac

export DATASET OPT EPOCHS=25
echo "== task ${SLURM_ARRAY_TASK_ID:-0}: DATASET=$DATASET  $ARCH $LOSS $OPT  $EPOCHS epochs"
time ./scripts/00-run-pipeline.sh "$ARCH" "$LOSS"

echo "== wrote ${RESULTS_ROOT:-results}/preproc/$DATASET/$ARCH-$LOSS (-AdamW for the AdamW runs)"
