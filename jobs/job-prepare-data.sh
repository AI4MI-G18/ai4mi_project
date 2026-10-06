#!/bin/bash
# Build both datasets the verification needs, before the array trains on them.
#
# For 3D you do
# sbatch jobs/job-prepare-data.sh SEGTHOR_PREPROC SEGTHOR_PREPROC_Z
#
# Idempotent: make skips a dataset that is already built, so resubmitting is free.
#SBATCH --job-name=pipeline-prepare
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=./jobs/logs/%x_%j.out
set -euo pipefail

WORKTREE=$(realpath ./)
cd "$WORKTREE"

echo "== $(date -Is)  job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "== tree    $WORKTREE"
echo "== branch  $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"

# setup.sh makes the data/train symlink and activates the venv. Sourced, not run:
# a subshell would throw the venv away silently.
source ./scripts/pipeline/setup.sh
echo "== python  $(command -v python)"

if (( $# )); then
    DATASETS=("$@")
else
    DATASETS=(SEGTHOR SEGTHOR_PREPROC)
fi

# submit.sh points this at the project space; the default is the worktree's data/.
DATA_ROOT="${DATA_ROOT:-data}"

for DATASET in "${DATASETS[@]}"; do
    echo "########## build $DATA_ROOT/$DATASET ##########"
    time make "$DATA_ROOT/$DATASET" DATA_ROOT="$DATA_ROOT" PROCS=32
    n=$(ls "$DATA_ROOT/$DATASET/train/img" | wc -l)
    v=$(ls "$DATA_ROOT/$DATASET/val/img" | wc -l)
    echo "== $DATA_ROOT/$DATASET: $n train, $v val slices"
    # stitch.py needs this to undo the resampling; SEGTHOR records spacing=null,
    # which is how the pipeline knows there is nothing to undo.
    python -c "import json; d=json.load(open('$DATA_ROOT/$DATASET/preprocessing.json')); \
print('== hu_window', d['hu_window'], 'spacing', d['spacing'])"
done
