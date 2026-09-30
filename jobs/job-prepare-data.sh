#!/bin/bash
# Build both datasets the verification needs, before the array trains on them.
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

for DATASET in SEGTHOR SEGTHOR_PREPROC; do
    echo "########## build data/$DATASET ##########"
    time make "data/$DATASET" PROCS=32
    n=$(ls "data/$DATASET/train/img" | wc -l)
    v=$(ls "data/$DATASET/val/img" | wc -l)
    echo "== data/$DATASET: $n train, $v val slices"
    # stitch.py needs this to undo the resampling; SEGTHOR records spacing=null,
    # which is how the pipeline knows there is nothing to undo.
    python -c "import json; d=json.load(open('data/$DATASET/preprocessing.json')); \
print('== hu_window', d['hu_window'], 'spacing', d['spacing'])"
done
