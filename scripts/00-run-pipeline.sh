#!/usr/bin/bash
# one dataset with one architecture and one loss, e.g.
#   ./scripts/00-run-pipeline.sh ENet DiceCE
#   EPOCHS=25 ./scripts/00-run-pipeline.sh ENetImproved Focal
#   DATASET=SEGTHOR ./scripts/00-run-pipeline.sh ENet CE      # no preprocessing
#
# DATASET picks how the data was prepared, and is the point of this branch:
#   SEGTHOR            slice only                     (the baseline to beat)
#   SEGTHOR_HU         + [-500, 500] HU window
#   SEGTHOR_RESAMPLE   + 1.5 mm in-plane resampling
#   SEGTHOR_PREPROC    + both                         (the default)
export DATASET="${DATASET:-SEGTHOR_PREPROC}"
export ARCH="${1:-ENet}"
export LOSS="${2:-CE}"

export RUN_DIR="results/preproc/$DATASET/$ARCH-$LOSS"
export VOL_DIR="volumes/preproc/$DATASET/$ARCH-$LOSS"

# setup.sh has to be sourced: it activates the venv, and a subshell would throw
# that away without saying so
source ./scripts/pipeline/setup.sh
./scripts/pipeline/preprocessing.sh
./scripts/pipeline/training.sh
./scripts/pipeline/stitching.sh
# see how the preprocessing works to figure out why we need this
./scripts/pipeline/eval3d.sh
