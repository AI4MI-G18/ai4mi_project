#!/usr/bin/bash
# one architecture with one loss, e.g.
#   ./scripts/00-run-pipeline.sh ENet DiceCE
#   EPOCHS=25 ./scripts/00-run-pipeline.sh ENetImproved Focal
# results land in results/segthor/<arch>-<loss> and volumes/segthor/<arch>-<loss>
export ARCH="${1:-ENet}"
export LOSS="${2:-CE}"
export RUN="$ARCH-$LOSS"

# setup.sh has to be sourced: it activates the venv, and a subshell would throw
# that away without saying so
source ./scripts/pipeline/setup.sh
./scripts/pipeline/slicing.sh
./scripts/pipeline/training.sh
./scripts/pipeline/stitching.sh
