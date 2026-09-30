#!/usr/bin/bash
# every architecture with every loss: 2 x 7 = 14 runs
#   EPOCHS=25 ./scripts/00-run-pipeline-full-matrix.sh
#
# On one dataset, whichever DATASET says (default SEGTHOR_PREPROC). To sweep the
# preprocessing variants instead, loop this:
#   for d in SEGTHOR SEGTHOR_HU SEGTHOR_RESAMPLE SEGTHOR_PREPROC; do
#       DATASET=$d EPOCHS=25 ./scripts/00-run-pipeline-full-matrix.sh
#   done
# which is 56 runs, so price it before starting.
DATASET="${DATASET:-SEGTHOR_PREPROC}"
ARCHS="ENet ENetImproved"
LOSSES="CE TopK Focal Dice DiceCE DiceTopK DiceFocal"

for arch in $ARCHS; do
    for loss in $LOSSES; do
        echo "########## $DATASET-$arch-$loss ##########"
        DATASET="$DATASET" ./scripts/00-run-pipeline.sh "$arch" "$loss"
    done
done
