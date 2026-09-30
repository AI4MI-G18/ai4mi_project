#!/usr/bin/bash
# every architecture with every loss: 2 x 7 = 14 runs
#   EPOCHS=25 ./scripts/00-run-pipeline-full-matrix.sh
ARCHS="ENet ENetImproved"
LOSSES="CE TopK Focal Dice DiceCE DiceTopK DiceFocal"

for arch in $ARCHS; do
    for loss in $LOSSES; do
        echo "########## $arch-$loss ##########"
        ./scripts/00-run-pipeline.sh "$arch" "$loss"
    done
done
