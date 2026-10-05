#!/usr/bin/bash
set -euo pipefail

time python eval3d.py --pred_folder "$VOL_DIR" \
    --gt_pattern "data/train/{id_}/GT.nii.gz" \
    --dest "$RUN_DIR/dice3d_val.npz" \
    --hd95_dest "$RUN_DIR/hd95_val.npz" \
    --num_classes 5
