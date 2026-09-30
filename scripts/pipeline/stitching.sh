#!/usr/bin/bash

# unresample patients. see how the resampling preprocessng works
PREPROC_ARG=()
if [[ -f "data/$DATASET/preprocessing.json" ]]; then
    PREPROC_ARG=(--preprocessing "data/$DATASET/preprocessing.json")
else
    echo "== no data/$DATASET/preprocessing.json; stitching without undoing any resampling" >&2
fi

python stitch.py --data_folder "$RUN_DIR/best_epoch/val" \
    --dest_folder "$VOL_DIR" \
    --num_classes 255 \
    --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/train/{id_}/GT.nii.gz" \
    "${PREPROC_ARG[@]}"

# Go and check $VOL_DIR/*gz with 3D Slicer / ITK-SNAP.
# They are labelmap, so overlap and get the segmentation you want.
