#!/usr/bin/bash
# run the 2D view, see the doc
#python ...

# run the 3D view. Some issues (segthor_train -> segthor_part1, and code), see the git log
python stitch.py --data_folder "results/segthor/$RUN/best_epoch/val" \
    --dest_folder "volumes/segthor/$RUN" \
    --num_classes 255 \
    --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/train/{id_}/GT.nii.gz"

# Go and check volumes/segthor/$RUN/*gz with 3D Slicer / ITK-SNAP.
#$ ls volumes/segthor/ENet-DiceCE/
#Patient_01.nii.gz  Patient_13.nii.gz  Patient_22.nii.gz  Patient_28.nii.gz  Patient_30.nii.gz
# They are labelmap, so overlap and get the segmentation you want.
