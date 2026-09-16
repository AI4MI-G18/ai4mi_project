#!/usr/bin/bash

# download the data

# missing exact command from the readme: verify file integraty and unzip
#make segthor_part1

# make slicers(?)
#make TOY2
#make SEGTHOR

# run the 2D view, see the doc
#python ...

# run the 3D view. Some issues (segthor_train -> segthor_part1, and code), see the git log
python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 \
    --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_part1/train/{id_}/GT.nii.gz"

# Go and check volumes/segthor/ce/*gz with 3D Slicer / ITK-SNAP.
#$ ls volumes/segthor/ce/
#Patient_01.nii.gz  Patient_11.nii.gz  Patient_15.nii.gz  Patient_17.nii.gz  Patient_19.nii.gz
# They are labelmap, so overlap and get the segmentation you want.

