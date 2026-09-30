#!/usr/bin/bash
# source this, do not run it: the venv activation below would die with the subshell

# download the data manually with uva credential

# go to project folder

# missing exact command from the readme: verify file integraty and unzip
#make segthor_part1

cd data
ln -sfn ../../../archive-data/segthor_train_full/train ./

cd ../
source ../../ai4mi_project/.venv/bin/activate

