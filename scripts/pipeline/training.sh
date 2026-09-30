#!/usr/bin/bash

# dry-run for development. EPOCHS=25 for a real run
time python main.py --dataset SEGTHOR --arch "$ARCH" --loss "$LOSS" --mode full \
    --epoch "${EPOCHS:-2}" --dest "results/segthor/$RUN" --gpu
