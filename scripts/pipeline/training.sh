#!/usr/bin/bash

# dry-run for development. EPOCHS=25 for a real run
time python main.py --dataset "$DATASET" --arch "$ARCH" --loss "$LOSS" --mode full \
    --epoch "${EPOCHS:-2}" --dest "$RUN_DIR" --opt "${OPT:-Adam}" --gpu --seed "${SEED:-0}" \
    --aug "${AUG:-none}" --scheduler "${SCHED:-none}"
