#!/usr/bin/bash

# dry-run for development
time python main.py --dataset SEGTHOR --mode full --epoch 2 --dest results/segthor/ce --gpu

#time python main.py --dataset SEGTHOR --mode full --epoch 25 --dest results/segthor/ce --gpu

