#!/usr/bin/env bash
# Plot dice_val.png for every untarred experiment under ../data-final-project/*/results/.
# Run from the repository root, with the .venv activated.

python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-CE/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-CE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-DiceCE/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-DiceCE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-CE/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-CE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceCE/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceCE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-Dice/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-Dice/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceFocal/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceFocal/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceTopK/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceTopK/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-TopK/dice_val.npy --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-TopK/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-CE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-CE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-DiceCE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-DiceCE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-CE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-CE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-DiceCE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-DiceCE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-CE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-CE/dice_val.png
python plot.py --headless --metric_file ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-DiceCE/dice_val.npy --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-DiceCE/dice_val.png

python scripts/merge_dice_val_plots.py ../data-final-project
