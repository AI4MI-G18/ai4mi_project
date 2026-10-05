#!/usr/bin/env bash
# Plot loss.png (training + validation loss) for every untarred experiment under ../data-final-project/*/results/.
# Run from the repository root, with the .venv activated.

python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-CE --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-CE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-DiceCE --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR/ENet-DiceCE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-CE --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-CE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceCE --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceCE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceFocal --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceFocal/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-Dice --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-Dice/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceTopK --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-DiceTopK/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-TopK --dest ../data-final-project/phase-03-pipeline-v0.2-production-run/results/preproc/SEGTHOR_PREPROC/ENet-TopK/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-CE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-CE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-DiceCE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet25D-DiceCE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-CE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-CE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-DiceCE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet3D-DiceCE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-CE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-CE/loss.png
python plot_losses.py --results-dir ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-DiceCE --dest ../data-final-project/phase-04-UNet-pipeline/results/preproc/SEGTHOR_PREPROC/UNet-DiceCE/loss.png

python scripts/merge_loss_plots.py ../data-final-project
