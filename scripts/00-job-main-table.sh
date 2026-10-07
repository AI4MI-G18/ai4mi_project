#!/bin/bash
set -euo pipefail

WORKTREE=$(realpath ./)
cd "$WORKTREE"

echo "== $(date -Is)  job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "== tree    $WORKTREE"
echo "== branch  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') at $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "== no GPU"

# AUG and SCHED as main.py takes them (none: left out of the run's name); DS through the _DS networks
#        DATASET          ARCH                LOSS       AUG   SCHED
CONFIGS=("SEGTHOR         ENet                CE         none  none"   # 0 baseline
         "SEGTHOR         ENet                DiceFocal  none  none"   # 1 + loss only
         "SEGTHOR_PREPROC ENet                CE         none  none"   # 2 + preprocessing only
         "SEGTHOR_PREPROC ENet                DiceFocal  all   poly"   # 3 our recipe, baseline network
         "SEGTHOR_PREPROC ENetImproved_DS     DiceFocal  all   poly"   # 4
         "SEGTHOR_PREPROC ENetImproved25D_DS  DiceFocal  all   poly"   # 5
         "SEGTHOR_PREPROC ENetImproved3D_DS   DiceFocal  all   poly"   # 6
         "SEGTHOR_PREPROC UNetSmall_DS        DiceFocal  all   poly"   # 7
         "SEGTHOR_PREPROC UNet25DSmall_DS     DiceFocal  all   poly"   # 8
         "SEGTHOR_PREPROC UNet3DSmall_DS      DiceFocal  all   poly")  # 9
SEEDS=3

i="${SLURM_ARRAY_TASK_ID:?submit it as an array, e.g. sbatch --array=0-29 jobs/job-main-table.sh}"
n=${#CONFIGS[@]}
if (( i >= n * SEEDS )); then
    echo "no task $i: $n configs x $SEEDS seeds = tasks 0-$(( n * SEEDS - 1 ))" >&2
    exit 1
fi
read -r DATASET ARCH LOSS AUG SCHED <<< "${CONFIGS[i % n]}"
SEED=$(( i / n ))
if [[ $AUG == none ]]; then AUG=""; fi
if [[ $SCHED == none ]]; then SCHED=""; fi

export DATASET SEED AUG SCHED EPOCHS=25
echo "== task $i: DATASET=$DATASET  $ARCH $LOSS  seed $SEED  $EPOCHS epochs  AUG=${AUG:-none}  SCHED=${SCHED:-none}"
time ./scripts/00-run-pipeline.sh "$ARCH" "$LOSS"

echo "== wrote results/preproc/$DATASET/$ARCH-$LOSS${AUG:+-aug_$AUG}${SCHED:+-sched_$SCHED}-s$SEED (and its -lcc)"
