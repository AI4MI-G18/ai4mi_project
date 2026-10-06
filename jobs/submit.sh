#!/bin/bash
# Submit the deep supervision runs on H100s: build SEGTHOR_PREPROC, then the U-Net
# (DiceCE) without and with deep supervision, once with Adam and once with AdamW.
# The 2.5D and the 3D U-Net are submitted apart, 4 runs each: the 3D one takes more
# than twice as long.
#
#     ./jobs/submit.sh           # the four 2.5D runs
#     ./jobs/submit.sh 3d        # the four 3D runs
#
# With --dryrun the same runs go to a CPU partition for two epochs each: enough to
# see the whole pipeline through, from the slices to the 3D evaluation, without
# waiting for a GPU or paying for one.
#
#     ./jobs/submit.sh --dryrun
#     ./jobs/submit.sh --dryrun 3d
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRYRUN=
WHICH=25d
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN=1 ;;
        25d|3d) WHICH="$arg" ;;
        *) echo "usage: $0 [--dryrun] [25d|3d]" >&2; exit 1 ;;
    esac
done

# Everything the run writes goes to the project space, not the worktree: the logs,
# and under results/ the sliced datasets, the runs and the stitched volumes.
OUT=/gpfs/scratch1/shared/scur0075/261006-01-dryrun
LOGS="$OUT/logs"
RESULTS_ROOT="$OUT/results"
DATA_ROOT="$RESULTS_ROOT/data"
VOLUMES_ROOT="$RESULTS_ROOT/volumes"
# Slurm does not create the log directory, and a job with nowhere to log dies silently.
mkdir -p "$LOGS" "$DATA_ROOT" "$VOLUMES_ROOT"

# Read by the job scripts and scripts/00-run-pipeline.sh. Named on the command line
# because sbatch here does not hand the submitting shell's variables on to the job.
ROOTS="ALL,RESULTS_ROOT=$RESULTS_ROOT,DATA_ROOT=$DATA_ROOT,VOLUMES_ROOT=$VOLUMES_ROOT"

# The dry run: two epochs, on a CPU partition in place of the H100s the job scripts
# ask for, so that no GPU node is held or charged. Options on the command line win
# over the #SBATCH lines, and main.py falls back to the CPU by itself when it finds
# no GPU. genoa rather than rome, where the queue is days long; the data is built
# there too, as the runs wait for it.
PREPARE_OPTS=()
DEEP_OPTS=()
WHAT="4 runs"
if [[ $DRYRUN ]]; then
    EPOCHS=2
    PREPARE_OPTS=(--partition=genoa)
    DEEP_OPTS=(--partition=genoa --gpus=0 --cpus-per-task=32)
    ROOTS+=",EPOCHS=$EPOCHS"
    WHAT="4 dry runs, $EPOCHS epochs on CPUs"
fi

# The jobs take the directory they are submitted from as the tree to run in.
cd "$HERE/.."

# Submitted by both the 2.5D and the 3D runs: make skips a dataset that is already
# built, so the second one costs a few seconds.
prepare=$(sbatch --parsable "${PREPARE_OPTS[@]}" --export="$ROOTS" --output="$LOGS/%x_%j.out" \
    "$HERE/job-prepare-data.sh" SEGTHOR_PREPROC)
echo "submitted $prepare (build SEGTHOR_PREPROC)"

# The time limits are in the job scripts, with where they come from: timed for 25
# epochs on a GPU, not for a dry run.
deep=$(sbatch --parsable --array=0-3 --dependency=afterok:"$prepare" "${DEEP_OPTS[@]}" \
    --export="$ROOTS" --output="$LOGS/%x_%a_%j.out" "$HERE/job-deep-$WHICH-h100.sh")
echo "submitted $deep  ($WHAT, $WHICH U-Net, held until $prepare succeeds)"

cat <<WATCH

    squeue --me

    tail -f $LOGS/pipeline-prepare_$prepare.out
    ls -t  $LOGS/pipeline-deep-${WHICH}_*.out | head -4

    # the runs, compared on the patients' own voxel grids
    cd $HERE/../
    python scripts/summarize_3d.py --root $RESULTS_ROOT/preproc

    sacct -j $prepare,$deep -X --format=JobID,JobName%18,State,Elapsed,AllocTRES%42

WATCH
