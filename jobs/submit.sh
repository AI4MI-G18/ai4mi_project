#!/bin/bash
# Submit the deep supervision runs on H100s: build SEGTHOR_PREPROC, then the U-Net
# (DiceCE) without and with deep supervision, once with Adam and once with AdamW.
# The 2.5D and the 3D U-Net are submitted apart, 4 runs each: the 3D one takes more
# than twice as long.
#
#     ./jobs/submit.sh           # the four 2.5D runs
#     ./jobs/submit.sh 3d        # the four 3D runs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WHICH="${1:-25d}"
case "$WHICH" in
    25d|3d) ;;
    *) echo "usage: $0 [25d|3d]" >&2; exit 1 ;;
esac

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

# The jobs take the directory they are submitted from as the tree to run in.
cd "$HERE/.."

# Submitted by both the 2.5D and the 3D runs: make skips a dataset that is already
# built, so the second one costs a few seconds.
prepare=$(sbatch --parsable --export="$ROOTS" --output="$LOGS/%x_%j.out" \
    "$HERE/job-prepare-data.sh" SEGTHOR_PREPROC)
echo "submitted $prepare (build SEGTHOR_PREPROC)"

# The time limits are in the job scripts, with where they come from.
deep=$(sbatch --parsable --array=0-3 --dependency=afterok:"$prepare" \
    --export="$ROOTS" --output="$LOGS/%x_%a_%j.out" "$HERE/job-deep-$WHICH-h100.sh")
echo "submitted $deep  (4 runs, $WHICH U-Net, held until $prepare succeeds)"

cat <<WATCH

    squeue --me

    tail -f $LOGS/pipeline-prepare_$prepare.out
    ls -t  $LOGS/pipeline-deep-${WHICH}_*.out | head -4

    # the runs, compared on the patients' own voxel grids
    cd $HERE/../
    python scripts/summarize_3d.py --root $RESULTS_ROOT/preproc

    sacct -j $prepare,$deep -X --format=JobID,JobName%18,State,Elapsed,AllocTRES%42

WATCH
