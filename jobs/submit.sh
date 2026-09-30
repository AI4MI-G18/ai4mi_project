#!/bin/bash
# Submit the verification: build both datasets, then the four runs in parallel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prepare=$(sbatch --parsable "$HERE/job-prepare-data.sh")
echo "submitted $prepare (build SEGTHOR and SEGTHOR_PREPROC)"

verify=$(sbatch --parsable --array=0-3 --dependency=afterok:"$prepare" \
    "$HERE/job-verify-a100.sh")
echo "submitted $verify  (4 runs, held until $prepare succeeds)"

cat <<WATCH

    squeue --me

    tail -f $HERE/logs/pipeline-prepare_$prepare.out
    ls -t  $HERE/logs/pipeline-verify_*.out | head -4

    # the four runs, compared on the patients' own voxel grids
    cd $HERE/../
    python scripts/summarize_3d.py --root results/preproc

    sacct -j $prepare,$verify -X --format=JobID,JobName%16,State,Elapsed,AllocTRES%42

WATCH
