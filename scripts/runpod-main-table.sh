#!/usr/bin/bash
# The tasks of scripts/job-main-table.sh without Slurm on one GPU
set -uo pipefail
cd "$(dirname "$0")/.."

TASKS=("$@")
(( ${#TASKS[@]} )) || TASKS=($(seq 0 29))
N_CONFIGS=10
CONFIGS_3D=" 6 9 "  # rows of ENetImproved3D_DS and UNetSmall3D_DS

if [[ ! -d data/train || -L data/train ]]; then
    echo "data/train has to be a folder with the Patient_XX scans, not a link" >&2
    exit 1
fi
# main.py falls back to CPU without saying with CUDA unavailable, circumvention
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; (torch.ones(1, device='cuda') + 1).item(); \
print('== GPU', torch.cuda.get_device_name(0), '- torch', torch.__version__)" || exit 1

# built once here, because parallel runs would build them
for D in SEGTHOR SEGTHOR_PREPROC; do
    [[ -d data/$D ]] || make "data/$D" PROCS="${PROCS:-8}" || exit 1
done

mkdir -p jobs/logs
run() {
    local log="jobs/logs/main_$1.out"
    if SLURM_ARRAY_TASK_ID=$1 bash scripts/job-main-table.sh > "$log" 2>&1 && ! grep -qE 'Traceback|Killed' "$log"; then
        echo "$(date +%T) done   task $1: $(grep -m1 '^== task' "$log")"
    else
        echo "$(date +%T) FAILED task $1, see $log"
    fi
}
export -f run

tasks_2d=(); tasks_3d=()
for i in "${TASKS[@]}"; do
    if [[ $CONFIGS_3D == *" $(( i % N_CONFIGS )) "* ]]; then tasks_3d+=("$i"); else tasks_2d+=("$i"); fi
done

printf '%s\n' "${tasks_3d[@]}" | xargs -r -P 1 -I{} bash -c 'run {}' &
printf '%s\n' "${tasks_2d[@]}" | xargs -r -P "${P2D:-2}" -I{} bash -c 'run {}'
wait

python scripts/summarize_3d.py --root results/preproc | tee results/main_table.txt
