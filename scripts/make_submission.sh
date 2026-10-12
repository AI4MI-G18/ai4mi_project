#!/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

G=group-18
RUN=UNet3DSmall_DS-DiceFocal-aug_all-sched_poly-s0
FOLDS=(SEGTHOR_PREPROC SEGTHOR_PREPROC_F1 SEGTHOR_PREPROC_F2 SEGTHOR_PREPROC_F3)

rm -rf "$G"
mkdir -p "$G/test/pred" "$G/val/pred" "$G/val/gt"

python scripts/predict_test.py --dest "$G/test/pred" --save_models "$G/bestmodel-$G.pkl"

for d in "${FOLDS[@]}"; do
    cp "volumes/preproc/$d/$RUN-lcc/"*.nii.gz "$G/val/pred/"
done
for p in "$G/val/pred/"*.nii.gz; do
    id=$(basename "$p" .nii.gz)
    cp "data/train/$id/GT.nii.gz" "$G/val/gt/$id.nii.gz"
done

python - "$G" "$RUN" "${FOLDS[@]}" <<'EOF'
import sys
import numpy as np
from pathlib import Path
g, run, folds = sys.argv[1], sys.argv[2], sys.argv[3:]
for metric, name in [("dice3d_val", "dice"), ("hd95_val", "hd95")]:
    merged = {}
    for d in folds:
        fold = np.load(Path("results/preproc", d, f"{run}-lcc", f"{metric}.npz"))
        assert not set(fold.files) & set(merged), "a patient is in two folds"
        merged |= {i: fold[i] for i in fold.files}
    np.savez(Path(g, "val", f"{name}.npz"), **merged)
    arr = np.stack(list(merged.values()))
    print(f"val/{name}.npz: {len(merged)} patients, arrays of shape {arr.shape[1:]}, mean over the 4 organs {arr[:, 1:].mean():.3f}")
EOF

git bundle create "$G/$G.bundle" master

n_test=$(ls data/test/*.nii.gz | wc -l)
echo "== test/pred $(ls "$G/test/pred" | wc -l)/$n_test, val/pred $(ls "$G/val/pred" | wc -l)/40, val/gt $(ls "$G/val/gt" | wc -l)/40"
tar czf "$G.tar.gz" "$G"
echo "== $G.tar.gz: $(du -h "$G.tar.gz" | cut -f1)"
