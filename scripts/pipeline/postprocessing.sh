#!/usr/bin/bash
set -euo pipefail

# Keep the largest connected component of every organ.
# Saved next to original predictions, both evaluated and compared
python postprocess.py --pred_folder "$VOL_DIR" --dest_folder "$VOL_DIR-lcc"
