#!/bin/bash
# Run SAM 3D on the holdout, then every training photo, then the public-test (eval) photos.
# Results save every 10 photos and a rerun resumes, so it is safe to stop and restart.
# usage: bash run_all.sh [DATA_ROOT] [--low-vram]
#   DATA_ROOT  folder with train/ and eval/ (default: ../../data/f3d, i.e. fve_f3dr/data/f3d)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${1:-$HERE/../../data/f3d}
LOW=${2:-}
PY="$HERE/.venv-sam3d/bin/python"
OUT="$HERE/../out/atlas/cloud"
mkdir -p "$OUT"
[ -f "$HERE/holdout_ids.txt" ] || { echo "missing holdout_ids.txt next to this script"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$HERE/gen_sam3d.py" "$OUT/sam3d_hold.npz" "$DATA/train" --ids "$HERE/holdout_ids.txt" $LOW 2>&1 | tee "$OUT/sam3d_hold.log" | grep --line-buffered -E "saved|FAIL|peak|Error|GEN DONE"
"$PY" "$HERE/gen_sam3d.py" "$OUT/sam3d_train.npz" "$DATA/train" $LOW 2>&1 | tee "$OUT/sam3d_train.log" | grep --line-buffered -E "saved|FAIL|Error|GEN DONE"
"$PY" "$HERE/gen_sam3d.py" "$OUT/sam3d_eval.npz" "$DATA/eval" $LOW 2>&1 | tee "$OUT/sam3d_eval.log" | grep --line-buffered -E "saved|FAIL|Error|GEN DONE"
echo "ALL DONE: $OUT/sam3d_hold.npz, sam3d_train.npz, sam3d_eval.npz"
