#!/bin/bash
# usage: launch.sh SESSION  -- T4 session, upload code + data in 20 MB parts, run tools/sam3d_oof.py (blocking).
# Result: out/atlas/kfold/oof/run/SESSION.{log,npz}
S=$1; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; O=$B/out/atlas/kfold/oof; C=$B/out/atlas/cloud/sam3d_all
PARTS=$(mktemp -d); trap 'rm -rf $PARTS' EXIT
for f in $O/kf_v27_e22.npz $O/gt_oof.npz $O/run/h7b2/size_knn.npz $O/run/h7b2/size_knn_axes.npz $C/sam3d_train_s*.npz; do
  split -b 20m -d $f $PARTS/$(basename $f).part; done
up() { for t in 1 2 3 4; do timeout 600 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 300 colab new -s $S --gpu T4 2>&1 | tail -1
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/f3dr/baselines/tools", "/content/f3dr/data")]; print("dirs ok")' > $PARTS/mk.py
timeout 120 colab exec -s $S -f $PARTS/mk.py 2>&1 | tail -1
for f in atlas.py geom.py tools/mixlib.py tools/sam3d_oof.py; do up $B/$f /content/f3dr/baselines/$f; done
for p in $PARTS/*.part*; do up $p /content/f3dr/data/$(basename $p); done
echo "uploaded $(date +%H:%M)"
timeout 14000 colab exec -s $S --timeout 13800 -f $HERE/run.py 2>&1 | tail -4
timeout 300 colab download -s $S /content/f3dr/run.log $O/run/$S.log 2>&1 | tail -1
timeout 300 colab download -s $S /content/f3dr/sam3d_oof.npz $O/run/$S.npz 2>&1 | tail -1
timeout 120 colab stop -s $S 2>&1 | tail -1
echo "DONE $(date +%H:%M)"
