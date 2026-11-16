#!/bin/bash
# usage: launch.sh SESSION SIZE_KNN.npz STEP  -- T4 session, upload code + data in 20 MB parts, run the resize test.
# Watch: colab download -s SESSION /content/f3dr/run.log ; result /content/f3dr/rescale.npz
S=$1; SIZE=$2; STEP=$3; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; O=$B/out/atlas/kfold/oof
PARTS=$(mktemp -d); trap 'rm -rf $PARTS' EXIT
split -b 20m -d $O/kf_v27_e22.npz $PARTS/kf_v27_e22.npz.part
split -b 20m -d $O/gt_oof.npz $PARTS/gt_oof.npz.part
up() { for t in 1 2 3 4; do timeout 600 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 300 colab new -s $S --gpu T4 2>&1 | tail -1
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/f3dr/baselines/tools", "/content/f3dr/data")]; print("dirs ok")' > $PARTS/mk.py
timeout 120 colab exec -s $S -f $PARTS/mk.py 2>&1 | tail -1
up $B/atlas.py /content/f3dr/baselines/atlas.py
up $B/geom.py /content/f3dr/baselines/geom.py
up $B/tools/mixlib.py /content/f3dr/baselines/tools/mixlib.py
up $B/tools/size_rescale_oof.py /content/f3dr/baselines/tools/size_rescale_oof.py
up $SIZE /content/f3dr/data/$(basename $SIZE)
for p in $PARTS/*.part*; do up $p /content/f3dr/data/$(basename $p); done
echo "uploaded $(date +%H:%M)"
timeout 14000 colab exec -s $S --timeout 13800 --env SIZE=$(basename $SIZE) --env STEP=$STEP \
  --env RESCALE_KS=${RESCALE_KS:-0.70,0.73,0.76,0.79,0.82,0.85} --env RESCALE_GAMMAS=${RESCALE_GAMMAS:-0.25,0.5,0.75,1.0} -f $HERE/run.py 2>&1 | tail -4
timeout 300 colab download -s $S /content/f3dr/run.log $O/run/$S.log 2>&1 | tail -1
timeout 300 colab download -s $S /content/f3dr/rescale.npz $O/run/$S.npz 2>&1 | tail -1
timeout 120 colab stop -s $S 2>&1 | tail -1
echo "DONE $(date +%H:%M)"
