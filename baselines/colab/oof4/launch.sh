#!/bin/bash
# usage: launch.sh SESSION "name:epoch,..."   (e.g. launch.sh oo4 kf_v26:28,kf_nocat:34) -- L4, upload code + the fold
# snapshots <name>_f<k>_ema_e<epoch>.pt, run setup.py blocking, then poll for the results (a dropped exec connection
# kills the job; setup.py skips finished jobs on rerun). Results: out/atlas/kfold/oof/<name>_e<epoch>.npz
S=$1; JOBS=$2; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; K=$B/out/atlas/kfold; ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
up() { for t in 1 2 3 4; do timeout 300 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 90 colab status 2>&1 | grep -q "^\[$S\]" || timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
T=$(mktemp -d); trap 'rm -rf $T' EXIT
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/f3dr/baselines/tools", "/content/f3dr/baselines/folds", "/content/ck")]; print("dirs ok")' > $T/mk.py
for t in 1 2 3; do timeout 120 colab exec -s $S -f $T/mk.py 2>&1 | grep -q "dirs ok" && break; sleep 10; done
for f in atlas.py geom.py tools/oof.py tools/mixlib.py folds/f3dr_k5_v1.csv; do up $B/$f /content/f3dr/baselines/$f; done
for job in ${JOBS//,/ }; do for k in 0 1 2 3 4; do f=${job%%:*}_f${k}_ema_e${job##*:}.pt; up $K/$f /content/ck/$f; done; done
echo "uploaded $(date +%H:%M)"
run() { timeout 14000 colab exec -s $S --timeout 13800 --env KAGGLE_API_TOKEN="$(tok KAGGLE_THREE_TOKEN)" --env JOBS="$JOBS" \
  -f $HERE/setup.py 2>&1 | grep -vE "^\s*[│╭╰]" | tail -2; }
run &
while true; do
  sleep 120
  timeout 300 colab download -s $S /content/run.log $K/oof/$S.log > /dev/null 2>&1
  grep -q "OOF4 DONE" $K/oof/$S.log 2>/dev/null && break
  st=$(timeout 120 colab status 2>&1)
  echo "$st" | grep -q "Aborted" && { echo "SIGN-IN EXPIRED $(date +%H:%M): run ! colab status"; continue; }
  echo "$st" | grep -q "^\[$S\]" || { echo "FAILED: session gone $(date +%H:%M)"; exit 1; }
  if echo "$st" | grep "^\[$S\]" | grep -q IDLE && ! pgrep -f "[c]olab exec -s $S " > /dev/null; then
    echo "idle without result, rerun $(date +%H:%M)"; run & fi
done
for job in ${JOBS//,/ }; do f=${job%%:*}_e${job##*:}.npz
  timeout 600 colab download -s $S /content/oof/$f $K/oof/$f 2>&1 | tail -1; done
grep -E "^fold|JOB|strict|mean" $K/oof/$S.log | tail -16
timeout 120 colab stop -s $S 2>&1 | tail -1
echo "DONE $(date +%H:%M)"
