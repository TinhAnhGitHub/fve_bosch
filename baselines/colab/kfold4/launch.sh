#!/bin/bash
# usage: launch.sh SESSION RUNS   (e.g. launch.sh m4a v26:0,v26:1,nocat:0) -- new L4 (or reuse a live one), upload the
# code, run setup.py blocking. The Kaggle token goes only through `colab exec --env`.
S=$1; RUNS=$2; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
up() { for t in 1 2 3 4; do timeout 300 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 90 colab status 2>&1 | grep -q "^\[$S\]" || timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
T=$(mktemp -d); trap 'rm -rf $T' EXIT
echo 'import os; os.makedirs("/content/code", exist_ok=True); print("dirs ok")' > $T/mk.py
for t in 1 2 3; do timeout 120 colab exec -s $S -f $T/mk.py 2>&1 | grep -q "dirs ok" && break; sleep 10; done
for f in atlas.py geom.py colab/run_exps.py out/atlas/queue/score2.py out/atlas/pl_t_s0_ema.pt folds/f3dr_k5_v1.csv; do
  up $B/$f /content/code/$(basename $f); done
echo "uploaded $(date +%H:%M)"
timeout 30000 colab exec -s $S --timeout 29800 --env KAGGLE_API_TOKEN="$(tok KAGGLE_THREE_TOKEN)" --env RUNS="$RUNS" \
  -f $HERE/setup.py 2>&1 | grep -vE "^\s*[│╭╰]" | tail -2
echo "EXEC END $(date +%H:%M)"
