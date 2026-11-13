#!/bin/bash
# usage: rebuild2.sh SESSION RUNS -- fresh L4: caches from Kaggle, code upload, launch the frozen fold runs
S=$1; RUNS=$2; cd /tmp/kf
timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
timeout 1500 colab exec -s $S --timeout 1400 --env KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" -f setup.py > setup_$S.log 2>&1
grep -q "SETUP DONE" setup_$S.log || { echo "SETUP FAILED $S"; exit 1; }
timeout 120 colab exec -s $S -f mk2.py > /dev/null 2>&1
for f in code/*; do for t in 1 2 3; do timeout 300 colab upload -s $S $f /content/$f 2>&1 | grep -q Uploaded && break; sleep 3; done; done
timeout 300 colab exec -s $S --timeout 250 --env RUNS=$RUNS -f launch2.py 2>&1 | tail -1
