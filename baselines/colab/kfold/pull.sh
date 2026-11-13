#!/bin/bash
# usage: pull.sh SESSION RUN FIRST LAST -- download RUN_ema_e<FIRST..LAST>.pt and RUN logs into baselines/out/atlas/kfold/
S=$1; R=$2; D=/mnt/c/tinhanh/project/ai_challenge/fve_f3dr/baselines/out/atlas/kfold; mkdir -p $D
for e in $(seq $3 $4); do
  f=${R}_ema_e$e.pt
  [ -s $D/$f ] && continue
  for t in 1 2 3; do timeout 300 colab download -s $S /content/run/out/$f $D/$f > /dev/null 2>&1 && break; sleep 3; done
  [ -s $D/$f ] || echo "MISSING $S $f"
done
timeout 120 colab download -s $S /content/run/out/$R.train.log $D/$R.train.log > /dev/null 2>&1
echo "pulled $R"
