#!/bin/bash
# usage: pull_h.sh SESSION RUN -- download RUN_ema_e{20..32 step 2}_h.pt (fp16 encoder) and the RUN log
S=$1; R=$2; D=/mnt/c/tinhanh/project/ai_challenge/fve_f3dr/baselines/out/atlas/kfold; mkdir -p $D
for e in 20 22 24 26 28 30 32; do
  f=${R}_ema_e${e}_h.pt
  [ -s $D/$f ] && continue
  for t in 1 2 3; do timeout 900 colab download -s $S /content/run/out/$f $D/$f > /dev/null 2>&1 && break; sleep 3; done
  [ -s $D/$f ] || echo "MISSING $S $f"
done
timeout 120 colab download -s $S /content/run/out/$R.train.log $D/$R.train.log > /dev/null 2>&1
echo "pulled $R"
