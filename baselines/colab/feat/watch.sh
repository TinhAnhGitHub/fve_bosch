#!/bin/bash
# usage: watch.sh SESSION NAME -- every 5 min pull run.log; when NAME.npz is saved, download it to out/atlas/feats/
cd "$(dirname "$0")"; S=$1; N=$2; OUT=../../out/atlas/feats
while true; do
  rm -f $S.run.log; timeout 120 colab download -s $S /content/feat/run.log $S.run.log > /dev/null 2>&1
  if [ ! -f $S.run.log ]; then echo "$(date +%H:%M) $S: no log (session gone?)"; misses=$((misses+1)); [ ${misses:-0} -ge 3 ] && exit 1
  elif grep -q "$N: saved" $S.run.log; then
    for t in 1 2 3; do timeout 1200 colab download -s $S /content/feat/out/$N.npz $OUT/$N.npz 2>&1 | grep -q Downloaded && break; done
    ls -la $OUT/$N.npz && echo "$(date +%H:%M) $S: DONE $N" && exit 0
  else misses=0; echo "$(date +%H:%M) $S: $(grep batch $S.run.log | tail -n 1)"; grep -q "Error\|Traceback" $S.run.log && { tail -n 5 $S.run.log; exit 1; }
  fi
  sleep 300
done
