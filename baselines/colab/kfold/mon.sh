#!/bin/bash
# Pull every train log from the three sessions into /tmp/kf/dl/<session>/ and print each run's last epoch line.
cd /tmp/kf
for s in ${SESSIONS:-kf kf2 kf3}; do
  mkdir -p dl/$s
  for f in $(timeout 60 colab ls -s $s /content/run/out 2>/dev/null | grep -E "train.log|summary.txt"); do
    timeout 120 colab download -s $s /content/run/out/$f dl/$s/$f > /dev/null 2>&1
  done
  for f in dl/$s/*.train.log; do
    [ -f "$f" ] && echo "$s $(basename $f .train.log): $(grep '^epoch' $f | tail -1 | cut -c1-90)"
  done
  [ -s dl/$s/summary.txt ] && sed "s/^/$s summary: /" dl/$s/summary.txt
done
