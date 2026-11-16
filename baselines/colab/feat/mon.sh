#!/bin/bash
# usage: mon.sh [SESSION ...] -- pull /content/feat/run.log of each session and print its last lines
cd "$(dirname "$0")"
for s in ${@:-ft1 ft2}; do
  timeout 120 colab download -s $s /content/feat/run.log $s.run.log > /dev/null 2>&1
  echo "== $s"; tail -3 $s.run.log 2>/dev/null
done
