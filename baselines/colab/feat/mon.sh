#!/bin/bash
# usage: mon.sh [SESSION ...] -- pull /content/feat/run.log of each session (fresh copy) and print its last lines
cd "$(dirname "$0")"
for s in ${@:-ft1 ft2}; do
  rm -f $s.run.log
  timeout 120 colab download -s $s /content/feat/run.log $s.run.log > /dev/null 2>&1
  echo "== $s"; [ -f $s.run.log ] && tail -n 3 $s.run.log | tr '\r' '\n' | tail -n 3 || echo "NO LOG (session gone?)"
done
