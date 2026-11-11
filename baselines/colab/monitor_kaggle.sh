#!/bin/bash
# Watch a Kaggle kernel. Prints one line when its status changes; when it ends, downloads its output files.
# usage: monitor_kaggle.sh OWNER/KERNEL [ACCOUNT one|two|three] [INTERVAL_SECONDS] [OUT_DIR]
# Tokens come from ~/.kaggle (account one) or KAGGLE_TWO_TOKEN / KAGGLE_THREE_TOKEN in the zaloai .env.
K=$1; ACC=${2:-three}; EVERY=${3:-300}
OUT=${4:-/mnt/c/tinhanh/project/ai_challenge/fve_f3dr/baselines/out/atlas/cloud/$(basename "$K")}
ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
case $ACC in
  two) export KAGGLE_API_TOKEN=$(grep '^KAGGLE_TWO_TOKEN=' $ENVF | cut -d= -f2-) ;;
  three) export KAGGLE_API_TOKEN=$(grep '^KAGGLE_THREE_TOKEN=' $ENVF | cut -d= -f2-) ;;
esac
last=""
echo "$(date +%H:%M) watching $K every ${EVERY}s"
while true; do
  s=$(kaggle kernels status "$K" 2>&1 | tail -1 | sed 's/.*KernelWorkerStatus\.//; s/"//g')
  if [ -n "$s" ] && [ "$s" != "$last" ]; then echo "$(date +%H:%M) status: $s"; last=$s; fi
  case "$s" in
    COMPLETE|ERROR|CANCEL*)
      mkdir -p "$OUT"
      kaggle kernels output "$K" -p "$OUT" > /dev/null 2>&1
      echo "$(date +%H:%M) $s: output in $OUT: $(ls "$OUT" | tr '\n' ' ')"
      log=$(ls "$OUT"/*.log 2>/dev/null | head -1)
      [ -n "$log" ] && python3 -c "
import json, sys
t = ''.join(x.get('data', '') for x in json.load(open('$log')))
lines = [l for l in t.splitlines() if any(k in l for k in ('saved', 's/image', 'FAIL', 'Error', 'ALL DONE', 'Traceback'))]
print('\n'.join(lines[-8:]))"
      exit 0 ;;
  esac
  sleep "$EVERY"
done
