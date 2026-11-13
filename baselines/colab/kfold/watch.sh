#!/bin/bash
# Every 10 min: ping each kernel (keep-alive), pull logs. Exit (notify) when a summary changes or a session is missing twice in a row.
cd /tmp/kf
before=$(cat dl/*/summary.txt 2>/dev/null | md5sum)
declare -A miss
while true; do
  sleep 600
  for s in kf kf2 kf3; do
    if colab sessions 2>/dev/null | grep -q "^\[$s\]"; then
      miss[$s]=0
      timeout 90 colab exec -s $s --timeout 60 -f ping.py > /dev/null 2>&1
    else
      miss[$s]=$(( ${miss[$s]:-0} + 1 ))
      [ ${miss[$s]} -ge 2 ] && { echo "SESSION LOST $s"; ./mon.sh; exit 0; }
    fi
  done
  ./mon.sh > mon_last.txt 2>&1
  now=$(cat dl/*/summary.txt 2>/dev/null | md5sum)
  [ "$now" != "$before" ] && { cat mon_last.txt; exit 0; }
done
