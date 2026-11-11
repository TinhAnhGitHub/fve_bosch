#!/bin/bash
# Watch a Colab session and the SAM 3D pipeline. Prints one line per change; exits when the run ends or the session is lost.
# usage: monitor_colab.sh [SESSION] [INTERVAL_SECONDS]
S=${1:-s4}; EVERY=${2:-120}
cd /mnt/c/tinhanh/project/ai_challenge/work_sam3d/colab
LOG=pipeline.log
seen=$(wc -l < $LOG 2>/dev/null || echo 0)
last_remote=""
cat > mon_probe.py <<'PY'
import subprocess
print(subprocess.run(
    "echo GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); "
    "echo GEN $(grep -E 's/image' /content/hold_s3d.log 2>/dev/null | tail -1); "
    "echo FAILS $(grep -c FAIL /content/hold_s3d.log 2>/dev/null); "
    "grep -q 'GEN DONE' /content/hold_s3d.log 2>/dev/null && echo GENDONE; "
    "grep -E 'Traceback|Error' /content/hold_s3d.log 2>/dev/null | tail -1 | sed 's/^/ERR /'",
    shell=True, capture_output=True, text=True).stdout)
PY
echo "$(date +%H:%M) monitoring session $S every ${EVERY}s"
while true; do
  # 1. new pipeline steps
  n=$(wc -l < $LOG 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    tail -n $((n - seen)) $LOG | sed 's/^/pipeline: /'
    seen=$n
  fi
  grep -q "ALL DONE" $LOG 2>/dev/null && { echo "$(date +%H:%M) DONE: results in baselines/out/atlas/cloud/sam3d_hold.npz"; exit 0; }
  grep -q "SESSION LOST\|WEIGHTS FAILED" $LOG 2>/dev/null && { echo "$(date +%H:%M) STOPPED: $(grep -E 'SESSION LOST|WEIGHTS FAILED' $LOG | tail -1)"; exit 1; }
  # 2. is the session still there?
  if ! colab sessions 2>/dev/null | grep -q "^\[$S\]"; then
    echo "$(date +%H:%M) SESSION $S LOST"; exit 1
  fi
  # 3. remote state, reported only when it changes
  r=$(timeout 150 colab exec -s $S --timeout 120 -f mon_probe.py 2>/dev/null | grep -E "^(GPU|GEN|FAILS|GENDONE|ERR)" | tr '\n' ' ')
  if [ -n "$r" ] && [ "$r" != "$last_remote" ]; then
    echo "$(date +%H:%M) remote: $r"
    last_remote=$r
  fi
  sleep $EVERY
done
