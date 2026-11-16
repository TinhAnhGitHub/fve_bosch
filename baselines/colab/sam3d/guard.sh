#!/bin/bash
# usage: guard.sh N  -- keep sessions s3d0..s3d<N-1> working until each logs SHARD DONE. Every 5 min:
#   session BUSY            -> fine; every 3rd round pull run.log + partial shard files (resume material)
#   session IDLE, not done  -> rerun setup.py on it (resume.sh; it skips finished steps, gen_sam3d resumes)
#   session gone            -> new VM (launch.sh uploads the pulled partial files first)
#   SHARD DONE              -> final pull, stop the VM
cd "$(dirname "$0")"; N=$1; OUT=../../out/atlas/cloud/sam3d_all; mkdir -p $OUT; round=0
pull() { for f in train eval; do timeout 900 colab download -s s3d$1 /content/sam3d/out/sam3d_${f}_s$1.npz $OUT/sam3d_${f}_s$1.npz.tmp > /dev/null 2>&1 \
  && mv $OUT/sam3d_${f}_s$1.npz.tmp $OUT/sam3d_${f}_s$1.npz; done; }
while true; do
  st=$(timeout 120 colab status 2>&1); left=0
  for k in $(seq 0 $((N-1))); do
    S=s3d$k; [ -f $S.done ] && continue; left=1
    line=$(echo "$st" | grep "^\[$S\]")
    rm -f $S.run.log; timeout 120 colab download -s $S /content/sam3d/run.log $S.run.log > /dev/null 2>&1
    last=$(grep -E 'saved|###|photos' $S.run.log 2>/dev/null | tail -n 1 | cut -c1-110)
    if grep -q "SHARD DONE" $S.run.log 2>/dev/null; then
      pull $k; timeout 120 colab stop -s $S > /dev/null 2>&1; touch $S.done; echo "$(date +%H:%M) $S DONE, stopped"
    elif [ -z "$line" ]; then
      echo "$(date +%H:%M) $S GONE -> new VM"; setsid nohup ./launch.sh $S $k/$N > $S.launch.log 2>&1 < /dev/null &
      sleep 60
    elif echo "$line" | grep -q IDLE; then
      if pgrep -f "colab exec -s $S " > /dev/null; then echo "$(date +%H:%M) $S idle but exec starting"
      else echo "$(date +%H:%M) $S IDLE -> resume ($last)"; setsid nohup ./resume.sh $S $k/$N > $S.launch.log 2>&1 < /dev/null & fi
    else
      echo "$(date +%H:%M) $S busy: $last"; [ $((round % 3)) -eq 2 ] && grep -q gen_sam3d $S.run.log && pull $k
    fi
  done
  [ $left -eq 0 ] && { echo "ALL SHARDS DONE $(date +%H:%M)"; exit 0; }
  round=$((round+1)); sleep 300
done
