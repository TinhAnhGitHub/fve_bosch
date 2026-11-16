#!/bin/bash
# usage: watch.sh SESSION K -- every 15 min pull run.log and the partial shard files (resume material if the VM dies);
# exits 0 after SHARD DONE (and stops the VM), 1 after 3 missed logs or a FAILED line.
cd "$(dirname "$0")"; S=$1; K=$2; OUT=../../out/atlas/cloud/sam3d_all; mkdir -p $OUT
pull() { for f in train eval; do timeout 900 colab download -s $S /content/sam3d/out/sam3d_${f}_s$K.npz $OUT/sam3d_${f}_s$K.npz > /dev/null 2>&1; done; }
while true; do
  rm -f $S.run.log; timeout 120 colab download -s $S /content/sam3d/run.log $S.run.log > /dev/null 2>&1
  if [ ! -f $S.run.log ]; then echo "$(date +%H:%M) $S: no log (session gone?)"; misses=$((misses+1)); [ ${misses:-0} -ge 3 ] && exit 1
  else
    misses=0; echo "$(date +%H:%M) $S: $(grep -E 'saved|###|photos' $S.run.log | tail -n 1 | cut -c1-140)"
    grep -q "FAILED" $S.run.log && { grep -B5 FAILED $S.run.log | tail -8; exit 1; }
    if grep -q "SHARD DONE" $S.run.log; then pull; ls -la $OUT/*_s$K.npz; timeout 120 colab stop -s $S > /dev/null 2>&1; echo "$(date +%H:%M) $S: DONE"; exit 0; fi
    grep -q "gen_sam3d" $S.run.log && pull
  fi
  sleep 900
done
