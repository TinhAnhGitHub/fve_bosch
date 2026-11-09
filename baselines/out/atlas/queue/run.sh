#!/bin/bash
# Each run: train, score against q3_sheet, append to results.log. A finished run is skipped on relaunch.
ROOT=/mnt/c/tinhanh/project/ai_challenge/fve_f3dr
Q=$ROOT/baselines/out/atlas/queue
BASE=q3_sheet
COMMON="--target mm --encoder dinov3l --no-flip-loss"
export HF_HUB_OFFLINE=1
cd $ROOT
run() {
  name=$1; shift
  [ -f $Q/$name.done ] && return
  echo "$(date +%H:%M) START $name: $*" >> $Q/results.log
  uv run -q baselines/atlas.py train $COMMON "$@" --ckpt baselines/out/atlas/$name.pt > $Q/$name.train.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    uv run -q --script $Q/score.py $ROOT $BASE $name > $Q/$name.score.log 2>&1
    rc=$?
    echo "$(date +%H:%M) $(grep -E "^saved" $Q/$name.train.log)" >> $Q/results.log
    grep -E "^$name|^RESULT" $Q/$name.score.log >> $Q/results.log
    [ $rc -eq 0 ] || echo "$(date +%H:%M) FAILED $name score rc=$rc" >> $Q/results.log
  else
    echo "$(date +%H:%M) FAILED $name train rc=$rc" >> $Q/results.log
  fi
  echo $rc > $Q/$name.done
}
run s6_sheet_seed1 --arch sheet --drop-oversized --seed 1
run s5_sheet_cap   --arch sheet --reweight-oversized
echo "$(date +%H:%M) RUNS DONE" >> $Q/results.log
