#!/bin/bash
# usage: guard.sh "m4a=RUNS" "m4b=RUNS" ...  -- keep each M4 session working until its run.log says "M4 EXIT". Every 10 min:
#   pull new EMA snapshots (<run>_ema_e<E>.pt) and train logs into out/atlas/kfold/ (so a lost VM costs little)
#   session gone            -> new VM with the same runs (launch.sh)
#   IDLE, no exec running   -> rerun setup.py (data download is skipped when present)
#   "M4 EXIT" in run.log    -> final pull, stop the VM
#   colab prints "Aborted." -> the CLI sign-in expired: log SIGN-IN EXPIRED (the user runs `! colab status`)
cd "$(dirname "$0")"; D=../../out/atlas/kfold; A=/content/run/f3dr/baselines/out/atlas; mkdir -p $D
pull() {
  for f in $(timeout 120 colab ls -s $1 $A 2>/dev/null | grep -oE "kf_[a-z0-9]+_f[0-9]_ema_e[0-9]+\.pt" | sort -u); do
    [ -s $D/$f ] && continue
    timeout 600 colab download -s $1 $A/$f $D/$f.tmp > /dev/null 2>&1 && mv $D/$f.tmp $D/$f || rm -f $D/$f.tmp
  done
  for f in $(timeout 120 colab ls -s $1 /content/run/out 2>/dev/null | grep -oE "kf_[a-z0-9]+_f[0-9]\.train\.log" | sort -u); do
    timeout 120 colab download -s $1 /content/run/out/$f $D/$f > /dev/null 2>&1; done
}
while true; do
  st=$(timeout 120 colab status 2>&1); left=0
  echo "$st" | grep -q Aborted && { echo "$(date +%H:%M) SIGN-IN EXPIRED: run ! colab status"; sleep 300; continue; }
  for arg in "$@"; do
    S=${arg%%=*}; RUNS=${arg#*=}; [ -f $S.done ] && continue; left=1
    line=$(echo "$st" | grep "^\[$S\]")
    rm -f $S.run.log; timeout 120 colab download -s $S /content/run.log $S.run.log > /dev/null 2>&1
    if grep -q "M4 EXIT" $S.run.log 2>/dev/null; then
      pull $S; timeout 120 colab stop -s $S > /dev/null 2>&1; touch $S.done; echo "$(date +%H:%M) $S DONE, stopped"
    elif [ -z "$line" ] && pgrep -f "[l]aunch.sh $S " > /dev/null; then echo "$(date +%H:%M) $S launching"
    elif [ -z "$line" ]; then
      echo "$(date +%H:%M) $S GONE -> new VM"; setsid nohup ./launch.sh $S $RUNS > $S.launch.log 2>&1 < /dev/null &
    elif echo "$line" | grep -q IDLE && ! pgrep -f "[c]olab exec -s $S " > /dev/null && ! pgrep -f "[l]aunch.sh $S " > /dev/null; then
      echo "$(date +%H:%M) $S IDLE -> rerun"; setsid nohup ./launch.sh $S $RUNS > $S.launch.log 2>&1 < /dev/null &
    else
      pull $S; echo "$(date +%H:%M) $S busy: $(ls $D | grep -cE "^kf_($(echo $RUNS | sed 's/:/_f/g; s/,/|/g'))_ema_e") snapshots here"
    fi
  done
  [ $left -eq 0 ] && { echo "ALL M4 DONE $(date +%H:%M)"; exit 0; }
  sleep 600
done
