#!/bin/bash
# usage: launch.sh SESSION OOF1.npz,OOF2.npz,...   (names under out/atlas/kfold/oof; first = reference)
# T4, upload code + gt_oof + the OOF files in 20 MB parts, run tools/mix_oof.py, poll for the result.
# Result: out/atlas/kfold/oof/run/SESSION.{log,npz}
S=$1; FILES=$2; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; O=$B/out/atlas/kfold/oof
PARTS=$(mktemp -d); trap 'rm -rf $PARTS' EXIT
for f in gt_oof.npz ${FILES//,/ }; do split -b 20m -d $O/$f $PARTS/$f.part; done
up() { for t in 1 2 3 4; do timeout 600 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 90 colab status 2>&1 | grep -q "^\[$S\]" || timeout 300 colab new -s $S --gpu T4 2>&1 | tail -1
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/f3dr/baselines/tools", "/content/f3dr/data")]; print("dirs ok")' > $PARTS/mk.py
for t in 1 2 3; do timeout 120 colab exec -s $S -f $PARTS/mk.py 2>&1 | grep -q "dirs ok" && break; sleep 10; done
for f in atlas.py geom.py tools/mixlib.py tools/mix_oof.py; do up $B/$f /content/f3dr/baselines/$f; done
for p in $PARTS/*.part*; do up $p /content/f3dr/data/$(basename $p); done
echo "uploaded $(date +%H:%M)"
run() { timeout 14000 colab exec -s $S --timeout 13800 --env FILES="$FILES" -f $HERE/run.py 2>&1 | grep -vE "^\s*[│╭╰]" | tail -2; }
run &
rm -f $O/run/$S.npz
while true; do
  sleep 120
  timeout 300 colab download -s $S /content/f3dr/run.log $O/run/$S.log > /dev/null 2>&1
  timeout 300 colab download -s $S /content/f3dr/mix_oof.npz $O/run/$S.npz > /dev/null 2>&1 && break
  st=$(timeout 120 colab status 2>&1)
  echo "$st" | grep -q "Aborted" && { echo "SIGN-IN EXPIRED $(date +%H:%M): run ! colab status"; continue; }
  echo "$st" | grep -q "^\[$S\]" || { echo "FAILED: session gone $(date +%H:%M)"; exit 1; }
  if echo "$st" | grep "^\[$S\]" | grep -q IDLE && ! pgrep -f "[c]olab exec -s $S " > /dev/null; then
    echo "idle without result, rerun $(date +%H:%M)"; run & fi
  echo "$(date +%H:%M) $(grep scored $O/run/$S.log 2>/dev/null | tail -n 1)"
done
timeout 300 colab download -s $S /content/f3dr/run.log $O/run/$S.log > /dev/null 2>&1
grep -A 20 "CD-L2" $O/run/$S.log
timeout 120 colab stop -s $S 2>&1 | tail -1
echo "DONE $(date +%H:%M)"
