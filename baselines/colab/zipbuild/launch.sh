#!/bin/bash
# usage: launch.sh SESSION   -- L4: build the mix zips below with tools/mix_zip.py, download to out/atlas/zips/.
# v39r = v39a's 11 members (reproduction check), v45m = + the 5 v26 fold models (epoch 21), v46m = + the 5 NoCat fold
# models (epoch 25). Every fold model is its own member, as in v39a. Polls; reruns when idle (finished zips skipped).
S=$1; HERE=$(cd "$(dirname "$0")" && pwd); B=$HERE/../..; A=$B/out/atlas; K=$A/kfold
SUB=/mnt/c/tinhanh/project/ai_challenge/f3dr_submissions; ENVF=/mnt/c/tinhanh/project/external/zaloai/.env; Z=$A/zips
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
up() { for t in 1 2 3 4; do timeout 600 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
M6="s_posrh_s1_ema.pt s_rh_s2_ema.pt pl_tds_s0_ema.pt pl_ds_s0_ema.pt pl_r_s1_ema.pt s_nocat_s0_ema.pt"
F27=$(echo kf_v27_f{0..4}_ema_e22.pt); F26=$(echo kf_v26_f{0..4}_ema_e21.pt); FNC=$(echo kf_nocat_f{0..4}_ema_e25.pt)
BUILDS="v39r=$M6 $F27;v45m=$M6 $F27 $F26;v46m=$M6 $F27 $F26 $FNC"
mkdir -p $Z; T=$(mktemp -d); trap 'rm -rf $T' EXIT
uv run -q --no-project --with pillow python $HERE/pack_photos.py $B/../data/f3d $T/photos.tar || exit 1
split -b 20m -d $T/photos.tar $T/photos.tar.part && rm $T/photos.tar
timeout 90 colab status 2>&1 | grep -q "^\[$S\]" || timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/f3dr/baselines/tools", "/content/ck", "/content/up")]; print("dirs ok")' > $T/mk.py
for t in 1 2 3; do timeout 120 colab exec -s $S -f $T/mk.py 2>&1 | grep -q "dirs ok" && break; sleep 10; done
for f in atlas.py geom.py tools/mixlib.py tools/mix_zip.py; do up $B/$f /content/f3dr/baselines/$f; done
for p in $T/photos.tar.part*; do up $p /content/up/$(basename $p); done
for c in s_posrh_s1_ema.pt s_rh_s2_ema.pt s_nocat_s0_ema.pt; do up $A/cloud/$c /content/ck/$c; done
for c in pl_tds_s0_ema.pt pl_ds_s0_ema.pt pl_r_s1_ema.pt; do up $SUB/v22_mix3_ds/$c /content/ck/$c; done
for c in $F27 $F26 $FNC; do up $K/$c /content/ck/$c; done
echo "uploaded $(date +%H:%M)"
run() { timeout 14000 colab exec -s $S --timeout 13800 --env HF_TOKEN="$(tok HF_CLI_TOKEN)" --env BUILDS="$BUILDS" \
  -f $HERE/setup.py 2>&1 | grep -vE "^\s*[│╭╰]" | tail -2; }
run &
while true; do
  sleep 120
  timeout 300 colab download -s $S /content/run.log $Z/$S.log > /dev/null 2>&1
  grep -q "ZIPS DONE" $Z/$S.log 2>/dev/null && break
  st=$(timeout 120 colab status 2>&1)
  echo "$st" | grep -q "Aborted" && { echo "SIGN-IN EXPIRED $(date +%H:%M): run ! colab status"; continue; }
  echo "$st" | grep -q "^\[$S\]" || { echo "FAILED: session gone $(date +%H:%M)"; exit 1; }
  if echo "$st" | grep "^\[$S\]" | grep -q IDLE && ! pgrep -f "[c]olab exec -s $S " > /dev/null; then
    echo "idle without result, rerun $(date +%H:%M)"; run & fi
  echo "$(date +%H:%M) $(grep -cE 'predicted 511' $Z/$S.log 2>/dev/null) checkpoint passes, $(grep -c '^BUILD' $Z/$S.log 2>/dev/null) zips"
done
grep -E "^BUILD|clouds from" $Z/$S.log
for n in v39r v45m v46m; do timeout 900 colab download -s $S /content/zips/$n.zip $Z/$n.zip 2>&1 | tail -1; done
timeout 120 colab stop -s $S 2>&1 | tail -1
echo "DONE $(date +%H:%M)"
