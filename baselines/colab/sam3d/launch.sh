#!/bin/bash
# usage: launch.sh SESSION SHARD   (e.g. launch.sh s3d0 0/3) -- new L4, upload the SAM 3D pipeline, run setup.py (blocking).
# Tokens only through `colab exec --env` (HF_CLI_TOKEN for the gated SAM 3D weights, KAGGLE_THREE_TOKEN for the photos).
S=$1; SHARD=$2; HERE=$(cd "$(dirname "$0")" && pwd); SRC=$HERE/../../sam3d; ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
up() { for t in 1 2 3 4; do timeout 300 colab upload -s $S "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done; echo "UPLOAD FAILED $1"; exit 1; }
timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
echo 'import os; [os.makedirs(d, exist_ok=True) for d in ("/content/sam3d/out", "/content/sam3d/resume")]; print("dirs ok")' > /tmp/claude-1000/mk_sam3d.py
timeout 120 colab exec -s $S -f /tmp/claude-1000/mk_sam3d.py 2>&1 | tail -1
for f in install.sh download_weights.sh gen_sam3d.py fix_flow_matching_model.py fix_dino.py; do up $SRC/$f /content/sam3d/$f; done
K=${SHARD%/*}; PART=$HERE/../../out/atlas/cloud/sam3d_all  # resume material pulled by guard.sh
# in 20 MB parts (uploads fail near 100 MB); setup.py joins them into Fast-SAM3D/out, where gen_sam3d.py resumes
T=$(mktemp -d); for f in train eval; do [ -f $PART/sam3d_${f}_s$K.npz ] && split -b 20m -d $PART/sam3d_${f}_s$K.npz $T/sam3d_${f}_s$K.npz.part; done
for p in $T/*.part*; do [ -f "$p" ] && up $p /content/sam3d/resume/$(basename $p); done; rm -rf $T
echo "uploaded $(date +%H:%M)"
timeout 30000 colab exec -s $S --timeout 29800 --env HF_TOKEN="$(tok HF_CLI_TOKEN)" --env KAGGLE_API_TOKEN="$(tok KAGGLE_THREE_TOKEN)" \
  --env SHARD="$SHARD" -f $HERE/setup.py 2>&1 | tail -3
echo "EXEC END $(date +%H:%M)"
