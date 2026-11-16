#!/bin/bash
# usage: launch.sh SESSION "SPEC [SPEC ...]"  -- new L4 session, upload the extractor, download photos, start extraction.
# Tokens go only through `colab exec --env` (HF_CLI_TOKEN; KAGGLE_THREE_TOKEN for account 3's photo datasets).
S=$1; SPECS=$2; HERE=$(cd "$(dirname "$0")" && pwd); ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
timeout 300 colab new -s $S --gpu L4 2>&1 | tail -1
timeout 120 colab exec -s $S -f $HERE/mk.py 2>&1 | tail -1
for t in 1 2 3; do timeout 300 colab upload -s $S $HERE/../../tools/extract_feats.py /content/feat/extract_feats.py 2>&1 | grep -q Uploaded && break; sleep 3; done
timeout 1500 colab exec -s $S --timeout 1400 --env HF_TOKEN="$(tok HF_CLI_TOKEN)" --env KAGGLE_API_TOKEN="$(tok KAGGLE_THREE_TOKEN)" \
  --env SPECS="$SPECS" -f $HERE/setup.py 2>&1 | tail -6
