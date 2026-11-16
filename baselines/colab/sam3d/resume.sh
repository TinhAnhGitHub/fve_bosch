#!/bin/bash
# usage: resume.sh SESSION SHARD -- rerun setup.py on a live session whose files are already uploaded (it skips done steps).
S=$1; SHARD=$2; HERE=$(cd "$(dirname "$0")" && pwd); ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
tok() { grep "^$1=" $ENVF | cut -d= -f2- | tr -d "\"'\r "; }
timeout 30000 colab exec -s $S --timeout 29800 --env HF_TOKEN="$(tok HF_CLI_TOKEN)" --env KAGGLE_API_TOKEN="$(tok KAGGLE_THREE_TOKEN)" \
  --env SHARD="$SHARD" -f $HERE/setup.py 2>&1 | tail -3
echo "EXEC END $(date +%H:%M)"
