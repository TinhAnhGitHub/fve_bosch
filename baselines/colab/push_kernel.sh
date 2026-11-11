#!/bin/bash
# Push one Kaggle kernel that runs several experiments in parallel on its 2 GPUs.
# usage: push_kernel.sh ACCOUNT SLUG AUGSETS name "flags" name "flags" ...
#   ACCOUNT  one, two or three: which Kaggle token to use (KAGGLE_*_TOKEN in the zaloai .env; one = ~/.kaggle/access_token)
#   AUGSETS  how many augmentation token sets to attach (0 = none; needs the account's f3dr-aug dataset)
# Needs the account's private datasets: f3dr-cache (token caches) and f3dr-code (atlas.py, geom.py, score2.py,
# run_exps.py, pl_t_s0_ema.pt). Refresh f3dr-code with `kaggle datasets version` after changing atlas.py.
set -euo pipefail
ENVF=/mnt/c/tinhanh/project/external/zaloai/.env
case "$1" in
  one) USER_NAME=anhanhanhanh; unset KAGGLE_API_TOKEN ;;
  two) USER_NAME=anhnguynnhtinh; export KAGGLE_API_TOKEN=$(grep '^KAGGLE_TWO_TOKEN=' $ENVF | cut -d= -f2-) ;;
  three) USER_NAME=eiyuumiru; export KAGGLE_API_TOKEN=$(grep '^KAGGLE_THREE_TOKEN=' $ENVF | cut -d= -f2-) ;;
  *) echo "account must be one, two or three"; exit 1 ;;
esac
SLUG=$2; AUG=$3; shift 3
D=$(mktemp -d)
ARGS=""
while [ $# -gt 0 ]; do ARGS="$ARGS, \"$1\", \"$2\""; shift 2; done
cat > $D/run.py <<EOF
import subprocess, sys
from pathlib import Path
inp = Path("/kaggle/input")
data = next(p.parent for p in inp.rglob("train_dinov3l.npz"))
code = next(p.parent for p in inp.rglob("run_exps.py"))
dirs = [str(data)]
if $AUG:
    dirs.append(str(next(p.parent for p in inp.rglob("train_dinov3l_aug0.npz"))))
subprocess.run([sys.executable, str(code / "run_exps.py"), "/kaggle/working", ":".join(dirs), str(code)$ARGS], check=True)
EOF
SRC="\"$USER_NAME/f3dr-cache\", \"$USER_NAME/f3dr-code\""
[ "$AUG" != "0" ] && SRC="$SRC, \"$USER_NAME/f3dr-aug\""
cat > $D/kernel-metadata.json <<EOF
{"id": "$USER_NAME/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [$SRC], "competition_sources": [], "kernel_sources": []}
EOF
kaggle kernels push -p $D --accelerator NvidiaTeslaT4 2>&1 | tail -1
