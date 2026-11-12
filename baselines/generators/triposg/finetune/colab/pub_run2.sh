#!/usr/bin/env bash
# Public test, second attempt: the 340 photos that the first (lost) VM had not saved. The first run's shards 0 and 1 (rows[0::6] and
# rows[1::6] of the sorted photo list) are safe on the laptop, so here the same sorted list is cut into 12 shards (rows[s::12]):
# the first run's shard 0 is residues {0, 6} and its shard 1 is {1, 7}, so shards 2 3 4 5 8 9 10 11 are what is left.
# Worker 0 takes 2 4 8 10, worker 1 takes 3 5 9 11 (about 43 photos each, about 40 minutes per shard); every finished shard is
# /content/pub2/lora_eval_n12_sK.npz and can be downloaded at once. Log /content/pub2.log ends with PUB2 DONE or PUB2 FAILED.
set -uo pipefail
PY=/content/venv/bin/python
FT=/content/f3dr/baselines/generators/triposg/finetune
export HF_HOME=/content/hf
OUT=/content/pub2
mkdir -p $OUT
step() { echo; echo "=== $(date -u +%H:%M:%S) $* ==="; }

step weights
cd $FT && $PY -c "
import torch
from huggingface_hub import snapshot_download as s
m = torch.load('/content/size_best.pt', map_location='cpu', weights_only=False)['metadata']
print(s(m['model'], revision=m['revision']))" 2>&1 | tail -1 || { echo "PUB2 FAILED (weights)"; exit 1; }

worker() {
  for s in "$@"; do
    [ -f $OUT/lora_eval_n12_s$s.npz ] && continue
    cd $FT && $PY predict.py --upstream /content/TripoSG --data /content/f3d --size /content/size_best.pt \
        --shape /content/lora_best.pt --split eval --steps 50 --guidance 7 --precision fp16 \
        --dense-depth 7 --hierarchical-depth 8 --shard $s --num-shards 12 --output $OUT/lora_eval_n12_s$s.npz \
        > $OUT/lora_n12_s$s.log 2>&1 || return 1
  done
}
step predict
worker 2 4 8 10 &
p0=$!
sleep 60
worker 3 5 9 11 &
p1=$!
wait $p0 || echo "worker 0 stopped early"
wait $p1 || echo "worker 1 stopped early"

step check
n=0
for s in 2 3 4 5 8 9 10 11; do [ -f $OUT/lora_eval_n12_s$s.npz ] && n=$((n + 1)) || { echo "shard $s missing"; tail -3 $OUT/lora_n12_s$s.log 2>/dev/null; }; done
[ "$n" = 8 ] && echo "PUB2 DONE" || { echo "PUB2 FAILED ($n of 8 shards)"; exit 1; }
