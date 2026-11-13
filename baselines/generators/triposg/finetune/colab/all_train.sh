#!/usr/bin/env bash
# TripoSG LoRA on all 2,042 training objects (no holdout), on a Colab A100. One phase per call:
#   all_train.sh setup            venv, TripoSG code, diso, contract tests          (reuses pub_setup.sh; needs /content/code.tgz)
#   all_train.sh vaecheck         30 objects encoded at 32,768 and at 204,800 surface points, VAE round trip scored at true size
#   all_train.sh encode POINTS    all 2,042 objects at POINTS surface points (the vaecheck winner)
#   all_train.sh train            configs/lora_a100_all.json: 20-epoch schedule, stops after 5, adapter saved after every epoch
#   all_train.sh predict EPOCH    the 511 public photos with epoch<EPOCH>.pt, two workers, then merge
# DATA must hold train/<group>/<category>/<id>/{rgb_industrial.png,mesh.ply} and eval/ (Drive mount or /content/f3d).
# Every phase logs to /content/all_<phase>.log and ends with "<PHASE> DONE" or "<PHASE> FAILED".
set -uo pipefail
PHASE=$1
DATA=${DATA:-/content/f3d}
PY=/content/venv/bin/python
FT=/content/f3dr/baselines/generators/triposg/finetune
W=/content/all
export HF_HOME=/content/hf
mkdir -p $W
exec > >(tee -a /content/all_$PHASE.log) 2>&1
step() { echo; echo "=== $(date -u +%H:%M:%S) $* ==="; }
fail() { echo "${PHASE^^} FAILED: $*"; exit 1; }

case $PHASE in
setup)
  bash $FT/colab/pub_setup.sh || fail "pub_setup.sh"
  ;;
vaecheck)
  cd $FT
  step manifest
  $PY prepare.py manifest --data $DATA --all-train --output $W/manifest_all.json || fail manifest
  # 30 objects: the hardest holdout shapes plus a seeded random spread; all other rows excluded from encoding
  $PY - <<'EOF' || fail "subset"
import json, random
m = json.load(open("/content/all/manifest_all.json"))
hard = ["omni_watermelon_041", "omni_durian_007", "omni_watermelon_033", "omni_pancake_004", "mf3d_corn_002", "omni_bread_031",
        "mf3d_lasagna_001", "omni_cucumber_005", "mf3d_nachos_004", "omni_fig_022", "omni_banana_018", "omni_pineapple_008",
        "mf3d_steak_001", "mf3d_waffle_003", "omni_egg_018", "mf3d_blueberries_008", "omni_peanut_019"]
ids = [r["id"] for r in m["rows"]]
rest = sorted(set(ids) - set(hard))
random.Random(42).shuffle(rest)
pick = set(hard) | set(rest[: 30 - len(hard)])
for r in m["rows"]:
    r["split"] = "val" if r["id"] in pick else "excluded"
json.dump(m, open("/content/all/manifest_vae30.json", "w"))
print("vae subset", sum(r["split"] == "val" for r in m["rows"]))
EOF
  for n in 32768 204800; do
    step "encode 30 objects at $n points"
    t0=$(date +%s)
    $PY prepare.py encode --upstream /content/TripoSG --data $DATA --manifest $W/manifest_vae30.json --output $W/cache_vae30_$n \
        --mesh-unit-mm 1 --pose pca --tokens 2048 --surface-points $n || fail "encode $n"
    echo "encode $n: $(( $(date +%s) - t0 )) s for 30 objects"
    step "VAE round trip at $n points"
    $PY vae_check.py --upstream /content/TripoSG --cache $W/cache_vae30_$n --data $DATA --count 30 --output $W/vae30_$n.npz \
        || fail "vae_check $n"
  done
  ;;
encode)
  N=${2:?surface points}
  cd $FT
  [ -f $W/manifest_all.json ] || $PY prepare.py manifest --data $DATA --all-train --output $W/manifest_all.json || fail manifest
  step "encode all at $N points"
  $PY prepare.py encode --upstream /content/TripoSG --data $DATA --manifest $W/manifest_all.json --output $W/cache_all_$N \
      --mesh-unit-mm 1 --pose pca --tokens 2048 --surface-points $N || fail encode
  ln -sfn $W/cache_all_$N $W/cache_all
  ;;
train)
  cd $FT
  step "train LoRA, all objects"
  $PY train.py --cache $W/cache_all --config configs/lora_a100_all.json --upstream /content/TripoSG --output $W/lora_all --no-val \
      $( [ -f $W/lora_all/latest.ckpt ] && echo "--resume $W/lora_all/latest.ckpt" ) || fail train
  ls -la $W/lora_all
  ;;
predict)
  E=${2:?epoch}
  cd $FT
  [ -f /content/size_best.pt ] || fail "upload the size head as /content/size_best.pt"
  worker() {
    for s in "$@"; do
      [ -f $W/pub_e$E/s$s.npz ] && continue
      $PY predict.py --upstream /content/TripoSG --data $DATA --size /content/size_best.pt --shape $W/lora_all/epoch$E.pt \
          --split eval --steps 50 --guidance 7 --precision fp16 --dense-depth 7 --hierarchical-depth 8 \
          --shard $s --num-shards 8 --output $W/pub_e$E/s$s.npz > $W/pub_e$E/s$s.log 2>&1 || return 1
    done
  }
  mkdir -p $W/pub_e$E
  step "predict 511 public photos, epoch $E"
  worker 0 2 4 6 & p0=$!
  sleep 60
  worker 1 3 5 7 & p1=$!
  wait $p0 || fail "worker 0"
  wait $p1 || fail "worker 1"
  $PY merge.py $W/pub_e$E/s*.npz --output $W/pub_e$E/lora_all_eval.npz --expect 511 || fail merge
  ;;
*) fail "unknown phase" ;;
esac
echo "${PHASE^^} DONE"
