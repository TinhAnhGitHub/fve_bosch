#!/bin/bash
# Fixed-mesh data: unpack, rebuild clouds, train the sheet model on all objects, score, predict.
# Each finished stage is skipped on relaunch.
ROOT=/mnt/c/tinhanh/project/ai_challenge/fve_f3dr
A=$ROOT/baselines/out/atlas
Q=$A/queue
ZIP=/mnt/c/Users/una8hc/Downloads/f2f5f8ed-43f0-40f4-ac03-e236b0486251.zip
NEW=/mnt/c/tinhanh/project/ai_challenge/data/f3d
L=$Q/fixed.log
say() { echo "$(date +%H:%M) $*" >> $L; }
stop() { say "FAILED $*"; exit 1; }
export HF_HUB_OFFLINE=1
cd $ROOT

if [ ! -f $Q/extract.ok ]; then
  python3 -I -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$NEW" > $Q/extract.log 2>&1 || stop extract
  say "unpacked train $(find $NEW/train -name rgb_industrial.png | wc -l) eval $(find $NEW/eval -name rgb_industrial.png | wc -l)"
  touch $Q/extract.ok
fi

if [ ! -f $Q/cache.ok ]; then
  mkdir -p $A/before_fixed_meshes
  # cache reuses the stored clouds when ids match, so the old file has to move out of the way
  [ -f $A/train.npz ] && mv $A/train.npz $A/before_fixed_meshes/
  [ -f $A/train_dense.npz ] && mv $A/train_dense.npz $A/before_fixed_meshes/
  HF_HUB_OFFLINE=0 uv run -q baselines/atlas.py cache > $Q/cache.log 2>&1 || stop cache
  uv run -q --no-project --with numpy python -I $Q/radii.py $A/train.npz $A/before_fixed_meshes/train.npz $ROOT/baselines/out/scale/scale.npz > $Q/radii.log 2>&1
  say "CACHE DONE"
  touch $Q/cache.ok
fi

if [ ! -f $Q/train.ok ]; then
  uv run -q baselines/atlas.py train --arch sheet --target mm --encoder dinov3l --no-flip-loss --ckpt baselines/out/atlas/fx_sheet.pt > $Q/fx_sheet.train.log 2>&1 || stop train
  say "$(grep -E '^saved' $Q/fx_sheet.train.log)"
  touch $Q/train.ok
fi

if [ ! -f $Q/score.ok ]; then
  uv run -q --script $Q/score.py $ROOT q3_sheet fx_sheet > $Q/fx_sheet.score.log 2>&1 || stop score
  grep -E "strict normal|^RESULT" $Q/fx_sheet.score.log >> $L
  touch $Q/score.ok
fi

if [ ! -f $Q/predict.ok ]; then
  uv run -q baselines/atlas.py predict --ckpt baselines/out/atlas/fx_sheet.pt > $Q/fx_sheet.predict.log 2>&1 || stop predict
  touch $Q/predict.ok
fi
say "PIPELINE DONE"
