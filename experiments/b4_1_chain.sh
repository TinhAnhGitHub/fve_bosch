#!/bin/bash
# B4.1 local: DINOv3-L/16 + conv multi-layer head + mesh aux targets + EMA + regularisation, bf16, folds given as args
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a; export HF_TOKEN="${HUGGINGFACE_ACCESS_TOKEN:-$HUGGINGFACE_API_TOKEN}" HF_HUB_OFFLINE=1
for f in "$@"; do
  uv run -q --with torch --with torchvision --with transformers --with peft --with timm --with einops --with pillow --with numpy --with pandas --with safetensors \
    kaggle_pkg/b4_2_train.py --fold $f --epochs 20 --batch 8 --backbone dinov3 --dtype bf16 --drop-path 0.1 --label-smooth 0.1 --scale-aug 0.15 \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m --data data/fve --prep kaggle_pkg/b42_prep --out experiments/out_b4_1 2>&1 \
    | grep --line-buffered -v "Warning\|Loading weights" > experiments/b4_1_f$f.out
done
