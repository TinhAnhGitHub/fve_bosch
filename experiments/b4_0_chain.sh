#!/bin/bash
# usage: b4_0_chain.sh fold...   (train + infer each fold in order)
cd "$(dirname "$0")/.."
F='Loading weights|Warning|warn'
for f in "$@"; do
  uv run --offline -q experiments/b4_0_finetune.py --epochs 10 --batch 8 --fold $f 2>&1 | grep --line-buffered -vE "$F" > experiments/b4_0_f$f.out
  uv run --offline -q experiments/b4_0_infer.py --folds $f 2>&1 | grep --line-buffered -vE "$F" >> experiments/b4_0_f$f.out
done
