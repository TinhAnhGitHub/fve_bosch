# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "transformers", "peft", "pillow", "numpy", "pandas", "scikit-learn", "scipy", "safetensors"]
# ///
"""B4.0 inference: per-fold held-out + val outputs from saved LoRA models, and a val submission.

Run:  uv run experiments/b4_0_infer.py --folds 0 [1 2 3 4] [--submit NAME]
Writes experiments/out_b4_0_f{k}/preds.npz (held-out and val head outputs); with --submit, averages the
val outputs of the given folds and writes a validated preds_volume.zip (cluster table from ALL train).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_0_finetune as ft  # noqa: E402
from b0_b1 import DATA, write_submission  # noqa: E402
from b3 import build_table, cluster_labels  # noqa: E402
from frozen import Experiment  # noqa: E402


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--submit", default=None)
    args = ap.parse_args()
    exp = Experiment()
    paths = lambda d, split: [DATA / split / r.group / r.category / r.sample_id / "rgb_industrial.png" for r in d.itertuples()]
    val_imgs = ft.load_images(paths(exp.val, "val"))
    for f in args.folds:
        d = Path(__file__).resolve().parent / f"out_b4_0_f{f}"
        _, va_idx = exp.splits(0)[f]
        net = ft.Net()
        set_peft_model_state_dict(net.backbone, load_file(d / "adapter_model.safetensors"))
        net.head.load_state_dict(torch.load(d / "head.pt"))
        out_q, _ = ft.predict(net, ft.load_images(paths(exp.train.iloc[va_idx], "train")))
        out_v, _ = ft.predict(net, val_imgs)
        np.savez(d / "preds.npz", va_idx=va_idx, out_heldout=out_q, out_val=out_v)
        print(f"fold {f}: saved held-out {out_q.shape} and val {out_v.shape} outputs", flush=True)
        del net
        torch.cuda.empty_cache()
    if args.submit:
        outs = np.mean([np.load(Path(__file__).resolve().parent / f"out_b4_0_f{f}" / "preds.npz")["out_val"]
                        for f in args.folds], axis=0)
        table, _ = build_table(exp.train, cluster_labels(exp.train))
        o = ft.build_o(table, None, exp.val, np.clip(outs[:, 0], -2.5, 2.5), sigmoid(outs[:, 1]))
        pred = np.exp(o["ln"])
        print(write_submission(args.submit, exp.val, pred))
        print(f"min={pred.min():.2f} median={np.median(pred):.2f} max={pred.max():.2f}")
        b34 = np.loadtxt(Path(__file__).resolve().parent.parent / "submissions" / "b3_4_knn" / "preds_volume.csv",
                         delimiter=",", skiprows=1, usecols=3)
        d_ = np.abs(np.log(pred / b34))
        print(f"vs B3.4: median |ln ratio| {np.median(d_):.3f}, cluster-size disagreements (>30x) {(d_ > np.log(30)).sum()}")


if __name__ == "__main__":
    main()
