# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Out-of-fold blend of fine-tuned B4.0 heads with B3.4 on the folds that have saved held-out outputs.

Both models are scored on the SAME held-out rows (seed-0 splits). B4.0's ln prediction is rebuilt from its saved
head outputs and the fold's training-only cluster table (as in b4_0_check.py); B3.4's comes from knn.collect(seed 0).
Blend: where the two agree on the size cluster (|diff| < ln 30) average in ln space with weight w on B4.0;
where they disagree on the cluster, keep B3.4's call (never hedge between modes).

Run:  uv run --offline experiments/b4_0_blend.py [--folds 0 1]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
import b3_4_knn as knn  # noqa: E402
from b0_b1 import smape  # noqa: E402
from b3 import build_table, cluster_labels  # noqa: E402
from frozen import Experiment  # noqa: E402

GAP = math.log(30.0)


def b4_ln(exp: Experiment, fold: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(Path(__file__).resolve().parent / f"out_b4_0_f{fold}" / "preds.npz")
    tr_idx, va_idx = exp.splits(0)[fold]
    assert (d["va_idx"] == va_idx).all(), "saved fold rows do not match the seed-0 split"
    T, Q = exp.train.iloc[tr_idx].reset_index(drop=True), exp.train.iloc[va_idx].reset_index(drop=True)
    table, _ = build_table(T, cluster_labels(T))
    out = d["out_heldout"]
    r, p = np.clip(out[:, 0], -2.5, 2.5), 1 / (1 + np.exp(-out[:, 1]))
    ln = np.empty(len(Q))
    for i, c in enumerate(Q["category"]):
        v = table.values.get(c)
        ln[i] = table.fallback if v is None else (next(iter(v.values())) if len(v) == 1 else (v[1] if p[i] > .5 else v[0])) + r[i]
    return va_idx, ln


def blend(ln4: np.ndarray, ln34: np.ndarray, w: float) -> np.ndarray:
    agree = np.abs(ln4 - ln34) < GAP
    return np.where(agree, w * ln4 + (1 - w) * ln34, ln34)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    exp = Experiment()
    knn.KS = (2,)
    o = knn.collect(exp, 0)
    ln34_all = knn.decode(o, 2, 0.25, 0.25)
    y_all, cats = exp.train["volume"].to_numpy(), exp.train["category"].to_numpy()

    idx, l4, l34 = [], [], []
    for f in args.folds:
        va_idx, ln4 = b4_ln(exp, f)
        idx.append(va_idx); l4.append(ln4); l34.append(ln34_all[va_idx])
        print(f"fold {f}: n={len(va_idx)}  B4.0 {smape(y_all[va_idx], np.exp(ln4)):.2f} | "
              f"B3.4 {smape(y_all[va_idx], np.exp(ln34_all[va_idx])):.2f}", flush=True)
    idx, l4, l34 = np.concatenate(idx), np.concatenate(l4), np.concatenate(l34)
    y, c = y_all[idx], cats[idx]
    print(f"\npooled over folds {args.folds} (n={len(idx)}); mode disagreements (>30x): "
          f"{int((np.abs(l4 - l34) >= GAP).sum())}")
    base = smape(y, np.exp(l34))
    print(f"{'w(B4.0)':>8} {'sMAPE':>7} {'vs B3.4':>8}")
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        s = smape(y, np.exp(blend(l4, l34, w)))
        print(f"{w:8.2f} {s:7.2f} {s - base:+8.2f}")

    # paired category-clustered bootstrap of the per-object gain for w=0.5
    per = lambda ln: 2 * np.tanh(np.abs(ln - np.log(y)) / 2) * 100
    gain = per(l34) - per(blend(l4, l34, 0.5))
    rng = np.random.default_rng(0)
    groups = {k: np.flatnonzero(c == k) for k in np.unique(c)}
    keys = list(groups)
    boots = [gain[np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nw=0.5 gain over B3.4: {gain.mean():+.2f} points, 95% category-bootstrap CI [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
