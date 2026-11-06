# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""OOF check on folds 0,1: does the Kaggle B4.2 (C-RADIO + conv head) add to the B3.4 + B4.0 blend?"""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
import b3_4_knn as knn
import b4_0_blend as bl
from b0_b1 import smape
from frozen import Experiment

GAP = math.log(30.0)
exp = Experiment(); knn.KS = (2,)
ln34_all = knn.decode(knn.collect(exp, 0), 2, .25, .25)
y_all = exp.train["volume"].to_numpy()
I, L34, L40, L42 = [], [], [], []
for f in (0, 1):
    va, ln40 = bl.b4_ln(exp, f)
    d = np.load(ROOT / "kaggle_out/b42" / f"out/f{f}/preds.npz"); p = np.load(ROOT / "kaggle_pkg/b42_prep" / f"fold{f}.npz")
    assert (d["va_idx"] == va).all()
    o = d["out_heldout"]; r = np.clip(o[:, 0], -2.5, 2.5)
    large = (1 / (1 + np.exp(-o[:, 1])) > .5) & np.isfinite(p["q_base1"])
    ln42 = np.where(large, p["q_base1"], p["q_base0"]) + r
    I.append(va); L34.append(ln34_all[va]); L40.append(ln40); L42.append(ln42)
    print(f"fold {f}: B3.4 {smape(y_all[va], np.exp(ln34_all[va])):.2f} | B4.0 {smape(y_all[va], np.exp(ln40)):.2f} | B4.2 {smape(y_all[va], np.exp(ln42)):.2f}")
I, a, b, c = np.concatenate(I), np.concatenate(L34), np.concatenate(L40), np.concatenate(L42); y = y_all[I]
S = lambda ln: smape(y, np.exp(ln))
def mix(parts, ws):
    base = parts[0]
    ok = np.all([np.abs(p - base) < GAP for p in parts], axis=0)
    return np.where(ok, sum(w * p for w, p in zip(ws, parts)), base)
print(f"pooled n={len(I)}: B3.4 {S(a):.2f} | B4.0 {S(b):.2f} | B4.2 {S(c):.2f}")
print(f"B3.4+B4.0 (0.5/0.5): {S(mix([a, b], [.5, .5])):.2f}")
print(f"B3.4+B4.2 (0.5/0.5): {S(mix([a, c], [.5, .5])):.2f}")
for w in ((.5, .25, .25), (.4, .3, .3), (.5, .5, 0), (.5, .3, .2), (.34, .33, .33)):
    print(f"B3.4/B4.0/B4.2 {w}: {S(mix([a, b, c], list(w))):.2f}")
print("corr of OOF ln errors  B4.0 vs B4.2:", np.corrcoef(b - np.log(y), c - np.log(y))[0, 1].round(3),
      "| B3.4 vs B4.2:", np.corrcoef(a - np.log(y), c - np.log(y))[0, 1].round(3), "| B3.4 vs B4.0:", np.corrcoef(a - np.log(y), b - np.log(y))[0, 1].round(3))
