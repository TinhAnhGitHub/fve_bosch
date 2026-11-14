# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""P3 gate: do TripoSR shape descriptors add size signal?  usage: p3_eval.py <desc.csv> <B7 dir with f*/preds.npz>
(a) category base + ridge(descriptors) fitted on the fold's training rows, scored on its held-out rows (vs base alone);
(b) B7 held-out + ridge correction of B7's log error from descriptors, 5-fold CV inside the 818 held-out rows (vs B7)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path(__file__).resolve().parent.parent / "kaggle_pkg/b42_prep"
FEATS = ["ln_e1", "a2", "a3", "fill", "hull", "spher", "watertight", "n_comp"]
rows = pd.read_csv(R / "rows.csv"); tr = rows[rows.split == "train"].reset_index(drop=True)
X = tr[["sample_id"]].merge(pd.read_csv(sys.argv[1]), how="left")[FEATS]
print(f"descriptor rows {X.notna().all(1).sum()}/{len(X)}; watertight {X.watertight.mean():.2f}")
X = X.fillna(X.median()).to_numpy()
sm = lambda y, ln: 2 * np.tanh(np.abs(ln - np.log(y)) / 2) * 100
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
ys, b7, xs = [], [], []
for f in (0, 1):
    d = np.load(R / f"fold{f}.npz"); o = np.load(Path(sys.argv[2]) / f"f{f}/preds.npz")["out_heldout"]
    large = (1 / (1 + np.exp(-o[:, 1])) > .5) & np.isfinite(d["q_base1"]); base = np.where(large, d["q_base1"], d["q_base0"])
    ok = d["ok"].astype(bool); m = ridge().fit(X[d["tr_idx"]][ok], d["r"][ok])
    print(f"(a) fold {f}: base {sm(d['q_y'], base).mean():.2f} -> base + ridge(desc) "
          f"{sm(d['q_y'], base + np.clip(m.predict(X[d['va_idx']]), -2.5, 2.5)).mean():.2f}")
    ys.append(d["q_y"]); b7.append(base + np.clip(o[:, 0], -2.5, 2.5)); xs.append(X[d["va_idx"]])
y, b7, Xq = map(np.concatenate, (ys, b7, xs)); fix = np.zeros_like(b7)
for a, b in KFold(5, shuffle=True, random_state=0).split(Xq):
    fix[b] = ridge().fit(Xq[a], (np.log(y) - b7)[a]).predict(Xq[b])
e0, e1 = sm(y, b7), sm(y, b7 + fix); dd = e1 - e0
print(f"(b) B7 {e0.mean():.2f} -> B7 + desc correction {e1.mean():.2f}  (diff {dd.mean():+.2f} +- {dd.std() / np.sqrt(len(dd)):.2f})")
