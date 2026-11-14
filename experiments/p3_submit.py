# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""P3 submission: B7 (mean of fold heads, no shrink) + ridge correction of B7's log error from TripoSR shape descriptors,
fitted on the 818 held-out rows of folds 0,1 (held-out test: +0.36 worse, i.e. expected to hurt; submitted on request).
usage: p3_submit.py <desc.csv> <B7 dir with f*/preds.npz> [name=p3_b7_shape]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "kaggle_pkg/b42_prep"
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
from b0_b1 import write_submission
from frozen import Experiment

FEATS = ["ln_e1", "a2", "a3", "fill", "hull", "spher", "watertight", "n_comp"]
src, NAME = Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "p3_b7_shape"
rows = pd.read_csv(R / "rows.csv"); X = rows[["sample_id"]].merge(pd.read_csv(sys.argv[1]), how="left")[FEATS]
X = X.fillna(X.median()).to_numpy(); tr, va = (rows.split == "train").to_numpy(), (rows.split == "val").to_numpy()
exp = Experiment(); assert list(rows[va].sample_id) == list(exp.val["sample_id"]), "val order mismatch"


def b7(o, b0, b1):
    large = (1 / (1 + np.exp(-o[:, 1])) > .5) & np.isfinite(b1)
    return np.where(large, b1, b0) + np.clip(o[:, 0], -2.5, 2.5)


Xq, err = [], []
for f in (0, 1):
    d = np.load(R / f"fold{f}.npz"); o = np.load(src / f"f{f}/preds.npz")["out_heldout"]
    Xq.append(X[tr][d["va_idx"]]); err.append(np.log(d["q_y"]) - b7(o, d["q_base0"], d["q_base1"]))
fix = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(np.concatenate(Xq), np.concatenate(err))
p = np.load(R / "fold0.npz")
ln = b7(np.mean([np.load(src / f"f{f}/preds.npz")["out_val"] for f in (0, 1)], axis=0), p["val_base0"], p["val_base1"])
corr = fix.predict(X[va])
pred = np.exp(ln + corr)
print(write_submission(NAME, exp.val, pred))
print(f"n={len(pred)} correction mean {corr.mean():+.3f} sd {corr.std():.3f} (alpha {fix[-1].alpha_:.1f}); "
      f"median {np.median(pred):.2f}")
