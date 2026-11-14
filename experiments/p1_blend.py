# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""P1: B3.4 (frozen DINOv2-L + DINOv3-H+ CLS, ridge + within-category kNN) refit on the official labels, blended with B7.
Held-out = seed-0 folds 0,1 (same rows as B7's held-out). Blend in log space: w * B3.4 + (1 - w) * B7.
usage: p1_blend.py <B7 dir with f*/preds.npz> [k w_ridge w_clf] [w_blend -> also write submission p1_b34_b7_w<w>]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b0_b1 import write_submission
import os
import b3_4_knn as K
from b3_4_knn import collect, decode, knn_parts
from frozen import Config, Experiment

# P1_FEATS=comma list of feature stems (CLS blocks) overrides the B3.4 default pair; NAME tags the submission
if os.environ.get("P1_FEATS"):
    K.CFG = Config(name="p1", features=tuple((f, K.CLS) for f in os.environ["P1_FEATS"].split(",")), n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
CFG = K.CFG
R = ROOT / "kaggle_pkg/b42_prep"; src = Path(sys.argv[1])
k, w_r, w_p = (int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (2, 0.25, 0.25)
sm = lambda y, ln: 2 * np.tanh(np.abs(ln - np.log(y)) / 2) * 100


def b7(o, b0, b1):
    large = (1 / (1 + np.exp(-o[:, 1])) > .5) & np.isfinite(b1)
    return np.where(large, b1, b0) + np.clip(o[:, 0], -2.5, 2.5)


exp = Experiment(); o = collect(exp, 0); b34_all = decode(o, k, w_r, w_p)
idx, l7 = [], []
for f in (0, 1):
    d = np.load(R / f"fold{f}.npz"); idx.append(d["va_idx"])
    l7.append(b7(np.load(src / f"f{f}/preds.npz")["out_heldout"], d["q_base0"], d["q_base1"]))
idx, l7 = np.concatenate(idx), np.concatenate(l7); y, l34 = exp.y[idx], b34_all[idx]
e7 = sm(y, l7)
print(f"held-out (818 rows): B7 {e7.mean():.2f} | B3.4 (k={k}, w_ridge={w_r}, w_clf={w_p}) {sm(y, l34).mean():.2f} | "
      f"corr of log errors {np.corrcoef(np.log(y) - l7, np.log(y) - l34)[0, 1]:.2f}")
for w in (0.3, 0.5, 0.6, 0.7, 0.8, 1.0):
    dd = sm(y, w * l34 + (1 - w) * l7) - e7
    print(f"  w={w}: {e7.mean() + dd.mean():.2f}  (vs B7 {dd.mean():+.2f} +- {dd.std() / np.sqrt(len(dd)):.2f})")

if len(sys.argv) > 5:
    w = float(sys.argv[5])
    X, X_val = exp.features(CFG); Z_tr, Z_va = exp._project(X, X_val, CFG.n_pca)
    Q = exp.val.assign(ln=np.nan); ov = exp.fit_predict(exp.train, Q, Z_tr, Z_va, CFG)
    for kk, parts in knn_parts(exp.train, Q, Z_tr, Z_va).items():
        for name, v in parts.items(): ov[f"{name}_{kk}"] = v
    p = np.load(R / "fold0.npz")
    v7 = b7(np.mean([np.load(src / f"f{f}/preds.npz")["out_val"] for f in (0, 1)], axis=0), p["val_base0"], p["val_base1"])
    pred = np.exp(w * decode(ov, k, w_r, w_p) + (1 - w) * v7)
    print(write_submission(os.environ.get("NAME", f"p1_b34_b7_w{w}"), exp.val, pred))
