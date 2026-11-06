# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B5.1: second-stage calibration of P(oversized) on out-of-fold stage-1 outputs.

Stage-1 (B3.4) OOF gives p_clf, p_knn(k=2,5). Stage-2 logistic regression on logit(p_clf), logit(p_knn2), logit(p_knn5),
logit(category oversized rate, leave-one-out) predicts the true mode; it is itself evaluated by 5-fold CV over the OOF rows
(grouped so each row is scored by a model that never saw it). Selection on seeds 0-2, check on seeds 3-5.
Compare against B3.4 decode (k=2, w_r=.25, w_p=.25, tau .5) and tau sweeps.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
import b3_4_knn as knn
from b0_b1 import smape
from b3 import cluster_labels
from frozen import Experiment

knn.KS = (2, 5)
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))


def decode_p(o, p, k=2, w_r=.25, tau=.5):
    mixed = np.isfinite(o["p_large"])
    large = np.where(mixed, p > tau, ~np.isfinite(o["ref_n"]) & np.isfinite(o["ref_l"]))
    base = np.where(mixed, np.where(large, o["ref_l"], o["ref_n"]), o["ln"] - o["r_hat"])
    r_knn = np.where(large, o[f"r1_{k}"], o[f"r0_{k}"])
    r = np.where(np.isfinite(r_knn), w_r * o["r_hat"] + (1 - w_r) * r_knn, o["r_hat"])
    return base + r


def stage2(exp, o, seed, feats):
    u = cluster_labels(exp.train).astype(float); cats = exp.train["category"].to_numpy()
    mixed = np.isfinite(o["p_large"])
    prior = np.full(len(u), np.nan)
    for c in np.unique(cats):
        i = np.flatnonzero(cats == c); n = len(i)
        if n > 1: prior[i] = (u[i].sum() - u[i]) / (n - 1)
    cols = {"clf": lg(o["p_large"]), "k2": lg(np.where(np.isfinite(o["p_2"]), o["p_2"], .5)),
            "k5": lg(np.where(np.isfinite(o["p_5"]), o["p_5"], .5)), "prior": lg(np.where(np.isfinite(prior), prior, .5))}
    X = np.column_stack([cols[f] for f in feats])
    p = np.where(mixed, np.nan, np.nan)
    idx = np.flatnonzero(mixed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(idx, u[idx]):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[idx[tr]], u[idx[tr]])
        p[idx[te]] = m.predict_proba(X[idx[te]])[:, 1]
    return p


def main():
    exp = Experiment(); y = exp.y
    sets = {"clf+k2+k5": ("clf", "k2", "k5"), "+prior": ("clf", "k2", "k5", "prior"), "k2+prior": ("k2", "prior"), "clf+k2": ("clf", "k2")}
    for label, seeds in (("selection seeds 0-2", (0, 1, 2)), ("fresh seeds 3-5", (3, 4, 5))):
        outs = [(s, knn.collect(exp, s)) for s in seeds]
        base = []; res = {k: [] for k in sets}; tau = {t: [] for t in (.4, .5, .6)}
        for s, o in outs:
            pb = np.where(np.isfinite(o["p_2"]), .25 * o["p_large"] + .75 * o["p_2"], o["p_large"])
            base.append(smape(y, np.exp(decode_p(o, pb))))
            for k, f in sets.items():
                p2 = stage2(exp, o, s, f); p2 = np.where(np.isfinite(o["p_large"]), p2, np.nan)
                res[k].append(smape(y, np.exp(decode_p(o, p2))))
                if k == "+prior":
                    for t in tau: tau[t].append(smape(y, np.exp(decode_p(o, p2, tau=t))))
        print(f"\n{label}: B3.4 {np.mean(base):.2f}")
        for k, v in res.items(): print(f"  stage2 {k:10s} {np.mean(v):.2f}  (gain {np.mean(base)-np.mean(v):+.2f})")
        for t, v in tau.items(): print(f"  stage2 +prior tau={t}: {np.mean(v):.2f}")
        sys.stdout.flush()

main()
