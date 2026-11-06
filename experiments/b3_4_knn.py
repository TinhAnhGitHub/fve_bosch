# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3.4: kNN features on top of the best B3.3 model (DINOv2-L + DINOv3-H+ CLS, no PCA).

For each query image, the k most similar TRAINING images of the same category (cosine on the
standardised features, inside the training fold only) give:
  p_knn  = similarity-weighted oversized rate of the neighbours
  r_knn  = similarity-weighted size correction of the neighbours in the chosen cluster
They are blended with the ridge correction and the classifier probability:
  r = w_r * r_ridge + (1 - w_r) * r_knn        p = w_p * p_clf + (1 - w_p) * p_knn
Weights and k are picked on seeds 0-2, then re-checked on fresh seeds 3-5.

Run:  uv run experiments/b3_4_knn.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from b0_b1 import smape, write_submission  # noqa: E402
from b3 import build_table, cluster_labels  # noqa: E402
from frozen import Config, Experiment  # noqa: E402

CLS = (0, 2, 4)
CFG = Config(name="B3.3 best", features=(("dinov2l_reg_518", CLS), ("dinov3_vith16plus_512", CLS)),
             n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
KS = (1, 2, 3, 4, 5, 10)
W_R = (1.0, 0.75, 0.5, 0.25, 0.0)
W_P = (1.0, 0.75, 0.5, 0.25, 0.0)


def unit(Z: np.ndarray) -> np.ndarray:
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def knn_parts(T: pd.DataFrame, Q: pd.DataFrame, Z_tr: np.ndarray, Z_va: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Per k: neighbour oversized rate and neighbour corrections for each cluster (NaN when unavailable)."""
    u_tr = cluster_labels(T)
    _, ref_loo = build_table(T, u_tr)
    r_tr = T["ln"].to_numpy() - ref_loo
    cats_tr, q_cats = T["category"].to_numpy(), Q["category"].to_numpy()
    S = unit(Z_va) @ unit(Z_tr).T
    out = {k: {n: np.full(len(Q), np.nan) for n in ("p", "r0", "r1")} for k in KS}
    by_cat = {c: np.flatnonzero(cats_tr == c) for c in np.unique(cats_tr)}
    for i, c in enumerate(q_cats):
        idx = by_cat.get(c)
        if idx is None:
            continue
        order = idx[np.argsort(-S[i, idx])]
        for k in KS:
            nb = order[:k]
            w = np.clip(S[i, nb], 1e-3, None)
            out[k]["p"][i] = np.sum(w * u_tr[nb]) / w.sum()
            for mode in (0, 1):
                m = order[(u_tr[order] == mode) & np.isfinite(r_tr[order])][:k]
                if len(m):
                    wm = np.clip(S[i, m], 1e-3, None)
                    out[k][f"r{mode}"][i] = np.sum(wm * r_tr[m]) / wm.sum()
    return out


def collect(exp: Experiment, seed: int) -> dict[str, np.ndarray]:
    """OOF arrays for the B3.3 model plus kNN parts, in train order."""
    n = len(exp.train)
    o = {k: np.full(n, np.nan) for k in ("ln", "r_hat", "p_large", "ref_n", "ref_l")}
    o |= {f"{name}_{k}": np.full(n, np.nan) for k in KS for name in ("p", "r0", "r1")}
    for f in exp.folds(CFG, seed):
        T, Q = exp.train.iloc[f.tr], exp.train.iloc[f.va]
        res = exp.fit_predict(T, Q, f.Z_tr, f.Z_va, CFG)
        for key, v in res.items():
            o[key][f.va] = v
        for k, parts in knn_parts(T, Q, f.Z_tr, f.Z_va).items():
            for name, v in parts.items():
                o[f"{name}_{k}"][f.va] = v
    return o


def decode(o: dict[str, np.ndarray], k: int, w_r: float, w_p: float, tau: float = 0.5) -> np.ndarray:
    """Blend ridge/classifier with kNN and return ln predictions."""
    mixed = np.isfinite(o["p_large"])
    p_knn = o[f"p_{k}"]
    p = np.where(np.isfinite(p_knn), w_p * o["p_large"] + (1 - w_p) * p_knn, o["p_large"])
    large = np.where(mixed, p > tau, ~np.isfinite(o["ref_n"]) & np.isfinite(o["ref_l"]))
    base = np.where(mixed, np.where(large, o["ref_l"], o["ref_n"]), o["ln"] - o["r_hat"])
    r_knn = np.where(large, o[f"r1_{k}"], o[f"r0_{k}"])
    r = np.where(np.isfinite(r_knn), w_r * o["r_hat"] + (1 - w_r) * r_knn, o["r_hat"])
    return base + r


def score_grid(exp: Experiment, outs: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for k, w_r, w_p in itertools.product(KS, W_R, W_P):
        s = [smape(exp.y, np.exp(decode(o, k, w_r, w_p))) for o in outs]
        rows.append({"k": k, "w_ridge": w_r, "w_clf": w_p, "smape": np.mean(s), "std": np.std(s)})
    return pd.DataFrame(rows).sort_values("smape")


def submit(exp: Experiment, k: int, w_r: float, w_p: float, folder: str) -> np.ndarray:
    """Fit on all train (ridge, classifier, kNN pool), predict val, write a validated zip."""
    X, X_val = exp.features(CFG)
    Z_tr, Z_va = exp._project(X, X_val, CFG.n_pca)
    Q = exp.val.assign(ln=np.nan)
    o = exp.fit_predict(exp.train, Q, Z_tr, Z_va, CFG)
    for kk, parts in knn_parts(exp.train, Q, Z_tr, Z_va).items():
        for name, v in parts.items():
            o[f"{name}_{kk}"] = v
    pred = np.exp(decode(o, k, w_r, w_p))
    path = write_submission(folder, exp.val, pred)
    print(f"submission: min={pred.min():.2f} median={np.median(pred):.2f} max={pred.max():.2f} -> {path}")
    return pred


def main() -> None:
    exp = Experiment()
    sel = [collect(exp, s) for s in (0, 1, 2)]
    grid = score_grid(exp, sel)
    base = grid[(grid.w_ridge == 1) & (grid.w_clf == 1)].iloc[0]
    print(f"B3.3 (no kNN) on seeds 0-2: {base.smape:.2f}")
    print("\ntop 10 kNN blends (seeds 0-2):")
    print(grid.head(10).round(3).to_string(index=False))
    print("\nkNN alone (w_ridge=0, w_clf=0) by k:")
    print(grid[(grid.w_ridge == 0) & (grid.w_clf == 0)].sort_values("k").round(2).to_string(index=False))

    best = grid.iloc[0]
    fresh = [collect(exp, s) for s in (3, 4, 5)]
    f_base = np.mean([smape(exp.y, np.exp(decode(o, 5, 1.0, 1.0))) for o in fresh])
    f_best = np.mean([smape(exp.y, np.exp(decode(o, int(best.k), best.w_ridge, best.w_clf))) for o in fresh])
    print(f"\nfresh seeds 3-5: B3.3 {f_base:.2f} | best blend (k={int(best.k)}, w_ridge={best.w_ridge}, "
          f"w_clf={best.w_clf}) {f_best:.2f} | gain {f_base - f_best:.2f}")
    submit(exp, int(best.k), best.w_ridge, best.w_clf, "b3_4_knn")
    exp.log(Config(name=f"B3.4 knn k={int(best.k)} wr={best.w_ridge} wp={best.w_clf}", features=CFG.features,
                   n_pca=None, C=0.01, tau=0.5, r_clip=2.5),
            {"smape": best.smape, "smape_std": best["std"], "smape_fresh": f_best, "smape_fresh_b33": f_base})


if __name__ == "__main__":
    main()
