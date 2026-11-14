# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""S3 prep: retrieval tokens for b4_2_train.py --retr.  For every row of fold f (its training rows, held-out rows, val
rows) the K most similar SAME-CATEGORY rows of the fold's training part (cosine on standardised frozen CLS features of
the tuned B3.4 set; a training row never retrieves itself).  Token = [ln_j - category base, oversized u_j, cosine,
valid, PCA_P(z_j) - PCA_P(z_i)]  (scaler/PCA/bases fitted on the fold's training part only).
Writes kaggle_pkg/b42_prep/retr{f}.npz with nb_t [n_tr,K,F], nb_q [n_heldout,K,F], nb_v [511,K,F].
Check printed per fold: held-out sMAPE of base + mean label of the top-2 tokens (should be ~kNN-alone, 36-37)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "kaggle_pkg/b42_prep"
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b3 import build_table, cluster_labels
from frozen import Config, Experiment

K, P = 8, 32
FEATS = ("dinov2l_reg_518", "dinov3_vith16plus_512", "dinov3_vit7b_512", "siglip2_so400m_512")
exp = Experiment(); rows = pd.read_csv(R / "rows.csv")
assert list(rows[rows.split == "train"].sample_id) == list(exp.train.sample_id), "train order mismatch"
assert list(rows[rows.split == "val"].sample_id) == list(exp.val.sample_id), "val order mismatch"
X, Xv = exp.features(Config(name="s3", features=tuple((f, (0, 2, 4)) for f in FEATS), n_pca=None))
unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
all_cats, val_cats = exp.train.category.to_numpy(), exp.val.category.to_numpy()

for f in range(5):
    d = np.load(R / f"fold{f}.npz"); tr, va = d["tr_idx"], d["va_idx"]
    T = exp.train.iloc[tr]; cats = all_cats[tr]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Zt = (X[tr] - mu) / sd
    pca = PCA(P, whiten=True, random_state=0).fit(Zt)
    Ut, Pt = unit(Zt), pca.transform(Zt)
    u = cluster_labels(T); table, _ = build_table(T, u)
    lab = T.ln.to_numpy() - np.array([table.values[c].get(0, table.values[c].get(1)) for c in cats])
    by_cat = {c: np.flatnonzero(cats == c) for c in np.unique(cats)}

    def tokens(A, qcats, self_idx=None):
        Za = (A - mu) / sd; S = unit(Za) @ Ut.T; Pa = pca.transform(Za)
        out = np.zeros((len(A), K, 4 + P), np.float32)
        for i, c in enumerate(qcats):
            idx = by_cat.get(c, np.array([], int))
            if self_idx is not None: idx = idx[idx != self_idx[i]]
            nb = idx[np.argsort(-S[i, idx])][:K]
            out[i, :len(nb)] = np.c_[lab[nb], u[nb], S[i, nb], np.ones(len(nb)), Pt[nb] - Pa[i]]
        return out

    nb_t, nb_q, nb_v = tokens(X[tr], cats, np.arange(len(tr))), tokens(X[va], all_cats[va]), tokens(Xv, val_cats)
    np.savez(R / f"retr{f}.npz", nb_t=nb_t, nb_q=nb_q, nb_v=nb_v)
    b = np.where(np.isfinite(d["q_base0"]), d["q_base0"], d["q_base1"])
    w = nb_q[:, :2, 3]; knn = (nb_q[:, :2, 0] * w).sum(1) / np.maximum(w.sum(1), 1)
    sm = 200 * np.tanh(np.abs(b + knn - np.log(d["q_y"])) / 2).mean()
    print(f"fold {f}: tokens {nb_t.shape} {nb_q.shape} {nb_v.shape}; empty held-out {int((nb_q[:, 0, 3] == 0).sum())}; "
          f"top-2 token check sMAPE {sm:.2f}", flush=True)
