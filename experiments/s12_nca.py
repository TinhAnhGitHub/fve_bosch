# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "torch"]
# ///
"""S1 / S2: single retrieval models on the frozen CLS features of the tuned B3.4 set (DINOv2-L, DINOv3-H+, DINOv3-7B,
SigLIP2), replacing B3.4's fixed cosine k=2 kNN + ridge.
  S1 (ModernNCA, Ye et al. ICLR'25): learned projection e = W x (input dropout); soft neighbours among the SAME-CATEGORY
     training rows, w_ij = softmax_j(-|e_i - e_j|^2 * t), t learned; trained leave-one-out on the training part with
     random neighbour sampling (50%) by the expected metric  sum_j w_ij * tanh(|ln y_i - ln y_j| / 2).
  S2 (+ TabR value module, Gorishniy et al. ICLR'24): neighbour j proposes ln y_j + delta(e_i - e_j) (MLP, zero-init),
     i.e. "this object looks bigger/smaller than neighbour j"; same loss on the proposals.
Prediction: sMAPE-optimal point of the weighted proposals ("med") or their weighted mean ("mean").
Fallback (no training row of the category): global sMAPE-optimal value (as B3).  Inputs: standardise + PCA(256) on the
training part.  Protocol as B3.4: pick (decode, steps) on seeds 0-2, report fresh seeds 3-5; then fit on all train.
usage: s12_nca.py [s1|s2|both] [--submit]"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b0_b1 import smape, write_submission
from b3 import argmin_t
from frozen import Config, Experiment

FEATS = ("dinov2l_reg_518", "dinov3_vith16plus_512", "dinov3_vit7b_512", "siglip2_so400m_512")
D, E, DROP, LR, WD, SAMPLE = 256, 64, 0.2, 3e-3, 1e-4, 0.5
CKPT = (100, 200, 300, 500, 800)
torch.set_num_threads(8)


def pairs(ca, cb, exclude_self):
    """(i, j) with ca[i] == cb[j] (i != j when both sides are the same rows)."""
    m = ca[:, None] == cb[None]
    if exclude_self: np.fill_diagonal(m, False)
    i, j = np.nonzero(m)
    return torch.from_numpy(i), torch.from_numpy(j)


def seg_softmax(x, i, n):
    mx = torch.full((n,), -1e30).scatter_reduce(0, i, x, "amax")
    ex = (x - mx[i]).exp()
    return ex / torch.zeros(n).index_add(0, i, ex)[i]


def decode(v, w, i, n, fallback):
    """per query: sMAPE-optimal proposal ('med') and weighted mean ('mean')."""
    med, mean = np.full(n, fallback), np.full(n, fallback)
    v, w, i = v.numpy(), w.numpy(), i.numpy()
    for q in np.unique(i):
        s = i == q; vq, wq = v[s], w[s]
        med[q] = vq[np.argmin(wq @ np.tanh(np.abs(vq[:, None] - vq[None]) / 2))]
        mean[q] = wq @ vq
    return med, mean


def fit_predict(Xt, lnt, ct, Xq, cq, value, seed):
    """train on (Xt, lnt, ct); return {(decode, step): ln predictions for Xq}."""
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    pca = PCA(D, random_state=0).fit((Xt - mu) / sd)
    A = torch.tensor(pca.transform((Xt - mu) / sd), dtype=torch.float32); s = A.std(); A = A / s
    B = torch.tensor(pca.transform((Xq - mu) / sd), dtype=torch.float32) / s
    y = torch.tensor(lnt, dtype=torch.float32); n = len(A)
    ti, tj = pairs(ct, ct, True); qi, qj = pairs(cq, ct, False)
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(DROP), nn.Linear(D, E))
    val = nn.Sequential(nn.Linear(E, 64), nn.GELU(), nn.Linear(64, 1)) if value else None
    if value: nn.init.zeros_(val[2].weight); nn.init.zeros_(val[2].bias)
    with torch.no_grad():
        e0 = enc[1](A); log_t = nn.Parameter(-torch.log(((e0[ti] - e0[tj]) ** 2).sum(1).median()) + np.log(5.0))
    params = [*enc.parameters(), log_t] + ([*val.parameters()] if value else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    fb = argmin_t(lnt); out = {}

    def propose(ea, eb, i, j):
        d = ea[i] - eb[j]
        v = y[j] + (val(d).squeeze(1).clamp(-2.5, 2.5) if value else 0)
        return v, -(d ** 2).sum(1) * log_t.exp()

    for step in range(1, max(CKPT) + 1):
        enc.train()
        keep = torch.rand(len(ti)) < SAMPLE
        e = enc(A); i, j = ti[keep], tj[keep]
        v, lg = propose(e, e, i, j)
        w = seg_softmax(lg, i, n)
        loss = torch.zeros(n).index_add(0, i, w * torch.tanh((y[i] - v).abs() / 2)).sum() / len(torch.unique(i))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in CKPT:
            enc.eval()
            with torch.no_grad():
                v, lg = propose(enc(B), enc(A), qi, qj)
                med, mean = decode(v, seg_softmax(lg, qi, len(B)), qi, len(B), fb)
            out[("med", step)], out[("mean", step)] = med, mean
    return out


def oof(exp, X, value, seed):
    res = {}
    for k, (tr, va) in enumerate(exp.splits(seed)):
        o = fit_predict(X[tr], exp.ln[tr], exp.cats[tr], X[va], exp.cats[va], value, seed * 10 + k)
        for key, v in o.items(): res.setdefault(key, np.full(len(X), np.nan))[va] = v
    return {key: smape(exp.y, np.exp(v)) for key, v in res.items()}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    exp = Experiment()
    # *_cls.npz = the CLS blocks (0, 2, 4) of each feature file only (half the size); all 6 "blocks" = those columns
    X, Xv = exp.features(Config(name="s12", features=tuple((f + "_cls", tuple(range(6))) for f in FEATS), n_pca=None))
    for name, value in (("S1", False), ("S2", True)):
        if which not in ("both", name.lower()): continue
        t0 = time.time()
        sel = [oof(exp, X, value, s) for s in (0, 1, 2)]
        grid = {key: np.mean([r[key] for r in sel]) for key in sel[0]}
        print(f"{name} seeds 0-2: " + " ".join(f"{d}@{s}={v:.2f}" for (d, s), v in sorted(grid.items())), flush=True)
        best = min(grid, key=grid.get)
        fresh = np.mean([oof(exp, X, value, s)[best] for s in (3, 4, 5)])
        print(f"{name} pick {best}: seeds0-2 {grid[best]:.2f} | fresh3-5 {fresh:.2f}  ({time.time() - t0:.0f}s)  "
              f"[tuned B3.4: 34.44 | 34.37]", flush=True)
        if "--submit" in sys.argv:
            ln = fit_predict(X, exp.ln, exp.cats, Xv, exp.val.category.to_numpy(), value, 0)[best]
            pred = np.exp(ln)
            print(write_submission(f"{name.lower()}_nca{'_value' if value else ''}", exp.val, pred),
                  f"min {pred.min():.2f} median {np.median(pred):.2f} max {pred.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
