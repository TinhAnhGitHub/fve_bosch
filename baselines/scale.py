# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "scikit-learn", "torch", "trimesh"]
# ///
"""Size head: how big is the object, in millimetres.

The grader keeps scale, so CD-L2 grows with the square of the size error. The
atlas decoder only draws a unit-size shape. This script predicts the size

  s = RMS radius about the centroid, in mm      (geom.rms_radius)

and atlas.py multiplies the unit shape by it. The method is the FVE volume
pipeline retargeted from log volume to log radius, without its kNN step:

  1. Oversized clusters. About 10% of OmniObject3D meshes are 6-13x too large
     in length. Same rule as the FVE repo: a >= 30x volume gap inside a category
     (ln 30 / 3 in log radius), or a whole category that far above its group.
  2. Category table. Median log radius of each (category, cluster).
  3. Ridge on frozen image features predicts the offset from that median.
  4. Logistic regression gives P(oversized), blended with the category rate.
  5. One number for the metric. Squared Chamfer punishes (s - s_true)^2, so the
     best single size is the posterior MEAN, not the median:
       s = (1 - p) exp(mu0 + sig0^2 / 2) + p exp(mu1 + sig1^2 / 2)
  6. Settings are picked on 5-fold out-of-fold error in mm^2, mean (s - s_true)^2,
     which weights large objects the way the metric does (not sMAPE).

Features: features/dinov2l_reg_518.npz and features/dinov3_vith16plus_512.npz from
the FVE branch (CLS of 3 layers each, the B3.3 set). Without them it falls back to
the DINOv2-B class token and mean patch token from atlas.py.

  uv run baselines/atlas.py cache        # first: clouds give the size labels
  uv run baselines/scale.py              # writes baselines/out/scale/scale.npz
  uv run baselines/scale.py hedge        # tunes geom.hedge -> baselines/out/scale/hedge.json

Hedge: when P(oversized) is unsure, put points at both cluster sizes (geom.hedge).
Its threshold and share are tuned on every training object with its out-of-fold size
and its TRUE unit shape, so only the size decision is being scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "baselines" / "out" / "scale"
ATLAS = ROOT / "baselines" / "out" / "atlas"
GAP = np.log(30.0) / 3.0  # the FVE oversized gap, converted from volume to radius
FVE_FILES = ("dinov2l_reg_518", "dinov3_vith16plus_512")
CLS_BLOCKS = (0, 2, 4)  # [cls, mean patch] x 3 layers; keep the class tokens
FOLDS = 5
ALPHAS = (100.0, 1000.0, 3000.0, 10000.0)
CS = (0.01, 0.1)
W_P = (0.0, 0.25, 0.5, 0.75, 1.0)
Q_MIN = (0.02, 0.05, 0.1, 0.15, 0.2)
FRAC = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)


def cluster_labels(y: np.ndarray, cats: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """u = 1 for oversized: above the largest >= GAP jump in a category, or a whole category GAP above its group."""
    u = np.zeros(len(y), dtype=int)
    for c in np.unique(cats):
        idx = np.flatnonzero(cats == c)
        z = np.sort(y[idx])
        if len(z) > 1 and np.diff(z).max() >= GAP:
            cut = z[np.argmax(np.diff(z)) + 1]
            u[idx[y[idx] >= cut]] = 1
    base = {g: np.median(y[(groups == g) & (u == 0)]) for g in np.unique(groups)}
    for c in np.unique(cats):
        idx = cats == c
        if np.median(y[idx]) - base[groups[idx][0]] >= GAP:
            u[idx] = 1
    return u


class Table:
    """Median log radius per (category, cluster), with fallbacks for an empty cluster."""

    def __init__(self, y: np.ndarray, cats: np.ndarray, groups: np.ndarray, u: np.ndarray) -> None:
        self.med: dict[tuple[str, int], float] = {}
        for c in np.unique(cats):
            for k in (0, 1):
                m = (cats == c) & (u == k)
                if m.any():
                    self.med[(c, k)] = float(np.median(y[m]))
        both = [c for c in np.unique(cats) if (c, 0) in self.med and (c, 1) in self.med]
        self.gap = float(np.median([self.med[(c, 1)] - self.med[(c, 0)] for c in both])) if both else 2.0
        self.group = {g: float(np.median(y[(groups == g) & (u == 0)])) for g in np.unique(groups)}
        self.normal = float(np.median(y[u == 0]))
        self.rate = {c: (float(u[cats == c].sum()) + 0.5) / (float((cats == c).sum()) + 1.0) for c in np.unique(cats)}
        self.prior = float(u.mean())

    def center(self, cat: str, group: str, k: int) -> float:
        if (cat, k) in self.med:
            return self.med[(cat, k)]
        if (cat, 1 - k) in self.med:
            return self.med[(cat, 1 - k)] + (self.gap if k == 1 else -self.gap)
        base = self.group.get(group, self.normal)
        return base + (self.gap if k == 1 else 0.0)

    def centers(self, cats: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return np.array([[self.center(c, g, 0), self.center(c, g, 1)] for c, g in zip(cats, groups)])

    def rates(self, cats: np.ndarray) -> np.ndarray:
        return np.array([self.rate.get(c, self.prior) for c in cats])


def _fve_features(keys: list[str]) -> np.ndarray | None:
    blocks = []
    for name in FVE_FILES:
        path = ROOT / "features" / f"{name}.npz"
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        where = {k: i for i, k in enumerate(z["keys"].tolist())}
        if not all(k in where for k in keys):
            return None
        X = z["X"][[where[k] for k in keys]]
        d = X.shape[1] // 6
        blocks += [X[:, b * d : (b + 1) * d] for b in CLS_BLOCKS]
    return np.concatenate(blocks, 1).astype(np.float32)


def _atlas_features(train_tok: np.ndarray, val_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: DINOv2-B class token and mean patch token."""
    import torch
    from atlas import device, encoder, image_tensor, image_tokens

    def pool(tok: np.ndarray) -> np.ndarray:
        tok = tok.astype(np.float32)
        return np.concatenate([tok[:, 0], tok[:, 1:].mean(1)], 1)

    saved = OUT / "val_dinov2b.npz"
    ids = [p.parent.name for p in val_paths]
    if saved.exists() and np.load(saved)["ids"].tolist() == ids:
        val = np.load(saved)["X"]
    else:
        dev = device()
        net = encoder(dev)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(val_paths), 32):
                x = torch.stack([image_tensor(p) for p in val_paths[i : i + 32]]).to(dev)
                chunks.append(pool(image_tokens(net, x).cpu().numpy()))
        val = np.concatenate(chunks)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(saved, ids=np.array(ids), X=val)
    return pool(train_tok), val


def load() -> dict:
    from atlas import rows

    z = np.load(ATLAS / "train.npz", allow_pickle=False)
    ids = z["ids"].tolist()
    train_paths = rows("train")
    if [p.parent.name for p in train_paths] != ids:
        raise SystemExit("atlas cache is stale. Run: uv run baselines/atlas.py cache")
    val_paths = rows("val")
    pts = z["pts"].astype(np.float64)
    radius = np.sqrt((pts**2).sum(-1).mean(-1))  # clouds are centred already
    keys = [f"train/{i}" for i in ids] + [f"val/{p.parent.name}" for p in val_paths]
    X = _fve_features(keys)
    if X is None:
        print("FVE feature files not found, using DINOv2-B tokens from the atlas cache")
        tr, va = _atlas_features(z["feats"], val_paths)
        X = np.concatenate([tr, va])
    else:
        print(f"features {', '.join(FVE_FILES)}  dim {X.shape[1]}")
    n = len(ids)
    return {
        "ids": np.array(ids + [p.parent.name for p in val_paths]),
        "split": np.array(["train"] * n + ["val"] * len(val_paths)),
        "cats": np.array([p.parent.parent.name for p in train_paths + val_paths]),
        "groups": np.array([p.parent.parent.parent.name for p in train_paths + val_paths]),
        "X": X,
        "y": np.log(radius),
        "n": n,
    }


def ridge_offsets(A: np.ndarray, t: np.ndarray, B: np.ndarray, alphas) -> dict[float, np.ndarray]:
    """Kernel-form ridge with intercept, all alphas from one eigendecomposition."""
    mean = t.mean()
    w, V = np.linalg.eigh(A @ A.T)
    proj = V.T @ (t - mean)
    KB = B @ A.T @ V
    return {a: KB @ (proj / (w + a)) + mean for a in alphas}


def fold_parts(Xtr, ytr, ctr, gtr, Xte, cte, gte) -> dict:
    """Everything one fold needs: cluster centres, ridge offsets per alpha, P(oversized) per C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    u = cluster_labels(ytr, ctr, gtr)
    table = Table(ytr, ctr, gtr, u)
    centre_tr = table.centers(ctr, gtr)[np.arange(len(ytr)), u]
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    off = ridge_offsets(A, ytr - centre_tr, B, ALPHAS)
    clf = {}
    for C in CS:
        if 0 < u.sum() < len(u):
            clf[C] = LogisticRegression(C=C, max_iter=3000).fit(A, u).predict_proba(B)[:, 1]
        else:
            clf[C] = np.full(len(B), float(u.mean()))
    return {"centre": table.centers(cte, gte), "rate": table.rates(cte), "off": off, "clf": clf}


def combine(parts: dict, alpha: float, C: float, w_p: float) -> tuple[np.ndarray, np.ndarray]:
    p = w_p * parts["clf"][C] + (1.0 - w_p) * parts["rate"]
    return p, parts["centre"] + parts["off"][alpha][:, None]


def posterior_mean(p: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (1.0 - p) * np.exp(mu[:, 0] + sig[0] ** 2 / 2) + p * np.exp(mu[:, 1] + sig[1] ** 2 / 2)


def oof_parts(d: dict) -> list[tuple[np.ndarray, dict]]:
    n = d["n"]
    X, y, cats, groups = d["X"][:n], d["y"], d["cats"][:n], d["groups"][:n]
    fold = np.random.default_rng(1).permutation(n) % FOLDS
    out = []
    for f in range(FOLDS):
        te, tr = fold == f, fold != f
        out.append((te, fold_parts(X[tr], y[tr], cats[tr], groups[tr], X[te], cats[te], groups[te])))
        print(f"fold {f + 1}/{FOLDS}", flush=True)
    return out


def oof(d: dict, parts: list, alpha: float, C: float, w_p: float) -> dict:
    n = d["n"]
    y = d["y"]
    p, mu = np.zeros(n), np.zeros((n, 2))
    for te, part in parts:
        p[te], mu[te] = combine(part, alpha, C, w_p)
    u = cluster_labels(y, d["cats"][:n], d["groups"][:n])
    resid = y - mu[np.arange(n), u]
    sig = np.array([resid[u == k].std() if (u == k).sum() > 1 else 0.3 for k in (0, 1)])
    return {"p": p, "mu": mu, "sig": sig, "u": u, "s": posterior_mean(p, mu, sig)}


def report(name: str, s: np.ndarray, r: np.ndarray, u: np.ndarray, p: np.ndarray | None = None) -> float:
    mm2 = float(np.mean((s - r) ** 2))
    smape_r = float(np.mean(2 * np.abs(s - r) / (s + r)) * 100)
    v, vt = s**3, r**3
    smape_v = float(np.mean(2 * np.abs(v - vt) / (v + vt)) * 100)
    line = f"{name:28s} mm2 {mm2:9.1f}  normal {np.mean((s - r)[u == 0] ** 2):8.1f}  "
    line += f"oversized {np.mean((s - r)[u == 1] ** 2):10.1f}  sMAPE radius {smape_r:5.1f}%  volume {smape_v:5.1f}%"
    if p is not None and 0 < u.sum() < len(u):
        from sklearn.metrics import roc_auc_score

        line += f"  AUC {roc_auc_score(u, p):.3f}"
    print(line, flush=True)
    return mm2


def main() -> None:
    d = load()
    n = d["n"]
    r = np.exp(d["y"])
    u = cluster_labels(d["y"], d["cats"][:n], d["groups"][:n])
    print(f"{n} train  {len(d['ids']) - n} val  oversized {u.sum()}  radius median {np.median(r[u == 0]):.1f} mm "
          f"normal, {np.median(r[u == 1]) if u.any() else float('nan'):.1f} mm oversized")
    report("global median", np.full(n, np.median(r)), r, u)
    parts = oof_parts(d)
    best, best_mm2 = None, float("inf")
    for alpha in ALPHAS:
        for C in CS:
            for w_p in W_P:
                res = oof(d, parts, alpha, C, w_p)
                mm2 = report(f"alpha {alpha:g} C {C:g} w_p {w_p:g}", res["s"], r, u, res["p"])
                if mm2 < best_mm2:
                    best, best_mm2 = (alpha, C, w_p, res), mm2
    alpha, C, w_p, res = best
    median = (1 - res["p"]) * np.exp(res["mu"][:, 0]) + res["p"] * np.exp(res["mu"][:, 1])
    print(f"best alpha {alpha:g} C {C:g} w_p {w_p:g}  sig {res['sig'].round(3).tolist()}")
    report("best, posterior mean", res["s"], r, u, res["p"])
    report("best, no sigma^2/2 term", median, r, u)

    X, cats, groups = d["X"], d["cats"], d["groups"]
    p_val, mu_val = combine(fold_parts(X[:n], d["y"], cats[:n], groups[:n], X[n:], cats[n:], groups[n:]), alpha, C, w_p)
    p = np.concatenate([res["p"], p_val])
    mu = np.concatenate([res["mu"], mu_val])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "scale.npz",
        ids=d["ids"],
        split=d["split"],
        p=p,
        mu=mu,
        sig=res["sig"],
        s=posterior_mean(p, mu, res["sig"]),
        radius=np.concatenate([r, np.full(len(d["ids"]) - n, np.nan)]),
        u=np.concatenate([u, np.full(len(d["ids"]) - n, -1)]),
        params=np.array([alpha, C, w_p]),
    )
    print(f"train rows are out-of-fold, val rows from the full fit -> {OUT / 'scale.npz'}")


def cluster_sizes(z) -> np.ndarray:
    """Expected linear radius of each cluster: exp(mu + sig^2 / 2)."""
    return np.exp(z["mu"] + z["sig"][None, :] ** 2 / 2)


def tune_hedge() -> None:
    import torch
    from geom import hedge

    zs = np.load(OUT / "scale.npz", allow_pickle=False)
    pts = np.load(ATLAS / "train.npz", allow_pickle=False)["pts"]
    n = len(pts)
    p, sizes, u = zs["p"][:n], cluster_sizes(zs)[:n], zs["u"][:n]
    r = np.sqrt((pts.astype(np.float64) ** 2).sum(-1).mean(-1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = np.random.default_rng(0).choice(pts.shape[1], 2048, replace=False)

    def cd(pred: np.ndarray, gt: np.ndarray) -> float:
        a = torch.tensor(pred, dtype=torch.float32, device=dev)
        b = torch.tensor(gt, dtype=torch.float32, device=dev)
        d = torch.cdist(a, b).square()
        return float(d.min(1).values.mean() + d.min(0).values.mean())

    unit = [pts[i][sub] / r[i] for i in range(n)]
    gt = [pts[i][sub] for i in range(n)]
    plain = np.array([cd(hedge(unit[i], p[i], sizes[i], 1.0, 0.0), gt[i]) for i in range(n)])
    print(f"no hedge: CD-L2 {plain.mean():.1f}  normal {plain[u == 0].mean():.1f}  oversized {plain[u == 1].mean():.1f}")
    best = (1.0, 0.0, plain.mean())
    for q_min in Q_MIN:
        hit = np.flatnonzero(np.minimum(p, 1 - p) >= q_min)
        for frac in FRAC:
            cds = plain.copy()
            cds[hit] = [cd(hedge(unit[i], p[i], sizes[i], q_min, frac), gt[i]) for i in hit]
            print(f"q_min {q_min:.2f} frac {frac:.3f}  hedged {len(hit):4d}  CD-L2 {cds.mean():8.1f}  "
                  f"normal {cds[u == 0].mean():7.1f}  oversized {cds[u == 1].mean():9.1f}", flush=True)
            if cds.mean() < best[2]:
                best = (q_min, frac, float(cds.mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hedge.json").write_text(json.dumps({"q_min": best[0], "frac": best[1], "cd_true_shape": best[2]}))
    print(f"best q_min {best[0]} frac {best[1]}  CD-L2 {best[2]:.1f} vs {plain.mean():.1f} -> {OUT / 'hedge.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hedge":
        tune_hedge()
    else:
        main()
