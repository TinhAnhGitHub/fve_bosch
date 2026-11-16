# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
"""Object size (RMS radius, mm) from one photo: learned same-category kNN, the S4 technique of the volume track
(fve_bosch experiments/s4_wc.py), trained here on our own target, the true RMS radius of each training mesh.

How it predicts one object:
  1. frozen CLS features of several backbones, standardised, projected on the 256 axes of the within-category scatter;
  2. a linear layer (dropout) maps them to a space where close = similar size;
  3. soft neighbours = training objects of the same category (+ the food group's other categories, with a learned
     penalty, when the category has < RARE other rows); weight = softmax(-distance * learned temperature);
  4. each neighbour proposes its own size + a learned correction ("this photo looks bigger/smaller than me");
  5. prediction = weighted mean of the proposals. Trained leave-one-out on the training part, loss tanh(|y - v| / 2).
The target is y = 3 ln(radius) (a volume-like log), so S4's loss and settings apply unchanged.

Folds are ours (folds/f3dr_k5_v1.csv), so each out-of-fold size comes from models that never saw the object and lines
up with the out-of-fold clouds. Runs on CPU in minutes.

usage: size_knn.py OUT_DIR [--weighted] [--axes]
  writes OUT_DIR/size_knn{_w}.npz: ids, fold, oof_ln_r {step: (2042,)}, eval_ids, eval_ln_r {step: (511,)}, radius (true)
  --weighted: each object's loss weighted by its category's median radius^2 (the score is in mm^2)
  --axes: three targets instead of one, the spread (std, mm) of the true cloud along its longest, middle and shortest
          principal axis, one model each; writes size_knn_axes.npz with oof_ln_sd (steps, 3, 2042), eval_ln_sd, sd
"""
import csv
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data" / "f3d"
FEAT_DIRS = (ROOT.parent / "fve_bosch" / "features", ROOT / "baselines" / "out" / "atlas" / "feats")  # Track 2 files; ours (tools/extract_feats.py)
FOLDS = ROOT / "baselines" / "folds" / "f3dr_k5_v1.csv"
FEATS = ("dinov2l_reg_518", "dinov3_vith16plus_512", "siglip2_so400m_512", "dinov3_vitl16_512", "dinov3_vitl16_768")
CLS_BLOCKS = (0, 2, 4)  # each feature file holds 6 blocks: CLS and mean patch of 3 layers
D, E, DROP, LR, WD, SAMPLE, RARE = 256, 256, 0.1, 3e-3, 1e-4, 0.5, 8
CKPT = (15, 25, 35, 50, 75)
K_BAG = 4   # model seeds averaged per fold (out-of-fold)
K_SUB = 8   # model seeds averaged for the eval photos (trained on all 2,042)
G: dict = {}  # filled before the pool forks


def features(keys: list[str], stems: tuple[str, ...]) -> np.ndarray:
    parts = []
    for stem in stems:
        z = np.load(next(d / f"{stem}.npz" for d in FEAT_DIRS if (d / f"{stem}.npz").exists()))
        row = {k: i for i, k in enumerate(z["keys"])}
        X = z["X"][[row[k] for k in keys]]
        d = X.shape[1] // 6
        parts.append(X[:, np.concatenate([np.arange(b * d, (b + 1) * d) for b in CLS_BLOCKS])])
    return np.hstack(parts).astype(np.float32)


def project(Xt: np.ndarray, Xq: np.ndarray, ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-D axes of the within-category scatter of the standardised training features (S4 'wc' basis)."""
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Zt, Zq = (Xt - mu) / sd, (Xq - mu) / sd
    zm = Zt.mean(0)
    Zt, Zq = Zt - zm, Zq - zm
    W = Zt.copy()
    for c in np.unique(ct):
        W[ct == c] -= W[ct == c].mean(0)
    w, V = np.linalg.eigh((W @ W.T).astype(np.float64))
    idx = np.argsort(-w)[:D]
    P = (V[:, idx] / np.sqrt(w[idx])).astype(np.float32)
    A, B = (Zt @ W.T) @ P, (Zq @ W.T) @ P
    s = A.std()
    return A / s, B / s


def cand(ca: np.ndarray, ga: np.ndarray, cb: np.ndarray, gb: np.ndarray, same_rows: bool):
    """Neighbour pairs (i query, j training row): same category; plus the same food group when the query's category
    has fewer than RARE other training rows (flag x = 1)."""
    m = ca[:, None] == cb[None]
    if same_rows:
        np.fill_diagonal(m, False)
    i, j = np.nonzero(m)
    n_same = np.bincount(i, minlength=len(ca))
    rare = np.flatnonzero(n_same < RARE)
    ri, rj = np.nonzero((ga[rare][:, None] == gb[None]) & (ca[rare][:, None] != cb[None]))
    i2, j2 = rare[ri], rj
    return (torch.from_numpy(np.concatenate([i, i2])), torch.from_numpy(np.concatenate([j, j2])),
            torch.cat([torch.zeros(len(i)), torch.ones(len(i2))]))


def seg_softmax(x: torch.Tensor, i: torch.Tensor, n: int) -> torch.Tensor:
    mx = torch.full((n,), -1e30).scatter_reduce(0, i, x, "amax")
    ex = (x - mx[i]).exp()
    return ex / torch.zeros(n).index_add(0, i, ex)[i]


def fit(A: np.ndarray, B: np.ndarray, yt: np.ndarray, wt: np.ndarray, ct: np.ndarray, gt: np.ndarray,
        cq: np.ndarray, gq: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """S4 on (A train, B query); returns {step: predicted y for B}."""
    At, Bt, y, wobj = torch.from_numpy(A), torch.from_numpy(B), torch.tensor(yt, dtype=torch.float32), torch.tensor(wt, dtype=torch.float32)
    n, nq = len(At), len(Bt)
    ti, tj, tx = cand(ct, gt, ct, gt, True)
    qi, qj, qx = cand(cq, gq, ct, gt, False)
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(DROP), nn.Linear(D, E))
    val = nn.Sequential(nn.Linear(E * 2 + 1, 64), nn.GELU(), nn.Linear(64, 1))
    nn.init.zeros_(val[2].weight)
    nn.init.zeros_(val[2].bias)
    dist = lambda d: ((d ** 2).sum(1) + 1e-8).sqrt()  # noqa: E731
    with torch.no_grad():
        e0 = enc[1](At)
        s0 = tx == 0
        log_t = nn.Parameter(-torch.log(dist(e0[ti[s0]] - e0[tj[s0]]).median()) + np.log(5.0))
    log_pen = nn.Parameter(torch.tensor(np.log(2.0), dtype=torch.float32))
    opt = torch.optim.AdamW([*enc.parameters(), *val.parameters(), log_t, log_pen], lr=LR, weight_decay=WD)
    fallback = float(np.median(yt))

    def propose(ea, eb, i, j, x):
        d = ea[i] - eb[j]
        return (y[j] + val(torch.cat([d, ea[i], x[:, None]], 1)).squeeze(1).clamp(-2.5, 2.5),
                -dist(d) * log_t.exp() - log_pen.exp() * x)

    has = torch.zeros(nq, dtype=torch.bool)
    has[qi] = True
    out = {}
    for step in range(1, max(CKPT) + 1):
        enc.train()
        keep = torch.rand(len(ti)) < SAMPLE
        e = enc(At)
        i, j = ti[keep], tj[keep]
        v, lg = propose(e, e, i, j, tx[keep])
        w = seg_softmax(lg, i, n)
        per = torch.zeros(n).index_add(0, i, w * torch.tanh((y[i] - v).abs() / 2))
        seen = torch.zeros(n, dtype=torch.bool)
        seen[i] = True
        loss = (per * wobj)[seen].sum() / wobj[seen].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in CKPT:
            enc.eval()
            with torch.no_grad():
                v, lg = propose(enc(Bt), enc(At), qi, qj, qx)
                m = torch.zeros(nq).index_add(0, qi, seg_softmax(lg, qi, nq) * v)
            out[step] = torch.where(has, m, torch.tensor(fallback)).numpy().astype(np.float64)
    return out


def job(args: tuple[int, int]) -> tuple[tuple[int, int], dict[int, np.ndarray]]:
    """(fold or -1 for all-train -> eval, model seed) -> {step: predicted y}."""
    k, ms = args
    torch.set_num_threads(1)
    g = G
    if k < 0:
        A, B = g["proj"][-1]
        return args, fit(A, B, g["y"], g["w"], g["cat"], g["grp"], g["ecat"], g["egrp"], 1000 * ms + 99)
    tr, va = np.flatnonzero(g["fold"] != k), np.flatnonzero(g["fold"] == k)
    A, B = g["proj"][k]
    return args, fit(A, B, g["y"][tr], g["w"][tr], g["cat"][tr], g["grp"][tr], g["cat"][va], g["grp"][va], 1000 * ms + k)


def train_all(y: np.ndarray, n_eval: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Out-of-fold and eval predictions of target y (S4 scale: 3 ln size) averaged over model seeds."""
    G["y"] = y
    tasks = [(k, m) for k in range(5) for m in range(K_BAG)] + [(-1, m) for m in range(K_SUB)]
    with get_context("fork").Pool(14) as pool:
        res = dict(pool.map(job, tasks))
    oof = {s: np.zeros(len(y)) for s in CKPT}
    ev = {s: np.zeros(n_eval) for s in CKPT}
    for (k, m), r in res.items():
        for s, v in r.items():
            if k < 0:
                ev[s] += v / K_SUB
            else:
                oof[s][G["fold"] == k] += v / K_BAG
    return oof, ev


def principal_sd(ids: list[str]) -> np.ndarray:
    """(n, 3) std of each true training cloud along its principal axes, longest first."""
    with np.load(ROOT / "baselines" / "out" / "atlas" / "train.npz") as z:
        pos = {i: n for n, i in enumerate(z["ids"].tolist())}
        pts = z["pts"][[pos[i] for i in ids]].astype(np.float64)
    pts -= pts.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts) / pts.shape[1]
    return np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[:, ::-1], 1e-12))


def main() -> None:
    out_dir = Path(sys.argv[1])
    weighted = "--weighted" in sys.argv[2:]
    axes = "--axes" in sys.argv[2:]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[2:] if "=" in a)  # --feats=a,b replaces FEATS; --add=a,b adds
    stems = tuple(opt["feats"].split(",")) if "feats" in opt else FEATS + tuple(opt.get("add", "").split(",") if "add" in opt else ())
    t0 = time.time()
    with open(FOLDS) as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    fold = np.array([int(r["fold"]) for r in rows])
    cat = np.array([r["category"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    radius = np.array([float(r["radius"]) for r in rows])
    eval_dirs = sorted(p for p in (DATA / "eval").glob("*/*/*") if p.is_dir())
    eids = [p.name for p in eval_dirs]
    ecat = np.array([p.parent.name for p in eval_dirs])
    egrp = np.array([p.parent.parent.name for p in eval_dirs])
    X = features([f"train/{i}" for i in ids], stems)
    Xe = features([f"val/{i}" for i in eids], stems)
    y = 3 * np.log(radius)
    if weighted:
        med = {c: np.median(radius[cat == c]) for c in np.unique(cat)}
        w = np.array([med[c] ** 2 for c in cat])
        w = w / w.mean()
    else:
        w = np.ones(len(y))
    print(f"{len(ids)} train, {len(eids)} eval, features {X.shape[1]}, weighted={weighted}", flush=True)
    proj = {k: project(X[fold != k], X[fold == k], cat[fold != k]) for k in range(5)}
    proj[-1] = project(X, Xe, cat)
    G.update(w=w, cat=cat, grp=grp, ecat=ecat, egrp=egrp, fold=fold, proj=proj)
    print(f"projections ready {time.time() - t0:.0f} s", flush=True)
    if axes:
        sd = principal_sd(ids)
        res = []
        for a in range(3):
            res.append(train_all(3 * np.log(sd[:, a]), len(eids)))
            print(f"axis {a} trained {time.time() - t0:.0f} s", flush=True)
        for a, name in enumerate(("longest", "middle", "shortest")):
            for s in CKPT:
                e = res[a][0][s] / 3 - np.log(sd[:, a])
                print(f"axis {name:8s} step {s}: median |log err| {np.median(np.abs(e)):.4f}  mean {np.mean(np.abs(e)):.4f}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "size_knn_axes.npz", ids=np.array(ids), fold=fold, sd=sd, eval_ids=np.array(eids), steps=np.array(CKPT),
                 oof_ln_sd=np.stack([np.stack([res[a][0][s] / 3 for a in range(3)]) for s in CKPT]),
                 eval_ln_sd=np.stack([np.stack([res[a][1][s] / 3 for a in range(3)]) for s in CKPT]))
        print(f"saved {out_dir / 'size_knn_axes.npz'}  ({time.time() - t0:.0f} s)")
        return
    oof, ev = train_all(y, len(eids))
    print(f"trained {time.time() - t0:.0f} s", flush=True)
    true = np.log(radius)
    cat_med = np.zeros(len(y))
    for k in range(5):
        tr = fold != k
        for c in np.unique(cat[fold == k]):
            pool_c = true[tr & (cat == c)]
            cat_med[(fold == k) & (cat == c)] = np.median(pool_c) if len(pool_c) else np.median(true[tr])
    q = np.quantile(radius, [0.2, 0.4, 0.6, 0.8])
    quint = np.searchsorted(q, radius)
    rep = lambda name, e: print(f"{name:22s} median |log err| {np.median(np.abs(e)):.4f}  mean {np.mean(np.abs(e)):.4f}  "  # noqa: E731
                                f"by size quintile " + " ".join(f"{np.median(np.abs(e[quint == b])):.3f}" for b in range(5)))
    rep("category median", cat_med - true)
    for s in CKPT:
        rep(f"size_knn step {s}", oof[s] / 3 - true)
    tag = "_w" if weighted else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"size_knn{tag}.npz", ids=np.array(ids), fold=fold, radius=radius, eval_ids=np.array(eids),
             steps=np.array(CKPT), oof_ln_r=np.stack([oof[s] / 3 for s in CKPT]), eval_ln_r=np.stack([ev[s] / 3 for s in CKPT]))
    print(f"saved {out_dir / f'size_knn{tag}.npz'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
