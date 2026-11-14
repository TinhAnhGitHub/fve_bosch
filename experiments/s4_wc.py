# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "torch"]
# ///
"""S4 = S2b (s2_tune.py; board 28.53 with the F1 features) with two changes:
 1. within-category basis: the 256 input dims are the principal axes of the CATEGORY-CENTRED training features (pooled
    within-category scatter) instead of plain PCA.  The model only compares rows inside a category, plain PCA spends its
    axes on between-category differences.  Rows are projected uncentred, so same-category differences are exact.
 2. seed ensemble: K models (different init / neighbour sampling), ln predictions averaged.
Sweep (seeds 0-2, one model each): BASE (= S2b, plain PCA) + within-category basis x E x dropout x distance, steps 15..50.
Fresh seeds 3-5: best and BASE, single model and K_BAG ensemble.  Submission: best, K_SUB models on all train.
Check: BASE at 25 steps should print ~28.93 (s2_tune round 2).  usage: s4_wc.py"""
import itertools, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b0_b1 import smape, write_submission
from b3 import argmin_t
from frozen import Config, Experiment
from s12_nca import FEATS, seg_softmax
from s2_tune import cand

FSET = [f + "_cls" for f in FEATS] + ["dinov3_vith16plus_768_cls", "dinov3_vitl16_512_cls"]
D, LR, WD, SAMPLE, RARE = 256, 3e-3, 1e-4, 0.5, 8
CKPT, BASE_STEP, K_BAG, K_SUB = (15, 25, 35, 50, 75), 25, 4, 8
BASE = dict(space="pca", E=128, drop=0.2, dist="sq")
GRID = [BASE] + [dict(space="wc", E=e, drop=p, dist=d) for e, p, d in itertools.product((128, 256), (0.1, 0.2), ("sq", "euc"))]
G = {}  # filled before the pool forks


def project(Xt, Xq, ct):
    """{space: (train coords, query coords)} on the top-D axes of the total ('pca') / within-category ('wc') scatter."""
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Zt, Zq = ((Xt - mu) / sd).astype(np.float32), ((Xq - mu) / sd).astype(np.float32)
    zm = Zt.mean(0); Zt -= zm; Zq -= zm
    W = Zt.copy()
    for c in np.unique(ct): W[ct == c] -= W[ct == c].mean(0)
    out = {}
    for name, C in (("pca", Zt), ("wc", W)):  # axes = C.T @ P (unit norm), via the n x n gram matrix
        w, V = np.linalg.eigh((C @ C.T).astype(np.float64)); idx = np.argsort(-w)[:D]
        P = (V[:, idx] / np.sqrt(w[idx])).astype(np.float32)
        A, B = (Zt @ C.T) @ P, (Zq @ C.T) @ P; s = A.std()
        out[name] = (A / s, B / s)
    return out


def fit(A, B, yt, ct, gt, cq, gq, cfg, seed, ckpt=CKPT):
    """S2b on (A, B) with config cfg; returns {step: ln predictions for B}."""
    A, B, y = torch.from_numpy(A), torch.from_numpy(B), torch.tensor(yt, dtype=torch.float32)
    n, nq, E = len(A), len(B), cfg["E"]
    ti, tj, tx = cand(ct, gt, ct, gt, True, True, RARE); qi, qj, qx = cand(cq, gq, ct, gt, False, True, RARE)
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(cfg["drop"]), nn.Linear(D, E))
    val = nn.Sequential(nn.Linear(E * 2 + 1, 64), nn.GELU(), nn.Linear(64, 1))
    nn.init.zeros_(val[2].weight); nn.init.zeros_(val[2].bias)
    dist = (lambda d: (d ** 2).sum(1)) if cfg["dist"] == "sq" else (lambda d: ((d ** 2).sum(1) + 1e-8).sqrt())
    with torch.no_grad():
        e0 = enc[1](A); s0 = tx == 0
        log_t = nn.Parameter(-torch.log(dist(e0[ti[s0]] - e0[tj[s0]]).median()) + np.log(5.0))
    log_pen = nn.Parameter(torch.tensor(np.log(2.0), dtype=torch.float32))
    opt = torch.optim.AdamW([*enc.parameters(), *val.parameters(), log_t, log_pen], lr=LR, weight_decay=WD)
    fb, out = argmin_t(yt), {}

    def propose(ea, eb, i, j, x):
        d = ea[i] - eb[j]
        return (y[j] + val(torch.cat([d, ea[i], x[:, None]], 1)).squeeze(1).clamp(-2.5, 2.5),
                -dist(d) * log_t.exp() - log_pen.exp() * x)

    has = torch.zeros(nq, dtype=torch.bool); has[qi] = True
    for step in range(1, max(ckpt) + 1):
        enc.train()
        keep = torch.rand(len(ti)) < SAMPLE
        e = enc(A); i, j = ti[keep], tj[keep]
        v, lg = propose(e, e, i, j, tx[keep])
        w = seg_softmax(lg, i, n)
        loss = torch.zeros(n).index_add(0, i, w * torch.tanh((y[i] - v).abs() / 2)).sum() / len(torch.unique(i))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in ckpt:
            enc.eval()
            with torch.no_grad():
                v, lg = propose(enc(B), enc(A), qi, qj, qx)
                m = torch.zeros(nq).index_add(0, qi, seg_softmax(lg, qi, nq) * v)
            out[step] = torch.where(has, m, torch.tensor(fb, dtype=torch.float32)).numpy().astype(float)
    return out


def oof(args):
    """(config index, split seed, model seed) -> {step: OOF ln predictions}."""
    ci, seed, ms = args
    torch.set_num_threads(1)
    cfg = GRID[ci]; ln, cats, grp = G["ln"], G["cats"], G["grp"]; res = {}
    for k, (tr, va) in enumerate(G["splits"][seed]):
        A, B = G["proj"][(seed, k)][cfg["space"]]
        o = fit(A, B, ln[tr], cats[tr], grp[tr], cats[va], grp[va], cfg, seed * 10 + k + 1000 * ms)
        for s, v in o.items(): res.setdefault(s, np.full(len(ln), np.nan))[va] = v
    return args, res


def tag(cfg):
    return f"{cfg['space']}_E{cfg['E']}_d{cfg['drop']}_{cfg['dist']}"


def main():
    t0 = time.time()
    exp = Experiment()
    X, Xv = exp.features(Config(name="s4", features=tuple((f, tuple(range(6))) for f in FSET), n_pca=None))
    G.update(ln=exp.ln, cats=exp.cats, grp=exp.train.group.to_numpy(), splits={s: exp.splits(s) for s in range(6)})
    G["proj"] = {(s, k): project(X[tr], X[va], exp.cats[tr]) for s in range(6) for k, (tr, va) in enumerate(G["splits"][s])}
    print(f"projections {len(G['proj'])} folds in {time.time() - t0:.0f}s", flush=True)
    sc = lambda ln_: smape(exp.y, np.exp(ln_))
    with get_context("fork").Pool(os.cpu_count()) as pool:
        R = dict(pool.map(oof, [(ci, s, 0) for ci in range(len(GRID)) for s in (0, 1, 2)]))
        rows = sorted((np.mean([sc(R[(ci, s, 0)][st]) for s in (0, 1, 2)]), ci, st) for ci in range(len(GRID)) for st in CKPT)
        base = np.mean([sc(R[(0, s, 0)][BASE_STEP]) for s in (0, 1, 2)])
        print(f"BASE {tag(BASE)} steps {BASE_STEP} seeds 0-2: {base:.2f}  [s2_tune round 2: 28.93]", flush=True)
        print("all configs (seeds 0-2), best step first:", flush=True)
        for ci in range(len(GRID)):
            print(f"  {tag(GRID[ci]):22s} " + " ".join(f"s{st}={v:.2f}" for v, c, st in sorted(rows, key=lambda r: r[2]) if c == ci), flush=True)
        for key in ("E", "drop", "dist"):
            for val_ in sorted({c[key] for c in GRID[1:]}, key=str):
                print(f"  best wc with {key}={val_}: {min(v for v, ci, st in rows if ci and GRID[ci][key] == val_):.2f}", flush=True)
        _, bci, bst = rows[0]
        # fresh seeds: best and BASE, K_BAG model seeds each
        F = dict(pool.map(oof, [(ci, s, m) for ci in (bci, 0) for s in (3, 4, 5) for m in range(K_BAG)]))
    def fresh(ci, st):
        single = np.mean([sc(F[(ci, s, m)][st]) for s in (3, 4, 5) for m in range(K_BAG)])
        return single, np.mean([sc(np.mean([F[(ci, s, m)][st] for m in range(K_BAG)], 0)) for s in (3, 4, 5)])
    (b1, bk), (f1, fk) = fresh(0, BASE_STEP), fresh(bci, bst)
    print(f"BEST {tag(GRID[bci])} steps {bst}: seeds0-2 {rows[0][0]:.2f} | fresh3-5 single {f1:.2f}, x{K_BAG} ensemble {fk:.2f}\n"
          f"BASE fresh3-5 single {b1:.2f}, x{K_BAG} ensemble {bk:.2f}  (gain of best ensemble vs BASE single {b1 - fk:+.2f})  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    P = project(X, Xv, exp.cats)[GRID[bci]["space"]]; torch.set_num_threads(os.cpu_count())
    ln = np.mean([fit(*P, exp.ln, exp.cats, G["grp"], exp.val.category.to_numpy(), exp.val.group.to_numpy(), GRID[bci], 1000 * m,
                      (bst,))[bst] for m in range(K_SUB)], 0)
    pred = np.exp(ln)
    print(write_submission(f"s4_{tag(GRID[bci])}_s{bst}_k{K_SUB}", exp.val, pred),
          f"min {pred.min():.2f} median {np.median(pred):.2f} max {pred.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
