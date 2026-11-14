# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "torch"]
# ///
"""S2 tuning (see s12_nca.py for the model).  One run:
 1. diagnostics: OOF error of the S2 baseline split by category size and normal/oversized;
 2. sweep: embedding E x input dropout x value-module input (diff | diff + query) x sibling categories, steps 25..150;
    sibling (step 3): a query whose category has < RARE other training rows also retrieves the training rows of the
    other categories of its food group, with a learned logit penalty (and a cross flag into the value module);
 3. best on seeds 0-2 -> fresh seeds 3-5 (baseline re-run alongside); submission of the best if it differs.
Speed: PCA once per (seed, fold) and reused by every config; configs x seeds in 8 single-thread processes;
mean decode vectorised (the sMAPE-optimal 'med' decode lost everywhere in s12).  usage: s2_tune.py"""
import itertools, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b0_b1 import smape, write_submission
from b3 import argmin_t
from frozen import Config, Experiment
from s12_nca import FEATS, pairs, seg_softmax

D, LR, WD, SAMPLE = 256, 3e-3, 1e-4, 0.5
# round 2 (after S2b, board 28.82): S2b architecture fixed; sweep input features x rare threshold x (shorter) steps.
# *_cls / *_mean = CLS blocks (0,2,4) / mean-patch blocks (1,3,5) of each feature file; all 6 "blocks" = those columns
FSETS = {"F0": [f + "_cls" for f in FEATS],
         "F1": [f + "_cls" for f in FEATS] + ["dinov3_vith16plus_768_cls", "dinov3_vitl16_512_cls"],
         "F2": [f + "_cls" for f in FEATS] + [f + "_mean" for f in FEATS]}
CKPT = (5, 10, 15, 20, 25, 35, 50)
S2B = dict(E=128, drop=0.2, vin="dq", sib=True)
BASE = dict(S2B, fs="F0", rare=5)  # = S2b; compared at 25 steps
BASE_STEP = 25
GRID = [dict(S2B, fs=f, rare=r) for f, r in itertools.product(FSETS, (3, 5, 8, 12))]
G = {}  # filled before the pool forks: exp arrays + PCA cache


def project(Xt, Xq):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    pca = PCA(D, random_state=0).fit((Xt - mu) / sd)
    A = pca.transform((Xt - mu) / sd); s = A.std()
    return (A / s).astype(np.float32), (pca.transform((Xq - mu) / sd) / s).astype(np.float32)


def cand(ca, ga, cb, gb, same_rows, sib, rare_n):
    """candidate pairs: same category; plus (sib) same group when the query's category has < RARE other rows."""
    i, j = pairs(ca, cb, same_rows)
    if not sib: return i, j, torch.zeros(len(i))
    n_same = np.bincount(i.numpy(), minlength=len(ca))
    rare = np.flatnonzero(n_same < rare_n)
    m = (ga[rare][:, None] == gb[None]) & (ca[rare][:, None] != cb[None])
    ri, rj = np.nonzero(m)
    i2, j2 = torch.from_numpy(rare[ri]), torch.from_numpy(rj)
    return torch.cat([i, i2]), torch.cat([j, j2]), torch.cat([torch.zeros(len(i)), torch.ones(len(i2))])


def fit(A, B, yt, ct, gt, cq, gq, cfg, seed):
    """S2 with config cfg; returns {step: ln predictions for B}."""
    A, B, y = torch.from_numpy(A), torch.from_numpy(B), torch.tensor(yt, dtype=torch.float32)
    n, nq, E = len(A), len(B), cfg["E"]
    ti, tj, tx = cand(ct, gt, ct, gt, True, cfg["sib"], cfg["rare"]); qi, qj, qx = cand(cq, gq, ct, gt, False, cfg["sib"], cfg["rare"])
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(cfg["drop"]), nn.Linear(D, E))
    val = nn.Sequential(nn.Linear(E * (2 if cfg["vin"] == "dq" else 1) + cfg["sib"], 64), nn.GELU(), nn.Linear(64, 1))
    nn.init.zeros_(val[2].weight); nn.init.zeros_(val[2].bias)
    with torch.no_grad():
        e0 = enc[1](A); s0 = tx == 0
        log_t = nn.Parameter(-torch.log(((e0[ti[s0]] - e0[tj[s0]]) ** 2).sum(1).median()) + np.log(5.0))
    log_pen = nn.Parameter(torch.tensor(np.log(2.0), dtype=torch.float32))
    opt = torch.optim.AdamW([*enc.parameters(), *val.parameters(), log_t, log_pen], lr=LR, weight_decay=WD)
    fb, out = argmin_t(yt), {}

    def propose(ea, eb, i, j, x):
        d = ea[i] - eb[j]
        inp = torch.cat([d] + ([ea[i]] if cfg["vin"] == "dq" else []) + ([x[:, None]] if cfg["sib"] else []), 1)
        return y[j] + val(inp).squeeze(1).clamp(-2.5, 2.5), -(d ** 2).sum(1) * log_t.exp() - log_pen.exp() * x

    has = torch.zeros(nq, dtype=torch.bool); has[qi] = True
    for step in range(1, max(CKPT) + 1):
        enc.train()
        keep = torch.rand(len(ti)) < SAMPLE
        e = enc(A); i, j = ti[keep], tj[keep]
        v, lg = propose(e, e, i, j, tx[keep])
        w = seg_softmax(lg, i, n)
        loss = torch.zeros(n).index_add(0, i, w * torch.tanh((y[i] - v).abs() / 2)).sum() / len(torch.unique(i))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in CKPT:
            enc.eval()
            with torch.no_grad():
                v, lg = propose(enc(B), enc(A), qi, qj, qx)
                m = torch.zeros(nq).index_add(0, qi, seg_softmax(lg, qi, nq) * v)
            out[step] = torch.where(has, m, torch.tensor(fb, dtype=torch.float32)).numpy().astype(float)
    return out


def oof(args):
    """(config index or -1 for BASE, seed) -> {step: OOF ln predictions}."""
    ci, seed = args
    torch.set_num_threads(1)
    cfg = BASE if ci < 0 else GRID[ci]
    ln, cats, grp = G["ln"], G["cats"], G["grp"]; res = {}
    for k, (tr, va) in enumerate(G["splits"][seed]):
        A, B = G["pca"][(cfg["fs"], seed, k)]
        o = fit(A, B, ln[tr], cats[tr], grp[tr], cats[va], grp[va], cfg, seed * 10 + k)
        for s, v in o.items(): res.setdefault(s, np.full(len(ln), np.nan))[va] = v
    return args, res


def tag(cfg):
    return f"{cfg['fs']}_rare{cfg['rare']}_E{cfg['E']}_d{cfg['drop']}_{cfg['vin']}{'_sib' if cfg['sib'] else ''}"


def main():
    t0 = time.time()
    exp = Experiment()
    XS = {n: exp.features(Config(name=n, features=tuple((f, tuple(range(6))) for f in fs), n_pca=None)) for n, fs in FSETS.items()}
    G.update(ln=exp.ln, cats=exp.cats, grp=exp.train.group.to_numpy(), splits={s: exp.splits(s) for s in range(6)})
    G["pca"] = {(n, s, k): project(XS[n][0][tr], XS[n][0][va]) for n in FSETS for s in range(6)
                for k, (tr, va) in enumerate(G["splits"][s])}
    print(f"PCA cache {len(G['pca'])} folds in {time.time() - t0:.0f}s", flush=True)
    sc = lambda ln_: smape(exp.y, np.exp(ln_))
    with get_context("fork").Pool(os.cpu_count()) as pool:
        # 1. diagnostics + 2. sweep (seeds 0-2)
        R = dict(pool.map(oof, [(ci, s) for ci in [-1, *range(len(GRID))] for s in (0, 1, 2)]))
        base = np.mean([sc(R[(-1, s)][BASE_STEP]) for s in (0, 1, 2)])
        print(f"baseline S2b ({tag(BASE)}, {BASE_STEP} steps) seeds 0-2: {base:.2f}  [round 1: 29.39]", flush=True)
        n_cat = exp.train.category.map(exp.train.category.value_counts()).to_numpy()
        per = np.mean([200 * np.tanh(np.abs(R[(-1, s)][BASE_STEP] - exp.ln) / 2) for s in (0, 1, 2)], 0)
        for name, m in [("category n<5", n_cat < 5), ("n 5-10", (n_cat >= 5) & (n_cat <= 10)), ("n>10", n_cat > 10),
                        ("normal", exp.u_true == 0), ("oversized", exp.u_true == 1)]:
            print(f"  diag {name:13s} rows {m.sum():4d}  sMAPE {per[m].mean():6.2f}  share of total error "
                  f"{per[m].sum() / per.sum() * 100:4.1f}%", flush=True)
        rows = sorted(((np.mean([sc(R[(ci, s)][st]) for s in (0, 1, 2)]), ci, st) for ci in range(len(GRID)) for st in CKPT))
        print("top 10 (seeds 0-2):", flush=True)
        for v, ci, st in rows[:10]: print(f"  {v:.2f}  {tag(GRID[ci])} steps {st}", flush=True)
        for key in ("fs", "rare"):
            for val_ in sorted({c[key] for c in GRID}, key=str):
                best_v = min(v for v, ci, st in rows if GRID[ci][key] == val_)
                print(f"  best with {key}={val_}: {best_v:.2f}", flush=True)
        _, bci, bst = rows[0]
        # 3. fresh seeds for best and baseline
        F = dict(pool.map(oof, [(bci, s) for s in (3, 4, 5)] + [(-1, s) for s in (3, 4, 5)]))
    fb = np.mean([sc(F[(bci, s)][bst]) for s in (3, 4, 5)]); f0 = np.mean([sc(F[(-1, s)][BASE_STEP]) for s in (3, 4, 5)])
    print(f"BEST {tag(GRID[bci])} steps {bst}: seeds0-2 {rows[0][0]:.2f} | fresh3-5 {fb:.2f}  vs baseline "
          f"{base:.2f} | {f0:.2f}  (gain fresh {f0 - fb:+.2f})  [{time.time() - t0:.0f}s]", flush=True)
    A, B = project(*XS[GRID[bci]["fs"]]); torch.set_num_threads(os.cpu_count())
    ln = fit(A, B, exp.ln, exp.cats, G["grp"], exp.val.category.to_numpy(), exp.val.group.to_numpy(), GRID[bci], 0)[bst]
    pred = np.exp(ln)
    print(write_submission(f"s2b_{tag(GRID[bci])}_s{bst}", exp.val, pred),
          f"min {pred.min():.2f} median {np.median(pred):.2f} max {pred.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
