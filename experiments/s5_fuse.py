# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "torch"]
# ///
"""S5 = late fusion of S4 models (s4_wc.py; board 26.15).  One S4 model (wc basis, E256, drop 0.1, euc) per VIEW (= one
feature set, own standardise + within-category PCA), ln predictions averaged across views with equal view weight.
Lab (split seed 0, x4 ensembles, 35 steps): six-backbone S4 26.78, H+@768 alone 26.77, 7B@512 alone 27.40, H+768 + 7B512 26.25.
Protocol: every view, split seeds 0-5 x K_BAG model seeds, 5-fold OOF ln predictions at steps CKPT.  Seeds 0-2 pick each view's
step (x4 ensemble) and then the best non-empty subset of views.  Fresh seeds 3-5: single views, picked fusion, and its gain vs
the f1 x4 ensemble (go/no-go: gain >= 0.25).  All OOF saved to out_s5/oof.npz.  Submission: picked subset, K_SUB models per view.
Check: seed 0, models 0-1, step 35: h768 single ~27.2, b7_512 single ~27.85, h768+b7_512 (x2 each) ~26.35.
usage: s5_fuse.py [view ...]   (default h768 b7_512 f1; views: see VIEWS)
Phase-B features (GPU; feature file must exist before a view is used):
  uv run features/extract_dinov2.py --model facebook/dinov3-vith16plus-pretrain-lvd1689m --size 1024 --layers 16 24 32 \\
      --dtype bfloat16 --out dinov3_vith16plus_1024
  uv run features/extract_dinov2.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m --size 768 --layers 20 30 40 \\
      --dtype bfloat16 --out dinov3_vit7b_768"""
import itertools, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
from b0_b1 import smape, write_submission
from frozen import Config, Experiment
from s4_wc import FSET, K_BAG, K_SUB, fit, project

VIEWS = dict(h768=["dinov3_vith16plus_768"], b7_512=["dinov3_vit7b_512"], f1=FSET,
             h1024=["dinov3_vith16plus_1024"], b7_768=["dinov3_vit7b_768"])
CFG, CKPT, OUT = dict(space="wc", E=256, drop=0.1, dist="euc"), (25, 35, 50), ROOT / "experiments" / "out_s5"
G = {}  # filled before the pool forks


def spec(view):
    """((npz stem, blocks), ...) for a view; exits if a file is missing.  <s>_cls.npz (all 6 blocks) else <s>.npz CLS blocks."""
    if view not in VIEWS: sys.exit(f"unknown view {view!r}; known: {', '.join(VIEWS)}")
    out = []
    for s in VIEWS[view]:
        c = s if s.endswith("_cls") else s + "_cls"
        if (ROOT / "features" / f"{c}.npz").exists(): out.append((c, tuple(range(6))))
        elif not s.endswith("_cls") and (ROOT / "features" / f"{s}.npz").exists(): out.append((s, (0, 2, 4)))
        else: sys.exit(f"view {view}: missing features/{c}.npz" + ("" if s.endswith("_cls") else f" and features/{s}.npz"))
    return tuple(out)


def load(exp, view):
    """(X, Xv) for one view; drops the Experiment's raw-file cache so only one view is in memory."""
    X, Xv = exp.features(Config(name="s5", features=spec(view), n_pca=None)); exp._npz.clear()
    return X, Xv


def prepare(exp, views, seeds):
    """Fill G: labels, splits and the wc projection of every (view, split seed, fold)."""
    G.update(ln=exp.ln, cats=exp.cats, grp=exp.train.group.to_numpy(), splits={s: exp.splits(s) for s in seeds}, proj={})
    for v in views:
        t0, X = time.time(), load(exp, v)[0]
        G["proj"].update({(v, s, k): project(X[tr], X[va], exp.cats[tr])["wc"] for s in seeds for k, (tr, va) in enumerate(G["splits"][s])})
        print(f"projections {v} dim {X.shape[1]} in {time.time() - t0:.0f}s", flush=True); del X


def oof(args):
    """(view, split seed, model seed) -> {step: OOF ln predictions}."""
    v, seed, ms = args
    torch.set_num_threads(1)
    ln, cats, grp = G["ln"], G["cats"], G["grp"]; res = {}
    for k, (tr, va) in enumerate(G["splits"][seed]):
        o = fit(*G["proj"][(v, seed, k)], ln[tr], cats[tr], grp[tr], cats[va], grp[va], CFG, seed * 10 + k + 1000 * ms, CKPT)
        for s, p in o.items(): res.setdefault(s, np.full(len(ln), np.nan))[va] = p
    return args, res


def run(views, seeds, mseeds, procs=os.cpu_count()):
    with get_context("fork").Pool(procs) as pool:
        return dict(pool.map(oof, [(v, s, m) for v in views for s in seeds for m in mseeds]))


def main():
    t0, views = time.time(), sys.argv[1:] or ["h768", "b7_512", "f1"]
    for v in views: spec(v)  # fail fast on missing files
    OUT.mkdir(exist_ok=True); R = {}
    if (OUT / "oof.npz").exists():  # reuse views trained by an earlier run
        with np.load(OUT / "oof.npz") as z:
            for key in (k for k in z.files if "|" in k):
                v, s, m, t = key.split("|"); R.setdefault((v, int(s[1:]), int(m[1:])), {})[int(t[1:])] = z[key]
    new = [v for v in views if (v, 0, 0) not in R]
    exp = Experiment(); prepare(exp, new, range(6))
    R.update(run(new, range(6), range(K_BAG)))
    np.savez(OUT / "oof.npz", y=exp.y, **{f"{v}|s{s}|m{m}|t{t}": p for (v, s, m), o in R.items() for t, p in o.items()})
    sc = lambda ln_: smape(exp.y, np.exp(ln_))
    ens = lambda v, s, t: np.mean([R[(v, s, m)][t] for m in range(K_BAG)], 0)
    fuse = lambda sub, s: np.mean([ens(v, s, best[v]) for v in sub], 0)
    best = {}
    for v in views:
        r = {t: np.mean([sc(ens(v, s, t)) for s in (0, 1, 2)]) for t in CKPT}; best[v] = min(r, key=r.get)
        print(f"{v:8s} x{K_BAG} seeds 0-2: " + " ".join(f"s{t}={x:.2f}" for t, x in r.items()) + f"  -> step {best[v]}", flush=True)
    subs = sorted((np.mean([sc(fuse(sub, s)) for s in (0, 1, 2)]), sub)
                  for n in range(1, len(views) + 1) for sub in itertools.combinations(views, n))
    print("subsets (seeds 0-2, x4 per view):", flush=True)
    for x, sub in subs: print(f"  {'+'.join(sub):30s} {x:.2f}", flush=True)
    pick = subs[0][1]
    print("fresh seeds 3-5:", flush=True)
    for v in views:
        single = np.mean([sc(R[(v, s, m)][best[v]]) for s in (3, 4, 5) for m in range(K_BAG)])
        print(f"  {v:8s} step {best[v]}: single {single:.2f}, x{K_BAG} ensemble {np.mean([sc(ens(v, s, best[v])) for s in (3, 4, 5)]):.2f}", flush=True)
    fk = np.mean([sc(fuse(pick, s)) for s in (3, 4, 5)])
    gain = f"  gain vs f1 x{K_BAG} {np.mean([sc(ens('f1', s, best['f1'])) for s in (3, 4, 5)]) - fk:+.2f} (gate >= 0.25)" if "f1" in views else ""
    print(f"PICK {'+'.join(pick)}: seeds0-2 {subs[0][0]:.2f} | fresh3-5 fusion {fk:.2f}{gain}  [{time.time() - t0:.0f}s]", flush=True)
    torch.set_num_threads(os.cpu_count()); cq, gq, lns = exp.val.category.to_numpy(), exp.val.group.to_numpy(), []
    for v in pick:
        P = project(*load(exp, v), exp.cats)["wc"]
        lns.append(np.mean([fit(*P, exp.ln, exp.cats, G["grp"], cq, gq, CFG, 1000 * m, (best[v],))[best[v]] for m in range(K_SUB)], 0))
    pred = np.exp(np.mean(lns, 0))
    print(write_submission(f"s5_{'+'.join(pick)}_k{K_SUB}", exp.val, pred),
          f"min {pred.min():.2f} median {np.median(pred):.2f} max {pred.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
