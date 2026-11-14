# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""P1 tuning: B3.4 (ridge + within-category kNN) over feature-file combos (CLS-like blocks), official labels.
Select k / w_ridge on seeds 0-2, report fresh seeds 3-5 for each combo's pick.  usage: p1_tune.py [combo names...]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "experiments")]
import b3_4_knn as K
from b0_b1 import smape
from frozen import Config, Experiment

C = K.CLS
COMBOS = {
    "base v2l+v3h": ("dinov2l_reg_518", "dinov3_vith16plus_512"),
    "+siglip": ("dinov2l_reg_518", "dinov3_vith16plus_512", "siglip2_so400m_512"),
    "+v3l": ("dinov2l_reg_518", "dinov3_vith16plus_512", "dinov3_vitl16_512"),
    "v2l+v3h768": ("dinov2l_reg_518", "dinov3_vith16plus_768"),
    "+7b": ("dinov2l_reg_518", "dinov3_vith16plus_512", "dinov3_vit7b_512"),
    "+7b+siglip": ("dinov2l_reg_518", "dinov3_vith16plus_512", "dinov3_vit7b_512", "siglip2_so400m_512"),
    "v2l+7b": ("dinov2l_reg_518", "dinov3_vit7b_512"),
    "all": ("dinov2l_reg_518", "dinov3_vith16plus_512", "siglip2_so400m_512", "dinov3_vitl16_512", "dinov3_vith16plus_768", "dinov3_vit7b_512"),
}
names = sys.argv[1:] or list(COMBOS)
exp = Experiment()
for name in names:
    files = COMBOS[name]
    if not all((ROOT / "features" / f"{f}.npz").exists() for f in files):
        print(f"{name}: missing features, skipped"); continue
    K.CFG = Config(name=name, features=tuple((f, C) for f in files), n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
    sel = [K.collect(exp, s) for s in (0, 1, 2)]
    grid = {(k, w): np.mean([smape(exp.y, np.exp(K.decode(o, k, w, 0.0))) for o in sel]) for k in (1, 2, 3) for w in (0.0, 0.25, 0.5)}
    (k, w), s = min(grid.items(), key=lambda kv: kv[1])
    fresh = np.mean([smape(exp.y, np.exp(K.decode(K.collect(exp, s3), k, w, 0.0))) for s3 in (3, 4, 5)])
    print(f"{name:14s} k={k} w_ridge={w}: seeds0-2 {s:.2f} | fresh3-5 {fresh:.2f}", flush=True)
