# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B5.2: do frozen ConvNeXt features help the B3.4 (ridge + cluster + kNN) pipeline?  Seeds 0-2 select, 3-5 check.

Usage: b5_2_cnn_features.py STEM [STEM...]   (feature stems in features/, blocks (0,1,2))
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
import b3_4_knn as knn
from b0_b1 import smape
from frozen import Config, Experiment

knn.KS = (2,)
BASE = (("dinov2l_reg_518", (0, 2, 4)), ("dinov3_vith16plus_512", (0, 2, 4)))
exp = Experiment()
stems = sys.argv[1:]
variants = {"B3.4 baseline": BASE}
for st in stems:
    variants[f"{st} alone"] = ((st, (0, 1, 2)),)
    variants[f"baseline + {st}"] = BASE + ((st, (0, 1, 2)),)
res = {}
for name, feats in variants.items():
    knn.CFG = Config(name=name, features=feats, n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
    for label, seeds in (("sel", (0, 1, 2)), ("fresh", (3, 4, 5))):
        s = [smape(exp.y, np.exp(knn.decode(knn.collect(exp, sd), 2, .25, .25))) for sd in seeds]
        res[(name, label)] = np.mean(s)
    print(f"{name:55s} seeds0-2 {res[(name,'sel')]:.2f} | seeds3-5 {res[(name,'fresh')]:.2f}", flush=True)
