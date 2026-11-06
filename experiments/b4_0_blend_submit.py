# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Blend submission: B3.4 val predictions + mean of B4.0 fold heads, ln-space weight W on B4.0, B3.4 call on cluster disagreement."""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
from b0_b1 import write_submission
from b3 import build_table, cluster_labels
from frozen import Experiment

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
FOLDS = tuple(int(f) for f in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1)
NAME = sys.argv[3] if len(sys.argv) > 3 else "b4_0_blend_w04"
exp = Experiment()
csv = pd.read_csv(ROOT / "submissions/b3_4_knn/preds_volume.csv")
assert list(csv.iloc[:, 2]) == list(exp.val["sample_id"]), "val order mismatch"
ln34 = np.log(csv.iloc[:, 3].to_numpy())
outs = np.mean([np.load(ROOT / f"experiments/out_b4_0_f{f}/preds.npz")["out_val"] for f in FOLDS], axis=0)
table, _ = build_table(exp.train, cluster_labels(exp.train))
sig = 1 / (1 + np.exp(-outs[:, 1]))
r = np.clip(outs[:, 0], -2.5, 2.5)
ln4 = np.empty(len(r))
for i, c in enumerate(exp.val["category"]):
    v = table.values.get(c)
    ln4[i] = table.fallback if v is None else (next(iter(v.values())) if len(v) == 1 else (v[1] if sig[i] > .5 else v[0])) + r[i]
agree = np.abs(ln4 - ln34) < math.log(30)
ln = np.where(agree, W * ln4 + (1 - W) * ln34, ln34)
print(f"disagreements: {(~agree).sum()} of {len(ln)}; median |ln4-ln34| {np.median(np.abs(ln4-ln34)):.3f}")
pred = np.exp(ln)
print(write_submission(NAME, exp.val, pred))
print(f"min={pred.min():.2f} median={np.median(pred):.2f} max={pred.max():.2f}")
