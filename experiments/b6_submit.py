# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B6 submission: DINOv3-7B + query decoder, mean of fold heads' out_val, decoded with the full-train category bases.
usage: b6_submit.py <dir with f*/preds.npz> [folds=0,1] [name=b6_7b_decoder] [shrink C=0]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
from b0_b1 import write_submission
from frozen import Experiment

src = Path(sys.argv[1])
FOLDS = tuple(int(f) for f in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1)
NAME = sys.argv[3] if len(sys.argv) > 3 else "b6_7b_decoder"
exp = Experiment()
rows = pd.read_csv(ROOT / "kaggle_pkg/b42_prep/rows.csv")
assert list(rows[rows.split == "val"].sample_id) == list(exp.val["sample_id"]), "val order mismatch"
p = np.load(ROOT / "kaggle_pkg/b42_prep/fold0.npz")
out = np.mean([np.load(src / f"f{f}/preds.npz")["out_val"] for f in FOLDS], axis=0)
large = (1 / (1 + np.exp(-out[:, 1])) > .5) & np.isfinite(p["val_base1"])
# shrink r toward the category base by n/(n+C), n = training objects in the category: the head never sees the
# category, so its residual is unreliable for rare ones (OOF folds 0/1: 42.04/38.75 -> 41.12/37.65 at C=4)
C = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
n = exp.val["category"].map(exp.train["category"].value_counts()).fillna(0).to_numpy()
pred = np.exp(np.where(large, p["val_base1"], p["val_base0"]) + np.clip(out[:, 0], -2.5, 2.5) * n / (n + C))
print(write_submission(NAME, exp.val, pred))
print(f"n={len(pred)} oversized={large.sum()} min={pred.min():.2f} median={np.median(pred):.2f} max={pred.max():.2f}")
