# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Re-score saved B4.0 held-out outputs (fold 0) to confirm they match the training log (40.28)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from b0_b1 import smape
from b3 import build_table, cluster_labels
from frozen import Experiment

exp = Experiment()
d = np.load("experiments/out_b4_0_f0/preds.npz")
tr_idx, va_idx = exp.splits(0)[0]
assert (d["va_idx"] == va_idx).all()
T, Q = exp.train.iloc[tr_idx].reset_index(drop=True), exp.train.iloc[va_idx].reset_index(drop=True)
table, _ = build_table(T, cluster_labels(T))
out = d["out_heldout"]; r = np.clip(out[:, 0], -2.5, 2.5); p = 1 / (1 + np.exp(-out[:, 1]))
ln = np.empty(len(Q))
for i, c in enumerate(Q["category"]):
    v = table.values.get(c)
    ln[i] = table.fallback if v is None else (next(iter(v.values())) if len(v) == 1 else (v[1] if p[i] > .5 else v[0])) + r[i]
print(f"re-scored fold-0 held-out sMAPE from saved outputs: {smape(Q['volume'].to_numpy(), np.exp(ln)):.2f}")
