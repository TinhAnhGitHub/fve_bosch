# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3.5: decoding on top of B3.4 (k=2 kNN blend).

The predictive distribution in ln-space is a two-cluster mixture. Because sMAPE saturates at
200%, the sMAPE-optimal point never sits between the clusters: it is the cluster with enough
probability mass. So decoding reduces to two calibrated scalars, fitted out-of-fold:
  tau   - probability threshold for choosing the oversized cluster
  delta - global ln offset (corrects any systematic over/under-prediction)
Picked on seeds 0-2, re-checked on fresh seeds 3-5.

Run:  uv run experiments/b3_5_decode.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
import b3_4_knn as knn  # noqa: E402
from b0_b1 import smape, write_submission  # noqa: E402
from frozen import Config, Experiment  # noqa: E402

knn.KS = (2,)
K, W_R, W_P = 2, 0.25, 0.25
TAUS = np.round(np.arange(0.25, 0.751, 0.05), 2)
DELTAS = np.round(np.arange(-0.12, 0.121, 0.02), 2)


def grid(exp: Experiment, outs: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for tau, delta in itertools.product(TAUS, DELTAS):
        s = [smape(exp.y, np.exp(knn.decode(o, K, W_R, W_P, tau) + delta)) for o in outs]
        rows.append({"tau": tau, "delta": delta, "smape": np.mean(s), "std": np.std(s)})
    return pd.DataFrame(rows).sort_values("smape")


def main() -> None:
    exp = Experiment()
    sel = [knn.collect(exp, s) for s in (0, 1, 2)]
    g = grid(exp, sel)
    base = g[(g.tau == 0.5) & (g.delta == 0.0)].iloc[0]
    print(f"B3.4 decode (tau=0.5, delta=0) seeds 0-2: {base.smape:.2f}")
    print("\nby tau (delta=0):")
    print(g[g.delta == 0].sort_values("tau")[["tau", "smape"]].round(2).T.to_string(header=False))
    print("\nby delta (tau=0.5):")
    print(g[g.tau == 0.5].sort_values("delta")[["delta", "smape"]].round(2).T.to_string(header=False))
    best = g.iloc[0]
    print(f"\nbest: tau={best.tau} delta={best.delta} -> {best.smape:.2f} (gain {base.smape - best.smape:.2f})")

    # per-sample view of where the oversized decision matters (seed 0)
    o = sel[0]
    mixed = np.isfinite(o["p_large"])
    print(f"two-cluster objects: {mixed.sum()} of {len(mixed)}")

    fresh = [knn.collect(exp, s) for s in (3, 4, 5)]
    f0 = np.mean([smape(exp.y, np.exp(knn.decode(o, K, W_R, W_P, 0.5))) for o in fresh])
    f1 = np.mean([smape(exp.y, np.exp(knn.decode(o, K, W_R, W_P, best.tau) + best.delta)) for o in fresh])
    print(f"fresh seeds 3-5: B3.4 {f0:.2f} | B3.5 {f1:.2f} | gain {f0 - f1:.2f}")
    exp.log(Config(name=f"B3.5 decode tau={best.tau} delta={best.delta}", features=knn.CFG.features, n_pca=None,
                   C=0.01, tau=float(best.tau), r_clip=2.5),
            {"smape": best.smape, "smape_std": best["std"], "smape_fresh": f1, "smape_fresh_b34": f0})

    if f0 - f1 >= 0.5:
        X, X_val = exp.features(knn.CFG)
        Z_tr, Z_va = exp._project(X, X_val, None)
        Q = exp.val.assign(ln=np.nan)
        o = exp.fit_predict(exp.train, Q, Z_tr, Z_va, knn.CFG)
        for kk, parts in knn.knn_parts(exp.train, Q, Z_tr, Z_va).items():
            for name, v in parts.items():
                o[f"{name}_{kk}"] = v
        pred = np.exp(knn.decode(o, K, W_R, W_P, best.tau) + best.delta)
        print(write_submission("b3_5_decode", exp.val, pred))
    else:
        print("gain < 0.5 on fresh seeds: keep B3.4 decoding, no new submission")


if __name__ == "__main__":
    main()
