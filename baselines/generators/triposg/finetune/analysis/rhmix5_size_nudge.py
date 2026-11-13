"""Could the look-alike size correct RH-Mix5's own size? A quick look on the 204 holdout objects, no new alignment.

RH-Mix5 scores 27.4 as predicted and 16.5 when each object gets its best size (shape_best_size.py), so up to 11 points sit in its size.
RH-Mix5 hedges: its clouds are small (RMS radius about 0.77 of the true one) and the best resize is about x0.95, so the look-alike radius
cannot be used directly. What is tried here is a nudge: resize RH-Mix5's cloud by f = clip(exp(gamma * (log q - c)), lo, hi), with
q = look-alike radius / RMS radius of RH-Mix5's cloud, c the typical value of log q, and gamma between 0 (no change) and 1, optionally
only for objects whose two look-alikes agree. The score at any factor comes from the resize curves of shape_best_size.py (each cloud
aligned once, then resized), so these numbers are a stand-in for the real scorer, about 5% off.

gamma and c are fitted on 9/10 of the holdout objects and judged on the other tenth (10 folds, repeated with 20 shuffles), so the table
is out of sample within the holdout, but with 204 objects it is still noisy. The in-sample line is for orientation only.

usage: rhmix5_size_nudge.py SIZE_DIR HOLDOUT_DIR      (needs numpy)
"""
import sys
from pathlib import Path

import numpy as np

GAMMAS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)
LIMITS = ((0.85, 1.15), (0.75, 1.3))


def main():
    size_dir, hold = Path(sys.argv[1]), Path(sys.argv[2])
    curves = np.load(size_dir / "shape_best_size.npz")
    sizes = np.load(size_dir / "holdout_sizes.npz")
    with np.load(hold / "rhmix5_val.npz", allow_pickle=False) as z:
        pts = dict(zip(z["ids"].tolist(), z["pts"].astype(np.float64)))
    ids = curves["ids"].tolist()
    grid, rh = curves["grid"], curves["rhmix5"]
    where = {i: n for n, i in enumerate(sizes["ids"].tolist())}
    order = [where[i] for i in ids]
    look = sizes["blend"][order]
    gap = np.nan_to_num(sizes["neighbour_gap"][order], nan=9.9)
    atlas = np.array([np.sqrt(np.square(pts[i] - pts[i].mean(0)).sum(1).mean()) for i in ids])
    logq = np.log(look / atlas)
    best = grid[rh.argmin(1)]
    one = int(np.flatnonzero(grid == 1.0)[0])

    def cd_at(factor):
        return np.array([np.interp(f, grid, rh[n]) for n, f in enumerate(factor)])

    print(f"{len(ids)} holdout objects. RH-Mix5 as predicted {rh[:, one].mean():.2f}, with the best size for each object {rh.min(1).mean():.2f}")
    print(f"RMS radius of RH-Mix5's cloud / true radius: median {np.median(atlas / sizes['true'][order]):.2f}; best resize factor median {np.median(best):.2f}")
    for name, m in (("all objects", np.ones(len(ids), bool)), ("look-alikes agree (gap < 0.056)", gap < 0.056),
                    ("agree or partly (gap < 0.174)", gap < 0.174), ("disagree (gap >= 0.174)", gap >= 0.174)):
        r = np.corrcoef(logq[m], np.log(best[m]))[0, 1]
        print(f"  correlation of log(look-alike radius / RH-Mix5 radius) with the best resize, {name:32s} n={int(m.sum()):3d}: {r:+.2f}")

    def rule(gamma, c, lo, hi, use):
        factor = np.where(use, np.clip(np.exp(gamma * (logq - c)), lo, hi), 1.0)
        return cd_at(factor)

    print("\nresize RH-Mix5 by the nudge, in sample (c = median of log q), mean CD-L2:")
    for gate_name, gate in (("every object", np.ones(len(ids), bool)), ("only where the look-alikes agree or partly agree", gap < 0.174),
                            ("only where they agree", gap < 0.056)):
        for lo, hi in LIMITS:
            row = "  ".join(f"g={g:.2f}: {rule(g, np.median(logq[gate]), lo, hi, gate).mean():5.2f}" for g in GAMMAS)
            print(f"  {gate_name:48s} limits x{lo}..x{hi}   {row}")

    print("\nout of sample (10 folds x 20 shuffles; gamma and c fitted on 9/10, judged on 1/10), mean CD-L2 of RH-Mix5 resized:")
    rng = np.random.default_rng(0)
    for gate_name, gate in (("every object", np.ones(len(ids), bool)), ("only where the look-alikes agree or partly agree", gap < 0.174)):
        for lo, hi in LIMITS:
            totals = []
            for _ in range(20):
                folds = rng.permutation(len(ids)) % 10
                cd = np.empty(len(ids))
                for f in range(10):
                    train, test = folds != f, folds == f
                    c = np.median(logq[train & gate]) if (train & gate).any() else 0.0
                    gamma = min(GAMMAS, key=lambda g: rule(g, c, lo, hi, gate)[train].mean())
                    cd[test] = rule(gamma, c, lo, hi, gate)[test]
                totals.append(cd.mean())
            print(f"  {gate_name:48s} limits x{lo}..x{hi}   {np.mean(totals):5.2f}  (RH-Mix5 untouched {rh[:, one].mean():.2f})")


if __name__ == "__main__":
    main()
