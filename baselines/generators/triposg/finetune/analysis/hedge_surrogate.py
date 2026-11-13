"""Fast stand-in for the CD scorer, to try many size and hedge variants of the LoRA shape in seconds.

The real scorer (evaluate.py::aligned_cd) re-aligns every cloud with 24 starts x 20 ICP steps, about 25 s per object. Here each cloud
stays in the pose where it was aligned to the truth at the TRUE size (the saved lora_vs_truth/ply files, third layer) and is only
rescaled about its centre. It is only an approximation, not a bound: the real scorer's ICP can end in a better or a worse pose at the
wrong size. The first lines it prints compare it with the real scores on the arms that have them (2026-11-13 holdout run: true size
25.5 against 24.2, the size head's size 75.0 against 79.3, and the look-alike size 61.4 against 60.9), so its error, about 5% of the
mean, is visible before it is used.

usage: hedge_surrogate.py HOLDOUT_DIR SIZE_DIR     (HOLDOUT_DIR = baselines/out/triposg/holdout, SIZE_DIR = .../size_lookalike)
Needs numpy and scipy.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_layers(path):
    """The three layers (truth, output at the size head's size, output at the true size) of a visualize_holdout.py PLY file."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    end = raw.index(marker) + len(marker)
    n = int(next(line for line in raw[:end].decode("ascii").splitlines() if line.startswith("element vertex")).split()[2])
    v = np.frombuffer(raw, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n, offset=end)
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    k = n // 3
    return pts[:k], pts[k:2 * k], pts[2 * k:]


def cd(cloud, truth, truth_tree):
    return float(np.square(truth_tree.query(cloud)[0]).mean() + np.square(cKDTree(cloud).query(truth)[0]).mean())


def hedged(base, factors, weights, order):
    """Copies of `base` (the cloud at the true size, centred) scaled by `factors`; copy j gets weights[j] of the points."""
    counts = np.round(np.array(weights) / np.sum(weights) * len(base)).astype(int)
    counts[-1] = len(base) - counts[:-1].sum()
    parts, start = [], 0
    for factor, count in zip(factors, counts):
        parts.append(base[order[start:start + count]] * factor)
        start += count
    return np.concatenate(parts)


def main():
    hold, size_dir = Path(sys.argv[1]), Path(sys.argv[2])
    sizes = np.load(size_dir / "holdout_sizes.npz")
    scores = np.load(hold / "lora_scores.npz")
    ids = sizes["ids"].tolist()
    assert ids == scores["ids"].tolist()
    rank = {r["id"]: int(r["rank"]) for r in csv.DictReader(open(hold / "lora_vs_truth" / "index.csv"))}
    true, head, blend = sizes["true"], sizes["head"], sizes["blend"]
    rng = np.random.default_rng(0)
    items = []
    for n, sid in enumerate(ids):
        truth, _, at_true = read_layers(hold / "lora_vs_truth" / "ply" / f"{rank[sid]:03d}_{sid}.ply")
        centre = at_true.mean(0)
        items.append((truth, cKDTree(truth), at_true - centre, rng.permutation(len(at_true))))

    def run(radius, factors=(1.0,), weights=(1.0,)):
        """CD per object for the given predicted radii (mm array) and hedge copies (relative scales, point shares)."""
        out = np.empty(len(ids))
        for n, (truth, tree, base, order) in enumerate(items):
            scale = radius[n] / true[n]
            out[n] = cd(hedged(base, [scale * f for f in factors], weights, order), truth, tree)
        return out

    def show(name, values, real=None):
        extra = "" if real is None else f"   real scorer {real.mean():7.2f} / {np.median(real):6.2f}"
        print(f"{name:46s} mean {values.mean():7.2f}  median {np.median(values):6.2f}{extra}", flush=True)

    print("stand-in against the real scorer (mean / median CD-L2, 204 holdout objects):")
    show("LoRA at the true size", run(true), scores["oracle"])
    show("LoRA at the size head's size", run(head), scores["metric"])
    single = run(blend)
    show("LoRA at the look-alike blend (one size)", single)
    show("LoRA at the type median (one size)", run(sizes["cat_median"]))
    print("\nsize hedge on the look-alike blend: three copies at (1-d, 1, 1+d), a third of the points each")
    for d in (0.06, 0.10, 0.15, 0.20, 0.30):
        show(f"  d = {d:.2f}", run(blend, (1 - d, 1.0, 1 + d), (1, 1, 1)))
    print("five copies at (1-2d, 1-d, 1, 1+d, 1+2d), weights 1:2:3:2:1")
    for d in (0.05, 0.08, 0.12, 0.16):
        show(f"  d = {d:.2f}", run(blend, (1 - 2 * d, 1 - d, 1.0, 1 + d, 1 + 2 * d), (1, 2, 3, 2, 1)))
    print("\nhedge width set per object from how close and how much in agreement its look-alikes are:")
    best, gap = sizes["best_similarity"], np.nan_to_num(sizes["neighbour_gap"], nan=0.15)
    far = (best < np.quantile(best, 1 / 3)).astype(float)
    for base_d, extra_gap, extra_far in ((0.06, 0.5, 0.06), (0.08, 0.8, 0.08), (0.10, 1.0, 0.10)):
        widths = np.clip(base_d + extra_gap * gap + extra_far * far, 0.04, 0.45)
        out = np.empty(len(ids))
        for n, (truth, tree, base, order) in enumerate(items):
            d = widths[n]
            out[n] = cd(hedged(base, [blend[n] / true[n] * f for f in (1 - d, 1.0, 1 + d)], (1, 1, 1), order), truth, tree)
        show(f"  d = {base_d:.2f} + {extra_gap:.1f} x gap + {extra_far:.2f} if far", out)
    best_single = min((run(blend * f).mean(), f) for f in (0.94, 0.97, 1.0, 1.03, 1.06, 1.10))
    print(f"\nbest constant factor on the blend radius (a check that the blend is centred): x{best_single[1]:.2f} -> {best_single[0]:.2f}")


if __name__ == "__main__":
    main()
