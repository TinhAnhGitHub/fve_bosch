# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Stratified 5-fold split of the 2,042 training objects. usage: uv run -q baselines/tools/make_folds.py

Categories are visited in a seeded random order; inside each category the objects are sorted by
RMS radius and dealt round-robin from a global pointer that carries over between categories. So
every fold gets the small and the large members of each category, singletons and pairs spread
over the folds, and the folds hold 408 or 409 objects. No object families are merged: no true
duplicates were found (B3.1; the perceptual-hash pairs are different objects), so `family` is
the id until a near-duplicate review says otherwise.

Writes baselines/folds/f3dr_k5_v1.csv and a balance summary next to it.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
from atlas import OUT, holdout, rows  # noqa: E402

K, SEED = 5, 0
DEST = ROOT / "baselines" / "folds" / f"f3dr_k{K}_v1.csv"


def main() -> None:
    with np.load(OUT / "train.npz", allow_pickle=False) as z:
        ids, cats = z["ids"].tolist(), z["cats"].tolist()
        radius = np.sqrt(np.square(z["pts"]).sum(-1).mean(-1))
    paths = rows("train")
    if [p.parent.name for p in paths] != ids:
        raise SystemExit("train.npz does not match the training images")
    group = [p.parent.parent.parent.name for p in paths]
    source = [i.split("_", 1)[0] for i in ids]
    old = np.zeros(len(ids), bool)
    old[holdout(len(ids))[0]] = True

    rng = np.random.default_rng(SEED)
    fold = np.full(len(ids), -1)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)
    pointer = 0
    for c in rng.permutation(sorted(by_cat)):
        for j, i in enumerate(sorted(by_cat[c], key=lambda i: radius[i])):
            fold[i] = (pointer + j) % K
        pointer += len(by_cat[c])

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "fold", "family", "category", "group", "source", "radius", "old_holdout"])
        for i in range(len(ids)):
            w.writerow([ids[i], int(fold[i]), ids[i], cats[i], group[i], source[i], f"{radius[i]:.3f}", int(old[i])])

    lines = [f"{len(ids)} objects, {len(by_cat)} categories, {K} folds, seed {SEED} -> {DEST.name}"]
    lines.append("fold sizes: " + " ".join(str(int((fold == k).sum())) for k in range(K)))
    quint = np.searchsorted(np.quantile(radius, [0.2, 0.4, 0.6, 0.8]), radius)
    big = radius >= np.quantile(radius, 0.6)
    worst = 0.0
    for name, values in (("group", group), ("source", source), ("radius quintile", [f"q{q + 1}" for q in quint]), ("largest 40%", ["yes" if b else "no" for b in big])):
        lines.append(f"\n{name}: overall share, then per-fold share (points)")
        values = np.array(values)
        for v in sorted(set(values.tolist())):
            all_share = 100 * (values == v).mean()
            per = [100 * (values[fold == k] == v).mean() for k in range(K)]
            worst = max(worst, max(abs(p - all_share) for p in per))
            lines.append(f"  {v:12s} {all_share:5.1f} | " + " ".join(f"{p:5.1f}" for p in per))
    lines.append(f"\nlargest per-fold deviation from the overall share: {worst:.1f} points")
    lines.append("per-fold median radius: " + " ".join(f"{np.median(radius[fold == k]):.1f}" for k in range(K)))
    single = sorted(c for c, v in by_cat.items() if len(v) == 1)
    lines.append(f"singleton categories ({len(single)}, unseen while their fold is validated): {', '.join(single)}")
    unseen = [sum(1 for c, v in by_cat.items() if all(fold[i] == k for i in v)) for k in range(K)]
    lines.append("categories with no training object, per fold: " + " ".join(map(str, unseen)))
    lines.append("old 204 holdout objects per fold: " + " ".join(str(int(old[fold == k].sum())) for k in range(K)))
    text = "\n".join(lines)
    print(text)
    DEST.with_suffix(".summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
