"""Which shape is better when the size is no longer the problem? RH-Mix5 against the LoRA on the 204 holdout objects.

The LoRA's headline number (24.2 at the true size) gives it the object's real size, and RH-Mix5's holdout score gives it nothing. Here
both get the same help: each cloud is aligned to the truth once (24 proper rotation starts, 20 ICP steps, as evaluate.py), then resized
about its own centre over a grid of factors, and the best factor is picked per object with knowledge of the truth, as report 11.2 does for
Atlas. Neither cloud is moved when it is resized, so the "as given" rows equal the usual scores.

  as given       RH-Mix5 as it predicted itself (its own size); the LoRA at the true RMS radius (its oracle)
  best size      the best factor for each object, from ×0.50 to ×1.60 (an oracle for both)

usage: shape_best_size.py HOLDOUT_DIR OUT_NPZ     (HOLDOUT_DIR = baselines/out/triposg/holdout with rhmix5_val.npz and lora_vs_truth/)
Needs numpy, scipy, trimesh, pillow and matplotlib (it reuses visualize_holdout.py and hedge_surrogate.py).
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hedge_surrogate import read_layers  # noqa: E402
from visualize_holdout import align  # noqa: E402

GRID = np.round(np.arange(0.50, 1.6001, 0.05), 2)


def cd(cloud, truth, truth_tree):
    return float(np.square(truth_tree.query(cloud)[0]).mean() + np.square(cKDTree(cloud).query(truth)[0]).mean())


def curve(cloud, truth):
    tree = cKDTree(truth)
    centre = cloud.mean(0)
    return np.array([cd(centre + s * (cloud - centre), truth, tree) for s in GRID])


def work(job):
    sid, truth, rh, lora = job
    _, aligned, truth_rh = align(rh, truth)  # RH-Mix5 aligned to the truth, both in the truth's principal-axis frame
    return sid, curve(aligned, truth_rh), curve(lora, truth)


def main():
    hold, out = Path(sys.argv[1]), Path(sys.argv[2])
    with np.load(hold / "rhmix5_val.npz", allow_pickle=False) as z:
        rh = dict(zip(z["ids"].tolist(), z["pts"].astype(np.float64)))
    rank = {r["id"]: int(r["rank"]) for r in csv.DictReader(open(hold / "lora_vs_truth" / "index.csv"))}
    jobs = []
    for sid in sorted(rank):
        truth, _, lora_true = read_layers(hold / "lora_vs_truth" / "ply" / f"{rank[sid]:03d}_{sid}.ply")
        jobs.append((sid, truth, rh[sid], lora_true))
    with ProcessPoolExecutor(14) as pool:
        results = []
        for k, r in enumerate(pool.map(work, jobs, chunksize=1), start=1):
            results.append(r)
            if k % 25 == 0:
                print(f"{k} of {len(jobs)}", flush=True)
    ids = np.array([r[0] for r in results])
    rh_curve, lora_curve = np.array([r[1] for r in results]), np.array([r[2] for r in results])
    np.savez(out, ids=ids, grid=GRID, rhmix5=rh_curve, lora=lora_curve)
    one = int(np.flatnonzero(GRID == 1.0)[0])
    # the "as given" column is the plain aligned CD-L2 of RH-Mix5 (same alignment code as evaluate.py), kept for hybrid_rhmix.py
    np.savez(out.parent / "cd_rhmix5.npz", ids=ids, cd=rh_curve[:, one])
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(ids), size=(2000, len(ids)))
    print(f"\n{len(ids)} holdout objects, CD-L2 (mm^2), same scorer, same truth")
    print("                              as given          best size for each object   (median best factor)")
    for name, c in (("RH-Mix5", rh_curve), ("LoRA", lora_curve)):
        best = c.min(1)
        print(f"  {name:8s} mean/median   {c[:, one].mean():6.2f} / {np.median(c[:, one]):5.2f}      {best.mean():6.2f} / {np.median(best):5.2f}"
              f"            x{np.median(GRID[c.argmin(1)]):.2f}   at the grid edge for {100 * np.mean((c.argmin(1) == 0) | (c.argmin(1) == len(GRID) - 1)):.0f}% of objects")
    for label, a, b in (("as given", rh_curve[:, one], lora_curve[:, one]), ("best size", rh_curve.min(1), lora_curve.min(1))):
        d = b - a
        low, high = np.percentile(d[pick].mean(1), [2.5, 97.5])
        print(f"  LoRA minus RH-Mix5, {label:9s}: {d.mean():+6.2f} (95% interval {low:+.2f} to {high:+.2f}); LoRA better on {int((d < 0).sum())} of {len(d)} objects")


if __name__ == "__main__":
    main()
