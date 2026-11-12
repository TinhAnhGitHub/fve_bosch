"""Which labels did the TripoSG size head train on? Compare the radius of every train/val mesh in the TripoSG manifest
(sampled from the CURRENT meshes, like prepare.py does) with the legacy size cache baselines/out/scale/scale.npz.

The data-quality audit (docs/F3DR_data_quality_review.md) found scale.npz holds radii about 10x the current mesh radius for
265 objects and still flags 174 of them oversized. If the current meshes are the corrected ones, the TripoSG size head never saw
an oversized label, and there is nothing to drop.

usage: stale_check.py REPO_ROOT HOLDOUT_DIR     (HOLDOUT_DIR has metadata.json; needs numpy and trimesh)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

root, hold = Path(sys.argv[1]), Path(sys.argv[2])
DATA = root.parent / "data" / "f3d"
meta = json.load(open(hold / "metadata.json"))


def radius(row):
    mesh = trimesh.load(DATA / row["mesh"], force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, 8192, seed=42)
    pts = pts * meta["mesh_unit_mm"]
    return float(np.sqrt(np.square(pts - pts.mean(0)).sum(1).mean()))


if __name__ == "__main__":
    rows = [r for r in meta["rows"] if r["split"] in ("train", "val")]
    with ProcessPoolExecutor(14) as ex:
        cur = dict(zip([r["id"] for r in rows], ex.map(radius, rows, chunksize=8)))
    z = np.load(root / "baselines/out/scale/scale.npz", allow_pickle=False)
    legacy = dict(zip(z["ids"].tolist(), z["radius"].tolist()))
    flag = dict(zip(z["ids"].tolist(), z["u"].tolist()))
    ids = [i for i in cur if i in legacy]
    ratio = np.array([legacy[i] / cur[i] for i in ids])
    print(f"{len(cur)} train/val meshes in the TripoSG manifest, {len(ids)} also in scale.npz")
    print(f"legacy radius / current radius: median {np.median(ratio):.2f}; objects with ratio > 5: {(ratio > 5).sum()}")
    big = [i for i in ids if flag[i] == 1]
    print(f"objects flagged oversized in scale.npz and present here: {len(big)}; their current radius median "
          f"{np.median([cur[i] for i in big]):.1f} mm, max {max(cur[i] for i in big):.1f} mm; legacy radius median "
          f"{np.median([legacy[i] for i in big]):.1f} mm")
    r = np.array(list(cur.values()))
    print(f"all manifest radii: median {np.median(r):.1f} mm, 99th percentile {np.percentile(r, 99):.1f} mm, max {r.max():.1f} mm")
    val = [r["id"] for r in rows if r["split"] == "val"]
    v = np.array([legacy[i] / cur[i] for i in val if i in legacy])
    print(f"holdout objects in scale.npz: {len(v)}; legacy/current ratio > 5 on {(v > 5).sum()} of them")
