# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Mark training objects whose canonical frame is a mirror image (det -1). Writes out/atlas/train_mirror.npz. usage: mirror_marks.py root"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "baselines"))
from atlas import OUT, rows


def det(mesh_path):
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    rng = np.random.default_rng(0)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0)
    except Exception:
        v = np.asarray(mesh.vertices, np.float64); pts = v[rng.choice(len(v), 8192, replace=len(v) < 8192)]
    x = np.asarray(pts, np.float64); x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt.T
    sign = np.where(np.mean(y**3, axis=0) < 0.0, -1.0, 1.0)
    return float(np.linalg.det(vt.T @ np.diag(sign)))


with np.load(OUT / "train.npz") as z:
    ids = z["ids"].tolist()
where = {p.parent.name: p.parent / "mesh.ply" for p in rows("train")}
with ProcessPoolExecutor(8) as ex:
    d = np.array(list(ex.map(det, [where[i] for i in ids], chunksize=16)))
np.savez(OUT / "train_mirror.npz", ids=np.array(ids), mirror=d < 0)
print(f"{len(ids)} objects, mirrored {np.mean(d < 0):.3f}")
