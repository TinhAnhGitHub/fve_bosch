# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "trimesh"]
# ///
"""Scale-free shape descriptors from each train mesh (training-only auxiliary targets for B4.1/B4.2).

Per object: ln convexity (V / V_hull), ln compactness (S^1.5 / V), ln PCA eigenvalue ratios l2/l1 and l3/l1 of the vertices.
Saved to features/mesh_targets.npz as arrays aligned with Experiment().train row order, plus sample ids.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines"))
from b0_b1 import DATA
from frozen import Experiment


def descr(path: str) -> list[float]:
    m = trimesh.load(path, process=False)
    v = abs(float(m.volume)); s = float(m.area)
    hull = ConvexHull(m.vertices).volume
    ev = np.sort(np.linalg.eigvalsh(np.cov((m.vertices - m.vertices.mean(0)).T)))[::-1]
    return [np.log(v / hull), np.log(s ** 1.5 / v), np.log(ev[1] / ev[0]), np.log(ev[2] / ev[0])]


if __name__ == "__main__":
    exp = Experiment()
    paths = [str(DATA / "train" / r.group / r.category / r.sample_id / "mesh.ply") for r in exp.train.itertuples()]
    with ProcessPoolExecutor(6) as ex:
        X = np.array(list(ex.map(descr, paths, chunksize=8)), dtype=np.float64)
    print("shape", X.shape, "finite", np.isfinite(X).all(), "\nmean", X.mean(0).round(3), "\nstd", X.std(0).round(3))
    np.savez(ROOT / "features" / "mesh_targets.npz", X=X, ids=exp.train["sample_id"].to_numpy(),
             names=np.array(["ln_convexity", "ln_compactness", "ln_pca21", "ln_pca31"]))
