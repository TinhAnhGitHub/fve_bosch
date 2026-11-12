"""Local, parallel version of finetune/evaluate.py (same aligned_cd, same truth sampling) for the 4 holdout arms.
usage: score_local.py DATA_DIR  (run from this folder; writes <arm>_scores.npz and prints the table)"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, "/mnt/c/tinhanh/project/ai_challenge/fve_f3dr/baselines/generators/triposg/finetune")
from evaluate import aligned_cd  # noqa: E402
from common import metric_points, read_json  # noqa: E402

DATA = Path(sys.argv[1])
meta = read_json("metadata.json")
rows = {r["id"]: r for r in meta["rows"] if r["split"] == "val"}


def truth_of(sid):
    mesh = trimesh.load(DATA / rows[sid]["mesh"], force="mesh", process=False)
    t, _ = trimesh.sample.sample_surface(mesh, 8192, seed=42)
    t = t * meta["mesh_unit_mm"]
    return t - t.mean(0)


def score(job):
    sid, pred = job
    truth = truth_of(sid)
    radius = np.sqrt(np.square(truth).sum(1).mean())
    own = np.sqrt(np.square(pred - pred.mean(0)).sum(1).mean())
    return aligned_cd(pred, truth), aligned_cd(metric_points(pred, radius), truth), (own - radius) ** 2


if __name__ == "__main__":
    for arm in ("pretrained", "lora", "full", "fulllast"):
        z = np.load(f"{arm}_val.npz", allow_pickle=False)
        ids, pts = z["ids"].tolist(), z["pts"].astype(np.float64)
        assert set(ids) == set(rows) and len(ids) == 204 and pts.shape == (204, 8192, 3)
        with ProcessPoolExecutor(14) as ex:
            res = np.array(list(ex.map(score, zip(ids, pts), chunksize=4)))
        np.savez(f"{arm}_scores.npz", ids=np.array(ids), metric=res[:, 0], oracle=res[:, 1], radius_mse=res[:, 2])
        print(f"{arm:10s} metric mean {res[:, 0].mean():7.2f} median {np.median(res[:, 0]):7.2f} | "
              f"oracle-size mean {res[:, 1].mean():7.2f} median {np.median(res[:, 1]):7.2f} | radius MSE {res[:, 2].mean():7.1f}",
              flush=True)
