# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Resize every cloud of a submission zip with the size model (size_knn.py); the shape is unchanged.

Rule tested out-of-fold on 2,042 objects (size_rescale_oof.py): new r = own r * (k * knn r / own r) ** gamma, about the
cloud's centre. own r = RMS radius of the submitted cloud, knn r = size_knn's eval prediction.
k was tuned on single-recipe out-of-fold clouds (v27); a mixed zip has a different typical radius, so k is multiplied by
  calib = median(own / knn r) of this zip on the 511 photos  /  median(own / knn r) of the out-of-fold clouds.

usage: rescale_zip.py SRC_ZIP SIZE_KNN.npz OOF_CLOUDS.npz STEP K GAMMA OUT_DIR
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geom import pack  # noqa: E402


def read_zip(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".ply"):
                body = z.read(name).decode().split("end_header\n", 1)[1]
                out[Path(name).stem] = np.loadtxt(io.StringIO(body), dtype=np.float64)[:, :3]
    return out


def rms(p: np.ndarray) -> float:
    return float(np.sqrt(np.square(p - p.mean(0)).sum(1).mean()))


def main() -> None:
    src, size_path, oof_path, step, k, gamma, out = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
                                                     int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
                                                     Path(sys.argv[7]))
    sz = np.load(size_path)
    s = list(sz["steps"]).index(step)
    knn_eval = dict(zip(sz["eval_ids"].tolist(), np.exp(sz["eval_ln_r"][s])))
    pos = {i: n for n, i in enumerate(sz["ids"].tolist())}
    with np.load(oof_path) as o:
        pts = o["pts"].astype(np.float32)
        oof_ids = o["ids"].tolist()
    oof_own = np.sqrt(np.square(pts - pts.mean(1, keepdims=True)).sum(-1).mean(-1))
    oof_knn = np.exp(sz["oof_ln_r"][s][[pos[i] for i in oof_ids]])

    clouds = read_zip(src)
    ids = sorted(clouds)
    own = np.array([rms(clouds[i]) for i in ids])
    knn = np.array([knn_eval[i] for i in ids])
    calib = np.median(own / knn) / np.median(oof_own / oof_knn)
    new = own * (k * calib * knn / own) ** gamma
    print(f"calib {calib:.3f} (zip own/knn median {np.median(own / knn):.3f}, oof {np.median(oof_own / oof_knn):.3f})")
    print(f"new/own: median {np.median(new / own):.3f}, 5th-95th pct {np.percentile(new / own, 5):.3f}-"
          f"{np.percentile(new / own, 95):.3f}")
    out.mkdir(parents=True, exist_ok=True)
    resized = {}
    for i, r_old, r_new in zip(ids, own, new):
        c = clouds[i].mean(0)
        resized[i] = c + (clouds[i] - c) * (r_new / r_old)
    pack(out / "preds_geometry.zip", resized)
    np.savez(out / "sizes.npz", ids=np.array(ids), own_mm=own, knn_mm=knn, new_mm=new, k=k, gamma=gamma, calib=calib)


if __name__ == "__main__":
    main()
