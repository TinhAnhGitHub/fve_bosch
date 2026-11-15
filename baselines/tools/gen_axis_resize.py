# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "trimesh"]
# ///
"""Side quest: can a generator's shape (SAM 3D, TripoSG, IDOL) win if we predict its length, width and height?

Each generator cloud is put in its principal frame and scaled along each axis to a target spread (std along that axis).
Variants per object, all scored with the holdout scorer (finetune/evaluate.py aligned_cd: 24 rotations + ICP, mm^2):
  true_axes            the true spreads (ceiling; the axis oracle)
  noise_<s>            true spreads x exp(s * N(0,1)) per axis, fixed seed: how much size error the shape tolerates
  pred_iso             same shape, isotropic, RMS radius = size_knn's out-of-fold radius
  pred_axes            spreads = size_knn --axes out-of-fold predictions
  pred_axes_hedge<h>   pred_axes at 3 sizes x exp(-h), 1, exp(+h), a third of the points each (index stride)
All predictions are out-of-fold (models never saw the object). Compare with our own cloud at its own size.

usage: gen_axis_resize.py DATA METADATA SIZE_KNN.npz SIZE_KNN_AXES.npz OUT_TXT NAME=CLOUDS.npz [NAME=CLOUDS.npz ...]
  NAME=own:CLOUDS.npz scores a cloud as it is (our model's reference line).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generators" / "triposg" / "finetune" / "analysis"))
from size_lookalike import aligned_cd, read_metadata, sample_truth  # noqa: E402

NOISE = (0.03, 0.05, 0.08, 0.12, 0.18)
HEDGE = (0.05, 0.10)


def principal(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cloud in its principal frame (columns longest first) and its spread along each axis."""
    c = p - p.mean(0)
    val, vec = np.linalg.eigh(c.T @ c / len(c))
    return c @ vec[:, ::-1], np.sqrt(np.maximum(val[::-1], 1e-12))


def job(args: tuple) -> dict[str, float]:
    data, row, unit_mm, cloud, own_only, ln_r, ln_sd, seed = args
    truth = sample_truth((data, row, unit_mm))
    score = aligned_cd()
    if own_only:
        return {"own": score(cloud.astype(np.float64) - cloud.mean(0), truth)}
    q, sd_q = principal(cloud.astype(np.float64))
    _, sd_t = principal(truth)
    rng = np.random.default_rng(seed)
    out = {"true_axes": score(q * (sd_t / sd_q), truth)}
    for s in NOISE:
        out[f"noise_{s:.2f}"] = score(q * (sd_t * np.exp(s * rng.standard_normal(3)) / sd_q), truth)
    out["pred_iso"] = score(q * (np.exp(ln_r) / np.sqrt((q ** 2).sum(1).mean())), truth)
    axes = q * (np.exp(ln_sd) / sd_q)
    out["pred_axes"] = score(axes, truth)
    for h in HEDGE:
        mixed = np.concatenate([axes[0::3] * np.exp(-h), axes[1::3], axes[2::3] * np.exp(h)])
        out[f"pred_axes_hedge{h:.2f}"] = score(mixed, truth)
    return out


def main() -> None:
    data, metadata, size_path, axes_path, out = sys.argv[1:5] + [Path(sys.argv[5])]
    rows, unit_mm = read_metadata(metadata)
    by_id = {r["id"]: r for r in rows}
    sz, ax = np.load(size_path), np.load(axes_path)
    step = 25
    r_pos = {i: n for n, i in enumerate(sz["ids"].tolist())}
    a_pos = {i: n for n, i in enumerate(ax["ids"].tolist())}
    ln_r = sz["oof_ln_r"][list(sz["steps"]).index(step)]
    ln_sd = ax["oof_ln_sd"][list(ax["steps"]).index(step)]  # (3, n), longest first
    lines = []
    for spec in sys.argv[6:]:
        name, path = spec.split("=", 1)
        own_only = path.startswith("own:")
        with np.load(path.removeprefix("own:")) as z:
            ok = np.isfinite(z["pts"]).all((1, 2))  # failed photos are NaN
            ids, pts = [str(i) for i in z["ids"][ok]], z["pts"][ok]
        tasks = [(data, by_id[i], unit_mm, pts[n], own_only, ln_r[r_pos[i]], ln_sd[:, a_pos[i]], n) for n, i in enumerate(ids)]
        with ProcessPoolExecutor(14) as pool:
            res = list(pool.map(job, tasks, chunksize=1))
        lines.append(f"{name} (n={len(ids)})")
        for key in res[0]:
            v = np.array([r[key] for r in res])
            lines.append(f"  {key:22s} mean {np.nanmean(v):7.2f}  median {np.nanmedian(v):7.2f}")
        print("\n".join(lines[-len(res[0]) - 1:]), flush=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
