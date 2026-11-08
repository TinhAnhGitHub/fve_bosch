# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Metric geometry for the Food 3D Reconstruction grader.

The grader deletes rotation and translation and keeps scale. Mesh files in the
training set have no meaningful pose. `canonicalize` removes pose and leaves
millimetres in place, so a Chamfer loss can supervise shape and size.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

MIN_POINTS = 8192  # challenge rule: the grader rejects clouds with any other point count


def rms_radius(points: np.ndarray) -> float:
    """Size of a cloud in its own units: root mean square distance to the centroid."""
    x = np.asarray(points, np.float64)
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean()))


def hedge(unit: np.ndarray, p: float, sizes: tuple[float, float], q_min: float, frac: float) -> np.ndarray:
    """Millimetre cloud from a unit shape and a two-cluster size: normal (0) or oversized (1).

    When the less likely cluster has probability >= q_min, a `frac` share of the points
    moves to the other cluster's size and the rest stay at the likely size; the point
    count never changes (the grader needs exactly MIN_POINTS). Chamfer is two averages:
    wrong points cost only their share of the prediction-to-truth half, but if the other
    size was right they cover the truth, and the truth-to-prediction half would otherwise
    be the full size error. Otherwise one cloud at the expected size.
    """
    q = min(p, 1.0 - p)
    if frac <= 0.0 or q < q_min:
        return unit * ((1.0 - p) * sizes[0] + p * sizes[1])
    k = int(p >= 0.5)
    other = np.zeros(len(unit), dtype=bool)
    other[np.linspace(0, len(unit) - 1, max(1, round(frac * len(unit)))).astype(int)] = True
    return np.where(other[:, None], unit * sizes[1 - k], unit * sizes[k])


def canonicalize(points: np.ndarray) -> np.ndarray:
    """Center, then rotate onto principal axes. Do not rescale."""
    x = np.asarray(points, np.float64)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt.T
    # ponytail: sign from skewness. A near-symmetric axis can flip; the grader
    # treats that flip as a rotation. The sign only has to be deterministic.
    y *= np.where(np.mean(y**3, axis=0) < 0.0, -1.0, 1.0)
    return y


def cd_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    """Challenge score: mean squared nearest distance, both ways. Unit mm^2."""
    d = ((pred[:, None, :] - gt[None, :, :]) ** 2).sum(-1)
    return float(d.min(1).mean() + d.min(0).mean())


def _ply(points: np.ndarray) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in points)
    return "\n".join(lines) + "\n"


def pack(path: Path, clouds: dict[str, np.ndarray], min_points: int = MIN_POINTS) -> None:
    """Write preds_geometry.zip with geometry/{sample_id}.ply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample_id, pts in clouds.items():
            pts = np.asarray(pts, np.float64)
            if len(pts) != min_points:
                raise ValueError(f"{sample_id} has {len(pts)} points, the grader needs exactly {min_points}")
            zf.writestr(f"geometry/{sample_id}.ply", _ply(pts))
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        ids = {Path(n).stem for n in names}
        if ids != set(clouds) or len(ids) != len(names):
            raise ValueError("zip sample ids do not match")
        if not all(n.startswith("geometry/") and n.endswith(".ply") and n.count("/") == 1 for n in names):
            raise ValueError("zip paths must be geometry/{sample_id}.ply")


def _check() -> None:
    t = np.linspace(0.05, 1.0, 40)
    pts = np.stack([40.0 * t, 12.0 * t**2, 5.0 * np.sin(2.0 * np.pi * t)], 1)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = pts @ q.T + np.array([80.0, -15.0, 4.0])
    same = cd_l2(canonicalize(pts), canonicalize(moved))
    bigger = cd_l2(canonicalize(pts), canonicalize(pts * 2.0))
    assert same < 1e-6, same
    assert bigger > 10.0, bigger
    assert abs(rms_radius(pts * 3.0) - 3.0 * rms_radius(moved)) < 1e-9
    h = hedge(pts, 0.4, (1.0, 8.0), 0.2, 0.5)
    ratio = np.linalg.norm(h, axis=1) / np.linalg.norm(pts, axis=1)
    assert len(h) == len(pts) and np.isclose(ratio, 8.0).sum() == 20 and np.isclose(ratio, 1.0).sum() == 20
    assert np.allclose(hedge(pts, 0.1, (1.0, 8.0), 0.2, 0.5), pts * 1.7)
    pack(Path("/tmp/f3dr_check.zip"), {"a": pts}, min_points=len(pts))
    with zipfile.ZipFile("/tmp/f3dr_check.zip") as zf:
        body = zf.read("geometry/a.ply").decode()
    assert body.splitlines()[2] == f"element vertex {len(pts)}"
    print(f"check ok  pose_cd={same:.2e}  scale_cd={bigger:.1f}")


if __name__ == "__main__":
    _check()
