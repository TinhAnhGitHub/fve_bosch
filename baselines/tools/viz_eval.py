# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///
"""Render a submission zip's predictions next to the eval photos.   usage: viz_eval.py ZIP DATA_EVAL_DIR OUT_DIR

One PNG per object: the photo, then the predicted cloud seen along its three principal axes (colour = depth),
with its extent in mm. Also writes OUT_DIR/ply/<id>.ply (the submitted cloud) and OUT_DIR/index.csv.
The eval set has no ground truth, so these show only what the model predicts.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

zip_path, eval_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
(out / "ply").mkdir(parents=True, exist_ok=True)
photos = {p.parent.name: p for p in eval_dir.rglob("rgb_industrial.png")}
z = zipfile.ZipFile(zip_path)
rows = []
for name in sorted(n for n in z.namelist() if n.endswith(".ply")):
    sid = Path(name).stem
    raw = z.read(name)
    (out / "ply" / f"{sid}.ply").write_bytes(raw)
    text = raw.decode()
    pts = np.loadtxt(io.StringIO(text.split("end_header\n", 1)[1]))
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    y = c @ vt.T
    ext = y.max(0) - y.min(0)
    rms = float(np.sqrt((c ** 2).sum(1).mean()))
    p = photos.get(sid)
    group, cat = (p.parts[-4], p.parts[-3]) if p else ("?", "?")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    if p:
        ax[0].imshow(Image.open(p).resize((400, 400)))
    ax[0].set_title(f"{sid}\n{group} / {cat}", fontsize=9)
    lim = np.abs(y).max() * 1.05
    for a, (i, j), t in zip(ax[1:], [(0, 1), (0, 2), (1, 2)], ["long x middle axis", "long x short axis", "middle x short axis"]):
        a.scatter(y[::2, i], y[::2, j], s=0.4, c=y[::2, 3 - i - j], cmap="viridis")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_title(t, fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"prediction: {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm, RMS radius {rms:.1f} mm, {len(pts)} points", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{group}_{cat}_{sid}.png", dpi=60)
    plt.close(fig)
    rows.append((sid, group, cat, f"{ext[0]:.1f}", f"{ext[1]:.1f}", f"{ext[2]:.1f}", f"{rms:.1f}"))
with open(out / "index.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "group", "category", "extent1_mm", "extent2_mm", "extent3_mm", "rms_radius_mm"])
    w.writerows(rows)
print(len(rows), "objects ->", out)
