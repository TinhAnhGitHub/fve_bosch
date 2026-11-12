"""Draw the LoRA model's public-test clouds. The public test has no ground truth, so there is no score and no comparison: each
object is the photo plus the model output seen from three angles, with the size our size head chose.

usage: visualize_public.py SHARD_DIR DATA_DIR OUT_DIR
  SHARD_DIR  folder with the lora_eval_s*.npz shards that finetune/predict.py --shard writes, or one merged lora_eval.npz
  DATA_DIR   data/f3d (the public photos are eval/<group>/<category>/<id>/rgb_industrial.png)
  OUT_DIR    created: objects/<id>.png, ply/<id>.ply, index.csv and index.html

Objects that already have an image are skipped, so rerun it when more shards arrive. Each cloud is shown in its own principal-axis
frame (longest axis along x, shortest along z), centred, in mm: the size is our size head's guess for the radius, and nobody has
checked it against the real object. ply/<id>.ply is the same cloud (binary PLY, orange, mm) for MeshLab, CloudCompare or Blender.
Needs numpy, scipy, trimesh, pillow and matplotlib (it reuses visualize_holdout.py).
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_holdout import RGB, MODEL, pca, write_ply  # noqa: E402

VIEWS = [("Model output, oblique view", 28, -55), ("Top view (longest and middle axes)", 90, -90),
         ("Side view (longest and shortest axes)", 0, -90)]
CSS = """
body{font-family:sans-serif;max-width:1150px;margin:auto;padding:0 12px;color:#222}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #ccc;padding:3px 8px;text-align:right;font-size:14px}
td:nth-child(1),td:nth-child(2),th:nth-child(1),th:nth-child(2){text-align:left}
th{background:#f1f1f1;cursor:pointer;position:sticky;top:0}
.note{background:#f6f8fa;border-left:4px solid #3b6fb6;padding:6px 12px;margin:10px 0;font-size:14px}
.obj{margin:18px 0}
.oh{background:#eef3f9;padding:6px 10px;font-size:15px;border:1px solid #c9d6e6;border-bottom:none}
img{width:100%;border:1px solid #ddd;display:block}
"""
JS = """
function sortTable(n, numeric) {
  const tb = document.getElementById('t').tBodies[0];
  const rows = Array.from(tb.rows);
  const dir = (tb.dataset.col == n && tb.dataset.dir == 'asc') ? 'desc' : 'asc';
  rows.sort((a, b) => {
    const x = a.cells[n].dataset.v, y = b.cells[n].dataset.v;
    const c = numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y);
    return dir == 'asc' ? c : -c;
  });
  rows.forEach(r => tb.appendChild(r));
  tb.dataset.col = n; tb.dataset.dir = dir;
}
"""


def load_clouds(shard_dir):
    merged = shard_dir / "lora_eval.npz"
    files = [merged] if merged.exists() else sorted(shard_dir.glob("lora_eval_s*.npz"))
    if not files:
        raise SystemExit(f"no lora_eval.npz or lora_eval_s*.npz in {shard_dir}")
    clouds = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        for i, sid in enumerate(z["ids"].tolist()):
            clouds[sid] = (z["pts"][i].astype(np.float64), float(z["radius_mm"][i]))
    return clouds, files


def object_image(sid, category, photo, cloud, radius, path):
    lim = 1.1 * np.abs(cloud).max()
    fig = plt.figure(figsize=(13.5, 3.85))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(Image.open(photo).convert("RGB").resize((300, 300)))
    ax.axis("off")
    ax.text(0.5, -0.04, f"{sid} ({category})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    for j, (title, elev, azim) in enumerate(VIEWS, start=2):
        ax = fig.add_subplot(1, 4, j, projection="3d")
        ax.scatter(*cloud.T, s=0.7, c=MODEL, alpha=0.6, depthshade=False, linewidths=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.text2D(0.5, 0.0, f"our size guess: radius {radius:.0f} mm" if j == 2 else f"plot box +-{lim:.0f} mm",
                  transform=ax.transAxes, ha="center", va="top", fontsize=9)
        fig.text(0.01 + 0.98 / 4 * (j - 0.5), 0.995, title, ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.01 + 0.98 / 8, 0.995, "Photo (the model's input)", ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01, "Public test: no ground truth exists, so there is no score. The size is our size head's guess.",
             ha="center", va="bottom", fontsize=9, style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.14, wspace=0.02)
    fig.savefig(path, dpi=75)
    plt.close(fig)


def write_html(out, rows, total):
    sizes = [r["radius"] for r in rows]
    heads = [("Object", 0), ("Category", 0), ("Our size guess: radius (mm)", 1)]
    table = "".join(
        f'<tr><td data-v="{r["id"]}"><a href="#{r["id"]}">{r["id"]}</a></td><td data-v="{r["category"]}">{r["category"]}</td>'
        f'<td data-v="{r["radius"]:.2f}">{r["radius"]:.1f}</td></tr>' for r in rows)
    blocks = "\n".join(
        f'<div class="obj" id="{r["id"]}"><div class="oh"><b>{r["id"]}</b> ({r["category"]}) &nbsp;|&nbsp; our size guess: radius '
        f'{r["radius"]:.1f} mm</div><img loading="lazy" src="objects/{r["id"]}.png"></div>' for r in rows)
    head = "".join(f'<th onclick="sortTable({i}, {num})">{h}</th>' for i, (h, num) in enumerate(heads))
    html = f"""<html><head><meta charset="utf-8"><title>LoRA public-test outputs</title><style>{CSS}</style></head>
<body><h2>LoRA (epoch 4) on the public test: {len(rows)} of {total} objects so far</h2>
<div class="note"><b>There is no ground truth for the public test, so there is no score and no comparison here.</b> Each object is the
photo and the model's output seen from three angles. The size is our size head's guess for the object's radius (RMS distance of
the points from their centre). On the 204 holdout objects that guess was within 10% for 41% of the objects and off by more than
30% for 18%; the shapes at the true size scored 24 against 79 at the guessed size, so the size is the part to doubt.</div>
<p>Median guessed radius {statistics.median(sizes):.1f} mm, smallest {min(sizes):.1f}, largest {max(sizes):.1f}.
<b>To find one object:</b> use the browser's find box (Ctrl+F) with its full id (written <code>omni_&lt;food&gt;_&lt;number&gt;</code> or
<code>mf3d_&lt;food&gt;_&lt;number&gt;</code>). The first match is its row in the table; press Enter once more to land on its heading
and image further down. Click a column heading to sort. 3D files: open any file in the <code>ply</code> folder with MeshLab,
CloudCompare or Blender (orange, mm).</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>{table}</tbody></table>
{blocks}<script>{JS}</script></body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


def main():
    shard_dir, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    clouds, files = load_clouds(shard_dir)
    photos = {p.parent.name: p for p in (data / "eval").glob("*/*/*/rgb_industrial.png")}
    missing = sorted(set(clouds) - set(photos))
    if missing:
        raise SystemExit(f"no public photo for {missing[:3]} ...")
    (out / "objects").mkdir(parents=True, exist_ok=True)
    (out / "ply").mkdir(exist_ok=True)
    print(f"{len(clouds)} clouds from {[f.name for f in files]}", flush=True)
    rows = []
    for k, sid in enumerate(sorted(clouds), start=1):
        pts, radius = clouds[sid]
        category = photos[sid].parent.parent.name
        cloud = pca(pts)  # centred, principal axes, a proper rotation
        png = out / "objects" / f"{sid}.png"
        if not png.exists():
            object_image(sid, category, photos[sid], cloud, radius, png)
            write_ply(out / "ply" / f"{sid}.ply", [(cloud, RGB["guess"])])
        rows.append(dict(id=sid, category=category, radius=radius))
        if k % 25 == 0 or k == len(clouds):
            print(f"{k} of {len(clouds)}", flush=True)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "size_guess_radius_mm"])
        w.writerows([[r["id"], r["category"], f"{r['radius']:.2f}"] for r in rows])
    write_html(out, rows, 511)
    print("done:", out)


if __name__ == "__main__":
    main()
