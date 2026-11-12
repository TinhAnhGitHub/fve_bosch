"""Draw the holdout clouds of one arm next to the ground truth, best score first.

usage: visualize_holdout.py HOLDOUT_DIR DATA_DIR [ARM] [COUNT|all] [redraw]
  redraw       draw the images and index.html again from the saved ply/ files and index.csv, with no new alignment (a few
               minutes instead of a quarter of an hour)
  HOLDOUT_DIR  baselines/out/triposg/holdout (has metadata.json, <ARM>_val.npz and <ARM>_scores.npz)
  DATA_DIR     data/f3d
  ARM          lora (default), pretrained, full or fulllast
  COUNT        objects to draw: all (default), or a number picked evenly by rank of the score at the predicted size

Writes <HOLDOUT_DIR>/<ARM>_vs_truth/objects/<rank>_<id>.png (one image per object, best score first), index.csv (one line per
object) and index.html (a sortable table plus every object's image under a text heading with its id, so the browser's find box
reaches it). Columns of an image: photo | ground truth | model output at the size head's size over the truth | model output at
the true size over the truth. Output and truth are aligned like evaluate.py: 24 proper rotation starts, 20 rigid ICP steps, scale
never fitted. The recomputed scores are checked against the saved <ARM>_scores.npz.

Also writes, next to the images:
  ply/<rank>_<id>.ply   one binary PLY per object with three coloured layers in the truth's principal-axis frame, in mm:
                        blue = ground truth, orange = model output at the size head's size, green = model output at the true size.
                        Open it in MeshLab, CloudCompare or Blender.
and, in HOLDOUT_DIR, <ARM>_holdout_geometry.zip: the raw model output (the model's own pose, the size head's size) in the
challenge's submission layout, geometry/<id>.ply with exactly 8192 points (finetune/predict.py writes the same layout).
Needs numpy, scipy, trimesh, pillow and matplotlib.
"""
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TRUTH, MODEL = "#3b6fb6", "#e8742a"
RGB = {"truth": (59, 111, 182), "guess": (232, 116, 42), "true_size": (60, 160, 80)}  # the same blue/orange, plus green


def write_ply(path, layers):
    """Binary little-endian PLY with vertex colours. layers: list of (points (n, 3), (r, g, b)). Units: mm."""
    pts = np.concatenate([p for p, _ in layers]).astype("<f4")
    rgb = np.concatenate([np.tile(np.array(c, np.uint8), (len(p), 1)) for p, c in layers])
    verts = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts["x"], verts["y"], verts["z"] = pts.T
    verts["red"], verts["green"], verts["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\ncomment units: millimetres, the ground truth's principal-axis frame\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


def pca(points):
    p = points - points.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    rotation = vt.T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return p @ rotation


def rescale(points, radius_mm):
    """finetune/common.metric_points: centre, then scale so the RMS radius is radius_mm."""
    points = np.asarray(points, np.float64)
    points = points - points.mean(0)
    return points / np.sqrt(np.square(points).sum(1).mean()) * radius_mm


def align(pred, truth, iterations=20):
    """evaluate.aligned_cd, but it also returns the aligned prediction and the truth, both in the truth's principal-axis frame."""
    a, b = pca(pred), pca(truth)
    tree = cKDTree(b)
    best, best_q = float("inf"), None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rot = np.diag(signs)[:, list(perm)]
            if np.linalg.det(rot) < 0:
                continue
            q = a @ rot
            for _ in range(iterations):
                _, index = tree.query(q)
                match = b[index]
                ca, cb = q.mean(0), match.mean(0)
                u, _, vt = np.linalg.svd((q - ca).T @ (match - cb))
                correction = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
                q = (q - ca) @ (u @ correction @ vt) + cb
            score = np.square(tree.query(q)[0]).mean() + np.square(cKDTree(q).query(b)[0]).mean()
            if score < best:
                best, best_q = float(score), q
    return best, best_q, b


def work(job):
    """One object: sample the truth, then align the model output at the size head's size and at the true size."""
    sid, mesh_path, unit_mm, pred, r_guess = job
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    truth, _ = trimesh.sample.sample_surface(mesh, 8192, seed=42)
    truth = truth * unit_mm
    truth = truth - truth.mean(0)
    r_true = float(np.sqrt(np.square(truth).sum(1).mean()))
    sc_guess, q_guess, b = align(pred, truth)
    sc_true, q_true, _ = align(rescale(pred, r_true), truth)
    return dict(id=sid, truth=b.astype(np.float32), q_guess=q_guess.astype(np.float32), q_true=q_true.astype(np.float32),
                r_true=r_true, r_guess=r_guess, sc_guess=sc_guess, sc_true=sc_true)


def draw(ax, truth, model, lim, note):
    ax.scatter(*truth.T, s=0.7, c=TRUTH, alpha=0.5, depthshade=False, linewidths=0)
    if model is not None:
        ax.scatter(*model.T, s=0.7, c=MODEL, alpha=0.5, depthshade=False, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-55)
    ax.set_axis_off()
    ax.text2D(0.5, 0.0, note, transform=ax.transAxes, ha="center", va="top", fontsize=9)


def object_image(it, rank, total, path):
    """One object, one image: photo | ground truth | output at the size head's size over the truth | output at the true size over it."""
    fig = plt.figure(figsize=(13.5, 3.85))
    lim = 1.05 * max(np.abs(it["truth"]).max(), np.abs(it["q_guess"]).max(), np.abs(it["q_true"]).max())
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(Image.open(it["image"]).convert("RGB").resize((300, 300)))
    ax.axis("off")
    ax.text(0.5, -0.04, f"#{rank} of {total}: {it['id']} ({it['category']})", transform=ax.transAxes, ha="center", va="top",
            fontsize=9)
    cols = [(None, f"real radius {it['r_true']:.0f} mm"),
            (it["q_guess"], f"our size guess {it['r_guess']:.0f} mm  |  CD {it['sc_guess']:.0f}"),
            (it["q_true"], f"true size {it['r_true']:.0f} mm  |  CD {it['sc_true']:.0f}")]
    for j, (m, note) in enumerate(cols, start=2):
        draw(fig.add_subplot(1, 4, j, projection="3d"), it["truth"], m, lim, note)
    heads = ["Photo (the model's input)", "Ground truth (blue)", "Model output at our size guess\n(orange) over truth (blue)",
             "Model output at the TRUE size\n(orange) over truth (blue)"]
    for j, text in enumerate(heads):
        fig.text(0.01 + 0.98 / 4 * (j + 0.5), 0.995, text, ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01, "CD = Chamfer distance, the competition's CD-L2 in mm² (lower is better; our local estimate)",
             ha="center", va="bottom", fontsize=9, style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.14, wspace=0.02)
    fig.savefig(path, dpi=75)
    plt.close(fig)


def draw_objects(items, out):
    """objects/<rank>_<id>.png for every object, in rank order (best score first)."""
    (out / "objects").mkdir(exist_ok=True)
    for rank, it in enumerate(items, start=1):
        object_image(it, rank, len(items), out / "objects" / f"{rank:03d}_{it['id']}.png")
        if rank % 25 == 0 or rank == len(items):
            print(f"drew {rank} of {len(items)}", flush=True)


def read_ply_layers(path):
    """The three equally long layers (truth, output at the size head's size, output at the true size) of a write_ply file."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    end = raw.index(marker) + len(marker)
    n = int(next(line for line in raw[:end].decode("ascii").splitlines() if line.startswith("element vertex")).split()[2])
    v = np.frombuffer(raw, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n,
                      offset=end)
    pts = np.stack([v["x"], v["y"], v["z"]], 1)
    k = n // 3
    return pts[:k], pts[k:2 * k], pts[2 * k:]


def redraw(hold, data, arm):
    """Draw the images and index.html again from the saved PLY files and index.csv, without aligning anything."""
    meta = json.load(open(hold / "metadata.json"))
    rows = {r["id"]: r for r in meta["rows"] if r["split"] == "val"}
    out = hold / f"{arm}_vs_truth"
    items = []
    for r in csv.DictReader(open(out / "index.csv")):
        truth, q_guess, q_true = read_ply_layers(out / "ply" / f"{int(r['rank']):03d}_{r['id']}.ply")
        items.append(dict(id=r["id"], category=r["category"], image=data / rows[r["id"]]["image"], truth=truth, q_guess=q_guess,
                          q_true=q_true, r_true=float(r["true_radius_mm"]), r_guess=float(r["guess_radius_mm"]),
                          sc_guess=float(r["score_at_guess"]), sc_true=float(r["score_at_true_size"])))
    print(f"redrawing {len(items)} objects from the saved PLY files and index.csv", flush=True)
    draw_objects(items, out)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from index_html import write_index

    write_index(out, arm)


def main():
    hold, data = Path(sys.argv[1]), Path(sys.argv[2])
    arm = sys.argv[3] if len(sys.argv) > 3 else "lora"
    count = sys.argv[4] if len(sys.argv) > 4 else "all"
    if len(sys.argv) > 5 and sys.argv[5] == "redraw":
        return redraw(hold, data, arm)
    meta = json.load(open(hold / "metadata.json"))
    rows = {r["id"]: r for r in meta["rows"] if r["split"] == "val"}
    z = np.load(hold / f"{arm}_val.npz", allow_pickle=False)
    s = np.load(hold / f"{arm}_scores.npz", allow_pickle=False)
    ids = z["ids"].tolist()
    assert s["ids"].tolist() == ids, "score file and prediction file list the objects in a different order"
    order = np.argsort(s["metric"])
    pick = order if count == "all" else order[np.linspace(0, len(ids) - 1, int(count)).round().astype(int)]
    jobs = [(ids[k], data / rows[ids[k]]["mesh"], meta["mesh_unit_mm"], z["pts"][k].astype(np.float64), float(z["radius_mm"][k]))
            for k in pick]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"aligning {len(jobs)} objects on {workers} processes", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        items = list(ex.map(work, jobs, chunksize=1))
    saved = {i: (a, b) for i, a, b in zip(ids, s["metric"], s["oracle"])}
    diff = max(max(abs(it["sc_guess"] - saved[it["id"]][0]), abs(it["sc_true"] - saved[it["id"]][1])) for it in items)
    print(f"largest difference between my scores and the saved ones: {diff:.4f}", flush=True)
    for it in items:
        it["category"] = rows[it["id"]]["category"]
        it["image"] = data / rows[it["id"]]["image"]

    out = hold / f"{arm}_vs_truth"
    out.mkdir(exist_ok=True)
    (out / "ply").mkdir(exist_ok=True)
    for r, it in enumerate(items, start=1):
        write_ply(out / "ply" / f"{r:03d}_{it['id']}.ply",
                  [(it["truth"], RGB["truth"]), (it["q_guess"], RGB["guess"]), (it["q_true"], RGB["true_size"])])
    print(f"wrote {len(items)} coloured PLY files to {out / 'ply'}", flush=True)
    if count == "all":
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from geom import pack  # the challenge's own writer: geometry/<id>.ply, exactly 8192 points per cloud

        zip_path = hold / f"{arm}_holdout_geometry.zip"
        pack(zip_path, {sid: z["pts"][k] for k, sid in enumerate(ids)})
        print(f"wrote {zip_path} ({len(ids)} raw clouds)", flush=True)
    draw_objects(items, out)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "category", "true_radius_mm", "guess_radius_mm", "guess_over_true", "score_at_guess",
                    "score_at_true_size"])
        for r, it in enumerate(items, start=1):
            w.writerow([r, it["id"], it["category"], f"{it['r_true']:.1f}", f"{it['r_guess']:.1f}",
                        f"{it['r_guess'] / it['r_true']:.2f}", f"{it['sc_guess']:.2f}", f"{it['sc_true']:.2f}"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from index_html import write_index

    write_index(out, arm)
    print("done:", out)


if __name__ == "__main__":
    main()
