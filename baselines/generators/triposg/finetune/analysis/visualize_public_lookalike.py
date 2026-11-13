"""Draw the public test as the look-alike submission would send it: the LoRA clouds at the look-alike size, the two look-alike training
photos behind each size, and a score band taken from the holdout.

The public test has no ground truth, so no object has a score. Instead each object shows (1) its output, (2) the evidence for its size:
the two most alike training photos of the same food type with their real radii, and (3) how much those two agree. On the holdout the
agreement predicts the score well, so every object also gets the score that holdout objects with the same agreement had.

usage: visualize_public_lookalike.py SIZE_DIR DATA_DIR OUT_DIR
  SIZE_DIR  the size_lookalike.py output folder: public_lora_lookalike.npz, public_neighbours.csv, holdout_cd.npz, holdout_sizes.npz
  DATA_DIR  data/f3d (public photos eval/<group>/<category>/<id>/, training photos train/<group>/<category>/<id>/)
  OUT_DIR   created: objects/<id>.png, ply/<id>.ply, groups.png, index.csv, index.html
Needs numpy, scipy, trimesh, pillow and matplotlib (it reuses visualize_holdout.py and visualize_public.py).
"""
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_holdout import RGB, MODEL, pca, write_ply  # noqa: E402
from visualize_public import CSS, JS, VIEWS  # noqa: E402

CUTS = (0.056, 0.174)  # log-radius gap between the two look-alikes: the terciles of the holdout objects
GROUPS = [("agree", "agree: the two look-alike radii are within 6% of each other", "ok"),
          ("partly", "partly agree: 6% to 19% apart", "mid"),
          ("differ", "disagree: more than 19% apart", "bad"),
          ("one only", "only one look-alike in the training set (a rare food type)", "bad")]
LEADERBOARD_BEST, RATIO = 43.85, 1.5
LEADERBOARD_THIS = 102.82  # this recipe's zip (v36_lora_lookalike_size), uploaded by the user on 2026-11-13 09:27 Vietnam time
EXTRA_CSS = ".ok{background:#dff3df}.mid{background:#fdeccf}.bad{background:#fbd5d0}.grp{padding:1px 6px;font-weight:bold}"


def group_of(gap):
    return 3 if np.isnan(gap) else int(np.digitize(gap, CUTS))


def calibration(size_dir):
    """What the holdout objects scored at the look-alike size, by how much their two look-alikes agreed."""
    cd, sizes = np.load(size_dir / "holdout_cd.npz"), np.load(size_dir / "holdout_sizes.npz")
    score = cd["cd"][:, cd["arms"].tolist().index("blend")]
    groups = np.array([group_of(g) for g in sizes["neighbour_gap"]])
    stats = [dict(n=int((groups == g).sum()), mean=float(score[groups == g].mean()), median=float(np.median(score[groups == g])),
                  over100=float((score[groups == g] > 100).mean())) for g in range(4)]
    return stats, float(score.mean()), float(np.median(score))


def band_text(stat, group):
    if group == 3:
        return (f"only {stat['n']} holdout objects had a single look-alike, too few to judge (their mean score was {stat['mean']:.0f}; "
                "one of them, a steak, scored 473)")
    return (f"holdout objects with the same agreement scored a median {stat['median']:.0f} (mean {stat['mean']:.0f}), "
            f"{stat['over100'] * 100:.0f}% of them above 100")


def draw_object(job):
    sid, category, photo, cloud, radius, head, knn, formula, near, path = job
    lim = 1.1 * np.abs(cloud).max()
    fig = plt.figure(figsize=(17.5, 3.95))
    ax = fig.add_subplot(1, 6, 1)
    ax.imshow(Image.open(photo).convert("RGB").resize((300, 300)))
    ax.axis("off")
    ax.text(0.5, -0.04, f"{sid} ({category})", transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    for j, (title, elev, azim) in enumerate(VIEWS, start=2):
        ax = fig.add_subplot(1, 6, j, projection="3d")
        ax.scatter(*cloud.T, s=0.7, c=MODEL, alpha=0.6, depthshade=False, linewidths=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.text2D(0.5, 0.0, f"size from the look-alikes: radius {radius:.0f} mm" if j == 2 else f"plot box +-{lim:.0f} mm",
                  transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        fig.text(0.01 + 0.98 / 6 * (j - 0.5), 0.995, title.replace("Model output, ", "LoRA, "), ha="center", va="top", fontsize=10,
                 fontweight="bold")
    fig.text(0.01 + 0.98 / 12, 0.995, "Photo (the input)", ha="center", va="top", fontsize=10, fontweight="bold")
    for n, (nid, similarity, nradius, nphoto) in enumerate(near):
        ax = fig.add_subplot(1, 6, 5 + n)
        if nid:
            ax.imshow(Image.open(nphoto).convert("RGB").resize((300, 300)))
            ax.text(0.5, -0.04, f"{nid}\nsimilarity {similarity:.2f}, real radius {nradius:.1f} mm", transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.5)
        ax.axis("off")
        fig.text(0.01 + 0.98 / 6 * (4.5 + n), 0.995, f"Look-alike {n + 1} (training photo)", ha="center", va="top", fontsize=10,
                 fontweight="bold")
    fig.text(0.5, 0.008, f"Size: look-alikes say {knn:.1f} mm, the ridge formula {formula:.1f} mm, final {radius:.1f} mm "
             f"(the size head said {head:.1f} mm). No ground truth exists for the public test, so there is no score.",
             ha="center", va="bottom", fontsize=8.5, style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.16, wspace=0.03)
    fig.savefig(path, dpi=75)
    plt.close(fig)
    return path


def draw_groups(path, stats, overall, counts):
    stats = stats[:3]  # the three agreement groups; the fourth (a single look-alike, 3 holdout objects) is only in the table
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    colors = ["#4c9f4c", "#e0a030", "#d9534f"]
    ax.bar(range(3), [s["mean"] for s in stats], color=colors)
    for i, s in enumerate(stats):
        ax.text(i, s["mean"] + 2, f"mean {s['mean']:.0f}\nmedian {s['median']:.0f}", ha="center", fontsize=9)
    ax.axhline(24.2, ls="-.", color="#4c9f4c", lw=1, label="24.2: LoRA at the true size (its best case)")
    ax.axhline(LEADERBOARD_BEST / RATIO, ls=":", color="k", lw=1.3, label=f"about {LEADERBOARD_BEST / RATIO:.0f}: RH-Mix5 on this holdout (estimate)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["look-alikes\nagree", "partly\nagree", "look-alikes\ndisagree"])
    ax.set_ylim(0, max(s["mean"] for s in stats) * 1.3)
    ax.set_ylabel("holdout score at the look-alike size (mm$^2$)")
    ax.set_title(f"Score by how much the two look-alikes agree (holdout, 204 objects, overall mean {overall:.1f})", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.bar(range(3), counts[:3], color=colors)
    for i, c in enumerate(counts[:3]):
        ax.text(i, c + 4, str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["look-alikes\nagree", "partly\nagree", "look-alikes\ndisagree"])
    ax.set_ylabel("public test photos")
    ax.set_title(f"The {sum(counts)} public test photos by the same grouping", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_html(out, rows, stats, overall, median, counts):
    heads = [("Object", 0), ("Category", 0), ("Size head (mm)", 1), ("Look-alike size, final (mm)", 1), ("Final / size head", 1),
             ("Look-alikes' radii differ by (%)", 1), ("Best look-alike similarity", 1), ("Look-alikes", 0),
             ("Holdout median score for this agreement", 1)]
    table = []
    for r in rows:
        label, css = GROUPS[r["group"]][0], GROUPS[r["group"]][2]
        stat = "?" if r["group"] == 3 else f'{stats[r["group"]]["median"]:.0f}'
        value = 9999 if r["group"] == 3 else stats[r["group"]]["median"]
        differ = "" if np.isnan(r["gap"]) else f"{(np.exp(r['gap']) - 1) * 100:.0f}"
        table.append(f'<tr><td data-v="{r["id"]}"><a href="#{r["id"]}">{r["id"]}</a></td><td data-v="{r["category"]}">{r["category"]}</td>'
                     f'<td data-v="{r["head"]:.2f}">{r["head"]:.1f}</td><td data-v="{r["radius"]:.2f}">{r["radius"]:.1f}</td>'
                     f'<td data-v="{r["radius"] / r["head"]:.3f}">{r["radius"] / r["head"]:.2f}</td>'
                     f'<td data-v="{differ or 999}">{differ}</td><td data-v="{r["best"]:.3f}">{r["best"]:.2f}</td>'
                     f'<td data-v="{r["group"]}" class="{css}">{label}</td><td data-v="{value:.1f}" class="{css}">{stat}</td></tr>')
    blocks = []
    for r in rows:
        g = GROUPS[r["group"]]
        css = g[2]
        text = f'{"look-alikes " if r["group"] < 3 else ""}{g[1]}; {band_text(stats[r["group"]], r["group"])}'
        blocks.append(f'<div class="obj" id="{r["id"]}"><div class="oh"><b>{r["id"]}</b> ({r["category"]}) &nbsp;|&nbsp; size {r["radius"]:.1f} mm '
                      f'(size head {r["head"]:.1f} mm, x{r["radius"] / r["head"]:.2f}) &nbsp;|&nbsp; <span class="grp {css}">{text}</span></div>'
                      f'<img loading="lazy" src="objects/{r["id"]}.png"></div>')
    head = "".join(f'<th onclick="sortTable({i}, {num})">{h}</th>' for i, (h, num) in enumerate(heads))
    rows_html = "".join(table)
    group_rows = ""
    for g in range(4):
        group_rows += (f'<tr class="{GROUPS[g][2]}"><td style="text-align:left">{GROUPS[g][1]}</td><td>{stats[g]["n"]}</td>'
                       f'<td>{stats[g]["mean"]:.1f}</td><td>{stats[g]["median"]:.1f}</td><td>{stats[g]["over100"] * 100:.0f}%</td>'
                       f'<td>{counts[g]}</td></tr>')
    sizes = [r["radius"] for r in rows]
    local = LEADERBOARD_BEST / RATIO
    html = f"""<html><head><meta charset="utf-8"><title>LoRA on the public test, look-alike sizes</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><h2>LoRA (epoch 4) on the public test: the look-alike submission, {len(rows)} objects</h2>
<div class="note"><b>The public test has no ground truth, so no object here has a score.</b> What this page shows for each photo: the LoRA's
output scaled to the size taken from the 2 most alike training photos of the same food type (their photos and real radii are on the right
of every image), and how much those two look-alikes agree. On the 204 holdout objects the agreement predicted the score well, so each
object carries the score that holdout objects with the same agreement had. It is a band for objects like this one, not this object's score.</div>
<img src="groups.png" style="width:auto;max-width:100%;margin:8px 0">
<table class="sum" style="margin:6px 0"><tr><th style="cursor:default">Look-alikes</th><th style="cursor:default">Holdout objects</th>
<th style="cursor:default">Mean score</th><th style="cursor:default">Median</th><th style="cursor:default">Above 100</th>
<th style="cursor:default">Public test photos</th></tr>
{group_rows}
</table>
<p><b>How good is it? Worse than our best everywhere.</b> On the holdout this recipe scored a mean of {overall:.1f} (median {median:.1f}); the size head's own sizes scored 79.3,
and the true size 24.2, which is the LoRA's best case and not a result. Our best submission, RH-Mix5, scored {LEADERBOARD_BEST} on the leaderboard, and leaderboard scores have run 1.3 to 1.7 times holdout
scores, so RH-Mix5 is about {local:.0f} on this holdout. <b>This file was uploaded on 2026-11-13 and scored {LEADERBOARD_THIS} on the leaderboard</b>, 1.69 times its holdout mean.
Even the better two thirds of the objects (the look-alikes agree or partly agree: median about 19 to 20, mean 32 to 38) average above RH-Mix5's {local:.0f};
the third where they disagree has a mean of 106 and carries most of the loss. Nothing on this page is close to our best submission. The bands below only say
where the LoRA is least bad.</p>
<p>Median size {statistics.median(sizes):.1f} mm, smallest {min(sizes):.1f}, largest {max(sizes):.1f}.
<b>To find one object:</b> use the browser's find box (Ctrl+F) with its full id (written <code>omni_&lt;food&gt;_&lt;number&gt;</code> or
<code>mf3d_&lt;food&gt;_&lt;number&gt;</code>). The first match is its row in the table; press Enter once more to land on its heading and image further down.
Click a column heading to sort. Green: the look-alikes agree. Orange: partly. Red: they disagree, or only one look-alike exists.
3D files: open any file in the <code>ply</code> folder with MeshLab, CloudCompare or Blender (orange, mm, at the look-alike size).</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table>
{"".join(blocks)}<script>{JS}</script></body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


def main():
    size_dir, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    (out / "objects").mkdir(parents=True, exist_ok=True)
    (out / "ply").mkdir(exist_ok=True)
    table = list(csv.DictReader(open(size_dir / "public_neighbours.csv")))
    with np.load(size_dir / "public_lora_lookalike.npz", allow_pickle=False) as z:
        clouds = dict(zip(z["ids"].tolist(), z["pts"]))
    public = {p.parent.name: p for p in (data / "eval").glob("*/*/*/rgb_industrial.png")}
    train = {p.parent.name: p for p in (data / "train").glob("*/*/*/rgb_industrial.png")}
    if {r["id"] for r in table} != set(clouds) or set(clouds) != set(public) or len(table) != 511:
        raise SystemExit("the neighbour table, the clouds and the public photos do not cover the same 511 objects")
    stats, overall, median = calibration(size_dir)
    jobs, rows = [], []
    for r in table:
        sid = r["id"]
        radius, head = float(r["final_radius_mm"]), float(r["head_radius_mm"])
        gap = float(r["neighbour_gap"]) if r["neighbour_gap"] else float("nan")
        cloud = pca(clouds[sid].astype(np.float64))
        near = [(r[f"nn{n}_id"], float(r[f"nn{n}_similarity"]) if r[f"nn{n}_id"] else 0.0,
                 float(r[f"nn{n}_radius_mm"]) if r[f"nn{n}_id"] else 0.0, train.get(r[f"nn{n}_id"])) for n in (1, 2)]
        png = out / "objects" / f"{sid}.png"
        if not png.exists():  # delete the folder to draw everything again
            jobs.append((sid, r["category"], public[sid], cloud, radius, head, float(r["lookalike_radius_mm"]),
                         float(r["formula_radius_mm"]), near, png))
            write_ply(out / "ply" / f"{sid}.ply", [(cloud, RGB["guess"])])
        rows.append(dict(id=sid, category=r["category"], radius=radius, head=head, gap=gap, best=float(r["best_similarity"]),
                         group=group_of(gap)))
    counts = [sum(row["group"] == g for row in rows) for g in range(4)]
    print(f"{len(rows)} public objects; by look-alike agreement: {counts[:3]} (agree / partly / differ), {counts[3]} with one look-alike",
          flush=True)
    with ProcessPoolExecutor(14) as pool:
        for k, _ in enumerate(pool.map(draw_object, jobs, chunksize=4), start=1):
            if k % 50 == 0 or k == len(jobs):
                print(f"{k} of {len(jobs)}", flush=True)
    draw_groups(out / "groups.png", stats, overall, counts)
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "size_head_mm", "final_radius_mm", "look_alike_gap", "best_similarity", "agreement_group",
                         "holdout_median_score_for_group"])
        for r in rows:
            writer.writerow([r["id"], r["category"], f"{r['head']:.2f}", f"{r['radius']:.2f}",
                             "" if np.isnan(r["gap"]) else f"{r['gap']:.4f}", f"{r['best']:.4f}", GROUPS[r["group"]][0],
                             "" if r["group"] == 3 else f"{stats[r['group']]['median']:.1f}"])
    write_html(out, rows, stats, overall, median, counts)
    print("done:", out)


if __name__ == "__main__":
    main()
