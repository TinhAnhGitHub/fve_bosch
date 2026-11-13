"""Write index.html for a <ARM>_vs_truth folder made by visualize_holdout.py: what the number is, the means, a chart of the scores,
a sortable table with the CD-L2 of every object, and the page images. If the folder also has index_lookalike.csv (written by
size_lookalike.py table), the look-alike size guess and its score are added next to the size head's.

usage: index_html.py FOLDER [ARM]      (FOLDER has index.csv, objects/ and, optionally, index_lookalike.csv)
The chart (scores_overview.png) needs numpy and matplotlib; without them the page is written without it.
"""
import csv
import statistics
import sys
from pathlib import Path

CSS = """
body{font-family:sans-serif;max-width:1150px;margin:auto;padding:0 12px;color:#222}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #ccc;padding:3px 8px;text-align:right;font-size:14px}
td:nth-child(2),td:nth-child(3),th:nth-child(2),th:nth-child(3){text-align:left}
th{background:#f1f1f1;cursor:pointer;position:sticky;top:0}
.sum th,.sum td{cursor:default;position:static}
.cd{font-weight:bold}.bad{background:#fbd5d0}.mid{background:#fdeccf}.ok{background:#dff3df}
.note{background:#f6f8fa;border-left:4px solid #3b6fb6;padding:6px 12px;margin:10px 0;font-size:14px}
.obj{margin:18px 0}
.oh{background:#eef3f9;padding:6px 10px;font-size:15px;border:1px solid #c9d6e6;border-bottom:none}
.oh span{padding:1px 6px}
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
LEADERBOARD_BEST = 43.85  # RH-Mix5 on the public test, official grader
LOCAL_OVER_LEADERBOARD = 1.5  # leaderboard scores have run about 1.5 times our local holdout scores (report 8.2: 1.26 to 1.72)


def cls(cd):
    return "bad" if cd > 100 else "mid" if cd > 30 else "ok"


def read_lookalike(folder):
    """id -> (look-alike radius in mm, CD at that size), or {} when the folder has no index_lookalike.csv."""
    path = Path(folder) / "index_lookalike.csv"
    if not path.exists():
        return {}
    return {r["id"]: (float(r["lookalike_radius_mm"]), float(r["score_at_lookalike_size"])) for r in csv.DictReader(open(path))}


def draw_chart(path, head, look, true):
    """Mean CD per arm with the two reference lines, the share of objects below each CD, and look-alike against size head per object."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    arms = [("size head's guess", np.array(head), "#d9534f")]
    if look:
        arms.append(("look-alike size", np.array(look), "#2f6db5"))
    arms.append(("true size\n(oracle)", np.array(true), "#4c9f4c"))
    fig, axes = plt.subplots(1, 3 if look else 2, figsize=(15.5, 4.6))
    ax = axes[0]
    means = [float(a[1].mean()) for a in arms]
    ax.bar(range(len(arms)), means, color=[a[2] for a in arms])
    for i, m in enumerate(means):
        ax.text(i, m + 1.5, f"{m:.1f}", ha="center", fontsize=11, fontweight="bold")
    local = LEADERBOARD_BEST / LOCAL_OVER_LEADERBOARD
    ax.axhline(LEADERBOARD_BEST, ls="--", color="k", lw=1, label=f"{LEADERBOARD_BEST}: RH-Mix5 on the leaderboard (another test set)")
    ax.axhline(local, ls=":", color="k", lw=1.3, label=f"about {local:.0f}: RH-Mix5 on this holdout (estimate)")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_ylabel("mean CD-L2 over the holdout objects (mm$^2$)")
    ax.set_title("Mean score, lower is better", fontsize=10)
    ax = axes[1]
    for name, values, color in arms:
        ordered = np.sort(values)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100, color=color, lw=2,
                label=f"{name.replace(chr(10), ' ')} (median {np.median(values):.1f})")
    ax.set_xscale("log")
    for x in (30, 100):
        ax.axvline(x, ls=":", color="gray", lw=1)
    ax.set_xlabel("CD-L2 of one object (mm$^2$, log scale; dotted: 30 and 100, the table colours)")
    ax.set_ylabel("% of objects at or below this score")
    ax.set_title("How the objects are spread", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    if look:
        ax = axes[2]
        x, y = np.array(head), np.array(look)
        ax.scatter(x, y, s=12, alpha=0.6, color="#2f6db5", linewidths=0)
        low, high = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
        ax.plot([low, high], [low, high], "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("CD at the size head's guess (mm$^2$)")
        ax.set_ylabel("CD at the look-alike size (mm$^2$)")
        ax.set_title(f"One dot per object; below the line = look-alike size better\n(better on {int((y < x).sum())} of {len(x)} objects)", fontsize=10)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def write_index(folder, arm="lora"):
    folder = Path(folder)
    rows = list(csv.DictReader(open(folder / "index.csv")))
    look = read_lookalike(folder)
    if look and set(look) != {r["id"] for r in rows}:
        raise ValueError("index_lookalike.csv does not cover the same objects as index.csv")
    n = len(rows)
    guess = [float(r["score_at_guess"]) for r in rows]
    true = [float(r["score_at_true_size"]) for r in rows]
    new = [look[r["id"]][1] for r in rows] if look else []
    chart = draw_chart(folder / "scores_overview.png", guess, new, true)
    heads = [("Rank", 1), ("Object", 0), ("Category", 0), ("Real radius (mm)", 1), ("Size head's guess (mm)", 1), ("Guess / real", 1)]
    if look:
        heads.append(("Look-alike guess (mm)", 1))
    heads.append(("CD at the size head's guess (mm&sup2;)", 1))
    if look:
        heads.append(("CD at the look-alike size (mm&sup2;)", 1))
    heads += [("CD at the true size (mm&sup2;)", 1), ("Picture", 0)]
    body, blocks = [], []
    for r in rows:
        g, t = float(r["score_at_guess"]), float(r["score_at_true_size"])
        cells = (f'<td data-v="{r["rank"]}">{r["rank"]}</td><td data-v="{r["id"]}"><a href="#{r["id"]}">{r["id"]}</a></td>'
                 f'<td data-v="{r["category"]}">{r["category"]}</td><td data-v="{r["true_radius_mm"]}">{r["true_radius_mm"]}</td>'
                 f'<td data-v="{r["guess_radius_mm"]}">{r["guess_radius_mm"]}</td><td data-v="{r["guess_over_true"]}">'
                 f'{r["guess_over_true"]}</td>')
        extra_head = extra_cell = ""
        if look:
            radius, s = look[r["id"]]
            ratio = radius / float(r["true_radius_mm"])
            cells += f'<td data-v="{radius}">{radius:.1f}</td>'
            extra_cell = f'<td data-v="{s}" class="cd {cls(s)}">{s:.1f}</td>'
            extra_head = (f' &nbsp;|&nbsp; <span class="cd {cls(s)}">CD at the look-alike size {s:.1f}</span> (radius {radius:.1f} mm, '
                          f'x{ratio:.2f})')
        body.append(f'<tr>{cells}<td data-v="{g}" class="{"" if look else "cd "}{cls(g)}">{g:.1f}</td>{extra_cell}'
                    f'<td data-v="{t}" class="{cls(t)}">{t:.1f}</td><td><a href="#{r["id"]}">see image</a></td></tr>')
        blocks.append(
            f'<div class="obj" id="{r["id"]}"><div class="oh"><b>{r["id"]}</b> ({r["category"]}) &nbsp;|&nbsp; rank {r["rank"]} of {n}'
            f' &nbsp;|&nbsp; <span class="{"" if look else "cd "}{cls(g)}">CD at the size head\'s guess {g:.1f}</span>{extra_head} &nbsp;|&nbsp; '
            f'<span class="{cls(t)}">CD at the true size {t:.1f}</span> &nbsp;|&nbsp; real radius {r["true_radius_mm"]} mm, '
            f'size head\'s guess {r["guess_radius_mm"]} mm (x{r["guess_over_true"]})</div>'
            f'<img loading="lazy" src="objects/{int(r["rank"]):03d}_{r["id"]}.png"></div>')
    head = "".join(f'<th onclick="sortTable({i}, {num})">{h}</th>' for i, (h, num) in enumerate(heads))
    imgs = "\n".join(blocks)
    summary = (f'<tr><td style="text-align:left">At the size head\'s guess (what a submission would get today)</td>'
               f'<td class="cd">{statistics.mean(guess):.1f}</td><td>{statistics.median(guess):.1f}</td></tr>')
    if look:
        summary += (f'<tr><td style="text-align:left"><b>At the look-alike size guess (the new plan)</b></td>'
                    f'<td class="cd">{statistics.mean(new):.1f}</td><td>{statistics.median(new):.1f}</td></tr>')
    summary += (f'<tr><td style="text-align:left">At the true size (shape only, an oracle, not achievable)</td>'
                f'<td>{statistics.mean(true):.1f}</td><td>{statistics.median(true):.1f}</td></tr>')
    worst10 = sum(sorted(new or guess)[-10:]) / sum(new or guess)
    which = "look-alike size" if look else "size head's guess"
    look_note = ""
    if look:
        look_note = ("""<div class="note"><b>Look-alike size guess:</b> for each object, the real radii of the 2 training photos of the same food type
that look most alike (SigLIP2 image features), mixed with a small ridge formula on the same features. The pool is the 1,838 training objects:
<b>no holdout object is ever in it</b>, and the mixing weights were chosen out-of-fold on the pool, not on these objects. The LoRA shape is
only scaled to that radius. The pictures below still show the size head's guess and the true size; the look-alike size is in the numbers
and the chart only.</div>""")
    chart_html = '<img src="scores_overview.png" style="margin:10px 0">' if chart else ""
    local = LEADERBOARD_BEST / LOCAL_OVER_LEADERBOARD
    html = f"""<html><head><meta charset="utf-8"><title>{arm} vs ground truth, CD-L2</title><style>{CSS}</style></head>
<body><h2>{arm}: model output vs ground truth, {n} holdout objects, with the competition score (CD-L2)</h2>
<div class="note"><b>The score is CD-L2, the competition's Chamfer distance, in mm&sup2;. Lower is better.</b>
For every predicted point take the squared distance to the nearest true point, and for every true point the squared distance to
the nearest predicted point; average each and add them. The grader first turns and shifts the prediction to fit the truth but
never rescales it, so a wrong size is punished in full.<br>
These numbers are <b>our local estimate</b>: the truth is sampled at 8,192 points (the grader uses 10,000) and the fit is our own
(24 starting rotations, 20 ICP steps). It has not been checked against the official grader, and leaderboard scores have run about
1.3 to 1.7 times our local ones for our other models.</div>
{look_note}
<table class="sum"><tr><th></th><th>mean CD</th><th>median CD</th></tr>
{summary}</table>
{chart_html}
<p>The 10 worst objects make up {worst10 * 100:.0f}% of the total CD at the {which}. For scale: our best submission (RH-Mix5) scored
{LEADERBOARD_BEST} on the leaderboard (public test objects, official grader). Our models' leaderboard scores have run about
{LOCAL_OVER_LEADERBOARD} times their holdout scores, so on this holdout RH-Mix5 would be about {local:.0f}: that, not {LEADERBOARD_BEST},
is the number to compare these means with. It is an estimate; RH-Mix5 itself has not been scored on this holdout with this scorer.</p>
<p><b>To find one object:</b> use the browser's find box (Ctrl+F) with its full id (written <code>omni_&lt;food&gt;_&lt;number&gt;</code>
or <code>mf3d_&lt;food&gt;_&lt;number&gt;</code>). The first match is its row in the table; press Enter once more to land on its
heading and image further down. Or click the id in the table.
Click a column heading to sort. Red: CD above 100. Orange: 30 to 100. Green: below 30.
Pictures: blue = ground truth (points sampled from the real scan), orange = model output.
3D files: open any file in the <code>ply</code> folder with MeshLab, CloudCompare or Blender; blue = ground truth,
orange = model output at the size head's guess, green = model output at the true size; units are mm.</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>
{"".join(body)}
</tbody></table>
{imgs}<script>{JS}</script></body></html>"""
    (folder / "index.html").write_text(html, encoding="utf-8")
    print("wrote", folder / "index.html", "(with chart)" if chart else "(no chart: numpy/matplotlib missing)")


if __name__ == "__main__":
    write_index(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "lora")
