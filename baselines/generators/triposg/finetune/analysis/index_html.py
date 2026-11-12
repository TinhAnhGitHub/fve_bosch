"""Write index.html for a <ARM>_vs_truth folder made by visualize_holdout.py: what the number is, the means, a sortable table with the
CD-L2 of every object, and the page images.

usage: index_html.py FOLDER [ARM]      (FOLDER has index.csv and page_NN.png)
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
HEADS = [("Rank", 1), ("Object", 0), ("Category", 0), ("Real radius (mm)", 1), ("Our size guess (mm)", 1), ("Guess / real", 1),
         ("CD at our size guess (mm&sup2;)", 1), ("CD at the true size (mm&sup2;)", 1), ("Picture", 0)]


def cls(cd):
    return "bad" if cd > 100 else "mid" if cd > 30 else "ok"


def write_index(folder, arm="lora"):
    folder = Path(folder)
    rows = list(csv.DictReader(open(folder / "index.csv")))
    n = len(rows)
    guess = [float(r["score_at_guess"]) for r in rows]
    true = [float(r["score_at_true_size"]) for r in rows]
    worst10 = sum(sorted(guess)[-10:]) / sum(guess)
    body, blocks = [], []
    for r in rows:
        g, t = float(r["score_at_guess"]), float(r["score_at_true_size"])
        body.append(
            f'<tr><td data-v="{r["rank"]}">{r["rank"]}</td><td data-v="{r["id"]}"><a href="#{r["id"]}">{r["id"]}</a></td>'
            f'<td data-v="{r["category"]}">{r["category"]}</td><td data-v="{r["true_radius_mm"]}">{r["true_radius_mm"]}</td>'
            f'<td data-v="{r["guess_radius_mm"]}">{r["guess_radius_mm"]}</td><td data-v="{r["guess_over_true"]}">'
            f'{r["guess_over_true"]}</td><td data-v="{g}" class="cd {cls(g)}">{g:.1f}</td>'
            f'<td data-v="{t}" class="{cls(t)}">{t:.1f}</td><td><a href="#{r["id"]}">see image</a></td></tr>')
        blocks.append(
            f'<div class="obj" id="{r["id"]}"><div class="oh"><b>{r["id"]}</b> ({r["category"]}) &nbsp;|&nbsp; rank {r["rank"]} of {n}'
            f' &nbsp;|&nbsp; <span class="cd {cls(g)}">CD at our size guess {g:.1f}</span> &nbsp;|&nbsp; '
            f'<span class="{cls(t)}">CD at the true size {t:.1f}</span> &nbsp;|&nbsp; real radius {r["true_radius_mm"]} mm, '
            f'our guess {r["guess_radius_mm"]} mm (x{r["guess_over_true"]})</div>'
            f'<img loading="lazy" src="objects/{int(r["rank"]):03d}_{r["id"]}.png"></div>')
    head = "".join(f'<th onclick="sortTable({i}, {num})">{h}</th>' for i, (h, num) in enumerate(HEADS))
    imgs = "\n".join(blocks)
    html = f"""<html><head><meta charset="utf-8"><title>{arm} vs ground truth, CD-L2</title><style>{CSS}</style></head>
<body><h2>{arm}: model output vs ground truth, {n} holdout objects, with the competition score (CD-L2)</h2>
<div class="note"><b>The score is CD-L2, the competition's Chamfer distance, in mm&sup2;. Lower is better.</b>
For every predicted point take the squared distance to the nearest true point, and for every true point the squared distance to
the nearest predicted point; average each and add them. The grader first turns and shifts the prediction to fit the truth but
never rescales it, so a wrong size is punished in full.<br>
These numbers are <b>our local estimate</b>: the truth is sampled at 8,192 points (the grader uses 10,000) and the fit is our own
(24 starting rotations, 20 ICP steps). It has not been checked against the official grader, and leaderboard scores have run about
1.3 to 1.7 times our local ones for our other models.</div>
<table class="sum"><tr><th></th><th>mean CD</th><th>median CD</th></tr>
<tr><td style="text-align:left">At our size guess (what a submission would get)</td><td class="cd">{statistics.mean(guess):.1f}</td><td>{statistics.median(guess):.1f}</td></tr>
<tr><td style="text-align:left">At the true size (shape only, an oracle, not achievable)</td><td>{statistics.mean(true):.1f}</td><td>{statistics.median(true):.1f}</td></tr></table>
<p>The 10 worst objects make up {worst10 * 100:.0f}% of the total CD at our size guess. For scale: our best submission scored 43.85
on the leaderboard (public test objects, official grader), so it is not directly comparable with these holdout numbers.</p>
<p><b>To find one object:</b> use the browser's find box (Ctrl+F) with its full id (written <code>omni_&lt;food&gt;_&lt;number&gt;</code>
or <code>mf3d_&lt;food&gt;_&lt;number&gt;</code>). The first match is its row in the table; press Enter once more to land on its
heading and image further down. Or click the id in the table.
Click a column heading to sort. Red: CD above 100. Orange: 30 to 100. Green: below 30.
Pictures: blue = ground truth (points sampled from the real scan), orange = model output.
3D files: open any file in the <code>ply</code> folder with MeshLab, CloudCompare or Blender; blue = ground truth,
orange = model output at our size guess, green = model output at the true size; units are mm.</p>
<table id="t"><thead><tr>{head}</tr></thead><tbody>
{"".join(body)}
</tbody></table>
{imgs}<script>{JS}</script></body></html>"""
    (folder / "index.html").write_text(html, encoding="utf-8")
    print("wrote", folder / "index.html")


if __name__ == "__main__":
    write_index(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "lora")
