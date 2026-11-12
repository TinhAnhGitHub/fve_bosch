"""How good is the size head? Compare its holdout radius error with two trivial guesses learned from the train split:
one size for everything, and the median size of the object's category."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

DATA = Path("/mnt/c/tinhanh/project/ai_challenge/data/f3d")
meta = json.load(open("metadata.json"))


def radius(row):
    m = trimesh.load(DATA / row["mesh"], force="mesh", process=False)
    t, _ = trimesh.sample.sample_surface(m, 8192, seed=42)
    return float(np.sqrt(np.square(t - t.mean(0)).sum(1).mean()))


if __name__ == "__main__":
    rows = [r for r in meta["rows"] if r["split"] in ("train", "val")]
    with ProcessPoolExecutor(14) as ex:
        rad = dict(zip([r["id"] for r in rows], ex.map(radius, rows, chunksize=8)))
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    glob = np.median([rad[r["id"]] for r in tr])
    cat = {}
    for r in tr:
        cat.setdefault(r["category"], []).append(rad[r["id"]])
    cat = {c: np.median(v) for c, v in cat.items()}
    z = np.load("lora_val.npz")
    head = dict(zip(z["ids"].tolist(), z["radius_mm"]))
    true = np.array([rad[r["id"]] for r in va])
    for name, guess in (("one size for all", [glob] * len(va)),
                        ("category median", [cat.get(r["category"], glob) for r in va]),
                        ("our size head", [head[r["id"]] for r in va])):
        rel = np.abs(np.array(guess) / true - 1)
        print(f"{name:17s} median error {np.median(rel)*100:4.0f}%  within 10%: {(rel < .1).mean()*100:3.0f}%  "
              f"off >30%: {(rel > .3).mean()*100:3.0f}%")
