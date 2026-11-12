import json
import numpy as np, trimesh
from pathlib import Path
meta = json.load(open("metadata.json")); rows = {r["id"]: r for r in meta["rows"] if r["split"] == "val"}
z = np.load("lora_val.npz"); s = np.load("lora_scores.npz")
true = []
for sid in z["ids"].tolist():
    t, _ = trimesh.sample.sample_surface(trimesh.load(Path("/mnt/c/tinhanh/project/ai_challenge/data/f3d") / rows[sid]["mesh"], force="mesh", process=False), 8192, seed=42)
    true.append(np.sqrt(np.square(t - t.mean(0)).sum(1).mean()))
true, pred = np.array(true), z["radius_mm"]
rel = pred / true - 1
print(f"true radius median {np.median(true):.1f} mm; size-head error: median |rel| {np.median(np.abs(rel))*100:.0f}%, "
      f"within 10%: {(np.abs(rel)<0.1).mean()*100:.0f}%, off by >30%: {(np.abs(rel)>0.3).mean()*100:.0f}%")
gap = s["metric"] - s["oracle"]
for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, 9)):
    m = (np.abs(rel) >= lo) & (np.abs(rel) < hi)
    print(f"  |size error| {lo*100:.0f}-{min(hi,9)*100:.0f}%: n={m.sum():3d}  CD at predicted size {s['metric'][m].mean():7.1f}  at true size {s['oracle'][m].mean():6.1f}")
