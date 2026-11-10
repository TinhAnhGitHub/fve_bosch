# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Export everything the Kaggle trainer needs (no repo imports on Kaggle): row lists, per-fold training targets
(r, ok, oversized label, mixed mask, standardised aux = 3 sorted ln bbox dims + ln fill [+ 4 mesh descriptors]),
aux mean/std (to undo the standardisation), and decode tables (base ln values) for held-out and val rows.
Seed-0 folds, identical to B4.0 splits. The FVE download has no mesh.ply: the mesh descriptors are added only when
features/mesh_targets.npz exists (built from the F3DR track's meshes; check the rules before using it).
Output: kaggle_pkg/b42_prep/{rows.csv, fold{f}.npz}
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines"))
from b0_b1 import DATA
from b3 import build_table, cluster_labels
from frozen import Experiment

GAP = math.log(30.0)
OUT = ROOT / "kaggle_pkg" / "b42_prep"; OUT.mkdir(parents=True, exist_ok=True)
exp = Experiment()
MT = ROOT / "features" / "mesh_targets.npz"
if MT.exists() and "--no-mesh" not in sys.argv:
    mt = np.load(MT, allow_pickle=True)
    assert (mt["ids"] == exp.train["sample_id"].to_numpy()).all()
    MESH = mt["X"]
else:
    MESH = np.zeros((len(exp.train), 0))
print("mesh descriptors", MESH.shape[1])

rows = pd.concat([exp.train.assign(split="train")[["split", "group", "category", "sample_id"]],
                  exp.val.assign(split="val")[["split", "group", "category", "sample_id"]]])
rows.to_csv(OUT / "rows.csv", index=False)
n_tr = len(exp.train)


def base_values(table, cats):
    b0, b1 = np.empty(len(cats)), np.full(len(cats), np.nan)
    for i, c in enumerate(cats):
        v = table.values.get(c)
        if v is None: b0[i] = table.fallback
        elif len(v) == 1: b0[i] = next(iter(v.values()))
        else: b0[i], b1[i] = v[0], v[1]
    return b0, b1


gt = {r.sample_id: json.loads((DATA / "train" / r.group / r.category / r.sample_id / "gt.json").read_text())
      for r in exp.train.itertuples()}
full_table, _ = build_table(exp.train, cluster_labels(exp.train))
v0, v1 = base_values(full_table, exp.val["category"].to_numpy())
for f, (tr_idx, va_idx) in enumerate(exp.splits(0)):
    T, Q = exp.train.iloc[tr_idx].reset_index(drop=True), exp.train.iloc[va_idx].reset_index(drop=True)
    u = cluster_labels(T); table, ref_loo = build_table(T, u)
    r = T["ln"].to_numpy() - ref_loo
    ok = np.isfinite(r) & (np.abs(r) < GAP)
    mixed = T["category"].isin({c for c, v in table.values.items() if len(v) == 2}).to_numpy()
    dims = np.log(np.sort(np.array([gt[s]["bbox_mm"] for s in T["sample_id"]]), axis=1) / 10)
    fill = np.log(T["volume"].to_numpy() / np.prod(np.exp(dims), axis=1))
    aux = np.column_stack([dims, fill, MESH[tr_idx]])
    aux_mean, aux_std = aux.mean(0), aux.std(0) + 1e-6
    aux = (aux - aux_mean) / aux_std
    q0, q1 = base_values(table, Q["category"].to_numpy())
    np.savez(OUT / f"fold{f}.npz", tr_idx=tr_idx, va_idx=va_idx, r=np.nan_to_num(r), ok=ok, u=u.astype(np.float32),
             mixed=mixed, aux=aux.astype(np.float32), q_base0=q0, q_base1=q1, q_y=Q["volume"].to_numpy(),
             val_base0=v0, val_base1=v1, aux_mean=aux_mean, aux_std=aux_std)
    print(f"fold {f}: train {len(T)} held-out {len(Q)} aux {aux.shape} ok {ok.mean():.3f} mixed {mixed.mean():.3f}")
print("rows", len(rows), "train", n_tr, "val", len(exp.val))
