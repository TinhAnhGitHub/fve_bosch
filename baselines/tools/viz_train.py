# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh", "matplotlib"]
# ///
"""Run the RH-Mix5 mix on every training photo and compare with the true shape.   usage: viz_train.py ROOT OUT_DIR

The members are the same as `ens.py` plan e5_v22plus_rh2 (leaderboard 43.85). They saw the 1,838 training objects
during training, so those show how well the model fits; the 204 holdout objects (column `holdout` = 1) are unseen.
Per object: OUT_DIR/<split>_<group>_<category>_<id>.png (photo, truth, prediction, overlay, two views),
OUT_DIR/ply/<id>/{ground_truth,prediction}.ply aligned like the grader (centre + 10 rigid ICP steps),
and OUT_DIR/index.csv with the squared Chamfer distance (mm^2) per object.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / "baselines"))
from atlas import _load_cache, holdout, kabsch, load_net, mix, nearest, right_handed, rows

dev = torch.device("cuda")
A = root / "baselines/out/atlas"
SUB = root.parent / "f3dr_submissions"
MEMBERS = {  # RH-Mix5; the first is the right-handed reference the others are aligned to
    "v27": A / "cloud/s_posrh_s1_ema.pt",
    "v26": A / "cloud/s_rh_s2_ema.pt",
    "tds": SUB / "v22_mix3_ds/pl_tds_s0_ema.pt",
    "ds": SUB / "v22_mix3_ds/pl_ds_s0_ema.pt",
    "r1": SUB / "v22_mix3_ds/pl_r_s1_ema.pt",
}
SIGNS = torch.tensor([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=torch.float32, device=dev)

z, _ = _load_cache("dinov3l")
ids = z["ids"].tolist()
val, _, _ = holdout(len(ids))
is_hold = np.zeros(len(ids), bool)
is_hold[val] = True
gt_all = right_handed(z["pts"], ids)
feats = torch.tensor(z["feats"], dtype=torch.float32)
photos = {p.parent.name: p for p in rows("train")}


@torch.no_grad()
def cd(q, t):
    return float(nearest(q, t)[0].mean() + nearest(t, q)[0].mean())


@torch.no_grad()
def icp(q, t, iters=10):
    for _ in range(iters):
        _, j = nearest(q, t)
        R, tr = kabsch(q, t[j])
        q = q @ R.T + tr
    return q


@torch.no_grad()
def align_to(p, ref):
    pc = p - p.mean(0)
    best, bd = None, float("inf")
    for s in SIGNS:
        q = pc * s + ref.mean(0)
        d = cd(q[::8], ref[::8])
        if d < bd:
            best, bd = q, d
    return icp(best, ref)


preds = {}
for name, ck in MEMBERS.items():
    net, blob = load_net(ck, dev)
    vocab = blob["vocab"]
    c = torch.tensor([vocab.get(x, vocab.get("<unk>", 0)) for x in z["cats"].tolist()], dtype=torch.long)
    with torch.inference_mode():
        preds[name] = torch.cat([net(feats[i : i + 16].to(dev), c[i : i + 16].to(dev)).float().cpu() for i in range(0, len(ids), 16)])
    print("predicted", name, flush=True)
    del net
    torch.cuda.empty_cache()

(out / "ply").mkdir(parents=True, exist_ok=True)


def write_ply(path, P):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(fh, P, fmt="%.4f")


table = []
for k, sid in enumerate(ids):
    t = torch.tensor(gt_all[k], dtype=torch.float32, device=dev)
    ref = preds["v27"][k].to(dev)
    parts = [ref] + [align_to(preds[m][k].to(dev), ref) for m in list(MEMBERS)[1:]]
    p = torch.tensor(mix([q.cpu().numpy() for q in parts]), dtype=torch.float32, device=dev)
    q = icp(p - p.mean(0) + t.mean(0), t)
    score = cd(q, t)
    T, Q = t.cpu().numpy(), q.cpu().numpy()
    d = out / "ply" / sid
    d.mkdir(exist_ok=True)
    write_ply(d / "ground_truth.ply", T)
    write_ply(d / "prediction.ply", Q)
    ph = photos.get(sid)
    group, cat = (ph.parts[-4], ph.parts[-3]) if ph else ("?", "?")
    split = "holdout" if is_hold[k] else "train"
    c0 = T.mean(0)
    _, _, vt = np.linalg.svd(T - c0, full_matrices=False)
    G, P2 = (T - c0) @ vt.T, (Q - c0) @ vt.T
    lim = np.abs(np.concatenate([G, P2])).max() * 1.05
    rg = float(np.sqrt(((T - c0) ** 2).sum(1).mean()))
    rp = float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean()))
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    if ph:
        ax[0, 0].imshow(Image.open(ph).resize((400, 400)))
    ax[0, 0].set_title(f"{sid}\n{group} / {cat}  [{split}]", fontsize=9)
    ax[1, 0].axis("off")
    ax[1, 0].text(0, 0.5, f"squared Chamfer {score:.1f} mm^2\nRMS radius: truth {rg:.1f} mm, prediction {rp:.1f} mm\n\ngrey = truth, coloured = prediction", fontsize=10)
    for r, (i, j, view) in enumerate([(0, 1, "top"), (0, 2, "side")]):
        for c_, (title, pts, grey) in enumerate([("ground truth", G, None), ("RH-Mix5 prediction", P2, None), ("overlay", P2, G)], start=1):
            a = ax[r, c_]
            if grey is not None:
                a.scatter(grey[::4, i], grey[::4, j], s=0.3, c="0.75")
            a.scatter(pts[::4, i], pts[::4, j], s=0.3, c=pts[::4, 3 - i - j], cmap="viridis")
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{title} ({view})", fontsize=9)
    ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
    fig.tight_layout()
    fig.savefig(out / f"{split}_{group}_{cat}_{sid}.png", dpi=55)
    plt.close(fig)
    table.append((sid, split, group, cat, f"{score:.2f}", f"{rg:.1f}", f"{rp:.1f}"))
    if (k + 1) % 200 == 0:
        print(f"{k + 1}/{len(ids)}", flush=True)

with open(out / "index.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sample_id", "split", "group", "category", "chamfer_mm2", "rms_truth_mm", "rms_pred_mm"])
    w.writerows(table)
s = np.array([float(r[4]) for r in table])
h = np.array([r[1] == "holdout" for r in table])
print(f"seen in training ({(~h).sum()}): mean {s[~h].mean():.1f} median {np.median(s[~h]):.1f} | holdout ({h.sum()}): mean {s[h].mean():.1f} median {np.median(s[h]):.1f}")
print("DONE", flush=True)
