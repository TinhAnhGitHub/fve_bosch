# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Old and new checkpoints on the fixed answers, split by whether the object's mesh changed."""
import sys, numpy as np, torch
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root / "baselines"))
from atlas import _load_cache, holdout, load_net, nearest, kabsch
dev = torch.device("cuda")
z, v = _load_cache("dinov3l"); val, _, _ = holdout(len(z["ids"]))
with np.load(root / "baselines/out/atlas/before_fixed_meshes/train.npz", allow_pickle=False) as o:
    assert o["ids"].tolist() == z["ids"].tolist(); old = o["pts"]
rad = lambda p: np.sqrt(((p - p.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
ra, rn = rad(old), rad(z["pts"])
zs = np.load(root / "baselines/out/scale/scale.npz"); uu = dict(zip(zs["ids"], zs["u"]))
flag = np.array([uu[i] != 0 for i in z["ids"]]); same = np.abs(rn / ra - 1) < 0.03
print(f"all training objects: flagged {int(flag.sum())}; not flagged but scale changed >3%: {int((~flag & ~same).sum())}; ratio new/old of those: median {np.median((rn / ra)[~flag & ~same]):.2f}, min {(rn / ra)[~flag & ~same].min():.2f}, max {(rn / ra)[~flag & ~same].max():.2f}", flush=True)
g = torch.tensor(z["pts"][val], dtype=torch.float32).to(dev)
f = torch.tensor(z["feats"][val], dtype=torch.float32); c = torch.tensor([v[x] for x in z["cats"][val]])
@torch.no_grad()
def strict(pred, t, iters=10):
    q = pred - pred.mean(0) + t.mean(0)
    for _ in range(iters):
        _, j = nearest(q, t); R, tr = kabsch(q, t[j]); q = q @ R.T + tr
    return float(nearest(q, t)[0].mean()), float(nearest(t, q)[0].mean())
groups = {"mesh unchanged": same[val] & ~flag[val], "not flagged, scale changed": ~same[val] & ~flag[val], "flagged oversized before": flag[val]}
res = {}
for name in sys.argv[2:]:
    net, _ = load_net(root / f"baselines/out/atlas/{name}.pt", dev)
    with torch.inference_mode():
        P = torch.cat([net(f[i:i+4].to(dev), c[i:i+4].to(dev)) for i in range(0, len(val), 4)])
    s = np.array([strict(P[k], g[k]) for k in range(len(val))]); res[name] = s.sum(1)
    for lab, m in groups.items():
        print(f"{name:12s} {lab:28s} n={int(m.sum()):3d}  CD {s[m].sum(1).mean():8.1f}  stray {s[m,0].mean():8.1f}  missing {s[m,1].mean():7.1f}", flush=True)
    print(f"{name:12s} all holdout n={len(val)}  CD {s.sum(1).mean():.1f}", flush=True)
a, b = sys.argv[2:4]; m = np.where(groups["mesh unchanged"])[0]
d = res[b] - res[a]; rng = np.random.default_rng(0); nd = d[m][rng.integers(0, len(m), (2000, len(m)))].mean(1)
print(f"RESULT unchanged meshes: {b} lower than {a} in {100*(nd<0).mean():.1f}% of resamples; mean change {d[m].mean():+.1f} (5-95% {np.percentile(nd,5):+.1f} to {np.percentile(nd,95):+.1f})", flush=True)
