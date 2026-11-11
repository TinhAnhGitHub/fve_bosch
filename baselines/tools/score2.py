# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Fixed data: strict CD on all holdout objects, candidate against a baseline. usage: score2.py root base cand"""
import sys, numpy as np, torch
from pathlib import Path
root, base, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]; sys.path.insert(0, str(root / "baselines"))
import os
from atlas import _load_cache, cache_path, holdout, load_net, nearest, kabsch, right_handed
dev = torch.device("cuda")
z, _ = _load_cache("dinov3l"); val, _, _ = holdout(len(z["ids"]))
P_ = right_handed(z["pts"][val], z["ids"][val].tolist()) if os.environ.get("RH") else z["pts"][val]  # RH=1: score against the real, unmirrored shapes
gt = torch.tensor(P_, dtype=torch.float32).to(dev)
@torch.no_grad()
def strict(pred, t, iters=10):
    q = pred - pred.mean(0) + t.mean(0)
    for _ in range(iters):
        _, j = nearest(q, t); R, tr = kabsch(q, t[j]); q = q @ R.T + tr
    return float(nearest(q, t)[0].mean()), float(nearest(t, q)[0].mean())
res = {}
for name in (base, cand):
    net, blob = load_net(root / f"baselines/out/atlas/{name}.pt", dev)
    zz, _ = _load_cache(blob.get("encoder", "dinov2b"))  # each checkpoint reads its own encoder's tokens
    if blob.get("extra"):  # extra token cache appended channel-wise, as in training
        zz = dict(zz); zz["feats"] = np.concatenate([zz["feats"], np.load(cache_path(blob["extra"]))["feats"]], -1)
    f = torch.tensor(zz["feats"][val], dtype=torch.float32); c = torch.tensor([blob["vocab"][x] for x in zz["cats"][val].tolist()])
    with torch.inference_mode():
        P = torch.cat([net(f[i:i+4].to(dev), c[i:i+4].to(dev)) for i in range(0, len(val), 4)])
    s = np.array([strict(P[k], gt[k]) for k in range(len(val))]); res[name] = s.sum(1)
    print(f"{name:20s} strict all {s.sum(1).mean():5.1f} (stray {s[:,0].mean():5.1f}, missing {s[:,1].mean():5.1f})  median {np.median(s.sum(1)):5.1f}  n={len(val)}", flush=True)
    del net; torch.cuda.empty_cache()
if base != cand:
    d = res[cand] - res[base]; rng = np.random.default_rng(0); nd = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    print(f"RESULT {cand}: lower than {base} in {100*(nd<0).mean():.1f}% of resamples; mean change {d.mean():+.1f} mm^2 (5-95% {np.percentile(nd,5):+.1f} to {np.percentile(nd,95):+.1f})", flush=True)
