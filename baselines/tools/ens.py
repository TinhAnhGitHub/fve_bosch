# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Handedness-aware point-mix ensembles. usage: ens.py root holdout|eval|holdout_save

Each member's cloud is aligned to the reference member (the first one, right-handed) per
object: the best of the 8 axis sign flips (mirrors included), then rigid ICP; then the
points are pooled with an equal share per member. holdout: strict score vs real-handed
shapes for each plan. eval: writes one zip per plan into f3dr_submissions/<plan>/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

root, mode = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / "baselines"))
from atlas import ENCODERS, _load_cache, encoder, holdout, image_tensor, image_tokens, kabsch, load_net, mix, nearest, right_handed, rows
from geom import pack

dev = torch.device("cuda")
A = root / "baselines/out/atlas"
SUB = root.parent / "f3dr_submissions"
CK = {  # name -> checkpoint; leaderboard score of the model alone (or its mix) in the comment
    "v27": A / "cloud/s_posrh_s1_ema.pt",  # 52.89, right-handed: the reference
    "v26": A / "cloud/s_rh_s2_ema.pt",  # 54.77, right-handed
    "tds": SUB / "v22_mix3_ds/pl_tds_s0_ema.pt",  # v22 member (v22 = 46.35)
    "ds": SUB / "v22_mix3_ds/pl_ds_s0_ema.pt",  # v22 member
    "r1": SUB / "v22_mix3_ds/pl_r_s1_ema.pt",  # v22 member
    "v24": A / "n_pos_ema.pt",  # 55.35
    "v21": A / "pl_t_s0_ema.pt",  # 55.74
    "rh0": A / "cloud/s_rh_s0_ema.pt",  # right-handed seed 0, local 33.9
    "prh0": A / "cloud/s_posrh_s0_ema.pt",  # pos + right-handed seed 0, local 34.2
    "l1": A / "cloud/s_l1_s0_ema.pt",  # L1 Chamfer seed 0, 40 epochs, local 33.5
    "l1rh": A / "cloud/s_l1rh_s2_ema.pt",  # L1 Chamfer + right-handed seed 2, local 34.0
    "nocat": A / "cloud/s_nocat_s0_ema.pt",  # no category, image query: 54.01 alone
}
PLANS = {
    "e5_v22plus_rh2": ["v27", "v26", "tds", "ds", "r1"],
    "e7_all_strong": ["v27", "v26", "tds", "ds", "r1", "v24", "v21"],
    "e4_v22plus_v27": ["v27", "tds", "ds", "r1"],
    "e7_v22plus_rh4": ["v27", "v26", "rh0", "prh0", "tds", "ds", "r1"],
    "e4_rh_only": ["v27", "v26", "rh0", "prh0"],
    "e6_mix5_l1": ["v27", "v26", "tds", "ds", "r1", "l1"],
    "e7_mix5_l1x2": ["v27", "v26", "tds", "ds", "r1", "l1", "l1rh"],
    "e6_mix5_nocat": ["v27", "v26", "tds", "ds", "r1", "nocat"],
    "v22_itself": ["tds", "ds", "r1"],  # the 46.35 mix, aligned the same way, for reference
}
SIGNS = torch.tensor([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=torch.float32)


@torch.no_grad()
def cd(q, t):
    return float(nearest(q, t)[0].mean() + nearest(t, q)[0].mean())


@torch.no_grad()
def align_to(p, ref, iters=10):
    """Best sign flip (mirrors allowed) of p about its centre, then rigid ICP onto ref."""
    pc = p - p.mean(0)
    best, best_d = None, float("inf")
    for s in SIGNS.to(p.device):
        q = pc * s + ref.mean(0)
        d = cd(q[::8], ref[::8])
        if d < best_d:
            best, best_d = q, d
    q = best
    for _ in range(iters):
        _, j = nearest(q, ref)
        R, tr = kabsch(q, ref[j])
        q = q @ R.T + tr
    return q


@torch.no_grad()
def strict(pred, t, iters=10):
    q = pred - pred.mean(0) + t.mean(0)
    for _ in range(iters):
        _, j = nearest(q, t)
        R, tr = kabsch(q, t[j])
        q = q @ R.T + tr
    return cd(q, t)


def member_clouds(names, tokens_fn, cats, n_obj):
    out = {}
    for n in names:
        net, blob = load_net(CK[n], dev)
        tok = tokens_fn(blob.get("encoder", "dinov2b"))
        vocab = blob["vocab"]
        c = torch.tensor([vocab.get(x, vocab.get("<unk>", 0)) for x in cats], dtype=torch.long)
        with torch.inference_mode():
            out[n] = torch.cat([net(tok[i : i + 8].to(dev), c[i : i + 8].to(dev)).float() for i in range(0, n_obj, 8)]).cpu()
        print(f"{n}: {CK[n].name}", flush=True)
        del net
        torch.cuda.empty_cache()
    return out


names = sorted({n for p in PLANS.values() for n in p}, key=list(CK).index)
if mode == "holdout":
    z, _ = _load_cache("dinov3l")
    val, _, _ = holdout(len(z["ids"]))
    gt = torch.tensor(right_handed(z["pts"][val], z["ids"][val].tolist()), dtype=torch.float32)
    feats = torch.tensor(z["feats"][val], dtype=torch.float32)
    C = member_clouds(names, lambda enc: feats, z["cats"][val].tolist(), len(val))
    for n in names:
        s = np.array([strict(C[n][i].to(dev), gt[i].to(dev)) for i in range(len(val))])
        print(f"{n:6s} alone  {s.mean():6.2f} median {np.median(s):5.1f}", flush=True)
    for plan, mem in PLANS.items():
        s = []
        for i in range(len(val)):
            ref = C[mem[0]][i].to(dev)
            parts = [ref] + [align_to(C[m][i].to(dev), ref) for m in mem[1:]]
            s.append(strict(torch.tensor(mix([p.cpu().numpy() for p in parts])).to(dev), gt[i].to(dev)))
        s = np.array(s)
        print(f"{plan:18s} {s.mean():6.2f} median {np.median(s):5.1f}  ({'+'.join(mem)})", flush=True)
elif mode == "holdout_save":
    # ens.py root holdout_save OUT.npz [plan]: the plan's mixed clouds for the 204 holdout objects (default RH-Mix5), built exactly as
    # in `holdout` and `eval`, saved with the strict score of each object, so they can be scored and combined with other clouds.
    out_path, plan = Path(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 else "e5_v22plus_rh2")
    mem = PLANS[plan]
    z, _ = _load_cache("dinov3l")
    val, _, _ = holdout(len(z["ids"]))
    gt = torch.tensor(right_handed(z["pts"][val], z["ids"][val].tolist()), dtype=torch.float32)
    feats = torch.tensor(z["feats"][val], dtype=torch.float32)
    C = member_clouds(sorted(set(mem), key=list(CK).index), lambda enc: feats, z["cats"][val].tolist(), len(val))
    clouds, scores = [], []
    for i in range(len(val)):
        ref = C[mem[0]][i].to(dev)
        parts = [ref] + [align_to(C[m][i].to(dev), ref) for m in mem[1:]]
        cloud = mix([p.cpu().numpy() for p in parts])
        clouds.append(cloud)
        scores.append(strict(torch.tensor(cloud).to(dev), gt[i].to(dev)))
    np.savez(out_path, ids=z["ids"][val], pts=np.stack(clouds).astype(np.float32), strict=np.array(scores))
    print(f"{plan}: strict mean {np.mean(scores):.2f} median {np.median(scores):.1f} over {len(val)} objects -> {out_path}", flush=True)
else:
    paths = rows("val")
    enc = encoder(dev, "dinov3l")
    with torch.inference_mode():
        tok = torch.cat([image_tokens(enc, torch.stack([image_tensor(p, ENCODERS["dinov3l"]) for p in paths[i : i + 8]]).to(dev)).cpu() for i in range(0, len(paths), 8)])
    del enc
    torch.cuda.empty_cache()
    C = member_clouds(names, lambda e: tok, [p.parent.parent.name for p in paths], len(paths))
    for plan in sys.argv[3:]:
        mem = PLANS[plan]
        clouds = {}
        for i, p in enumerate(paths):
            ref = C[mem[0]][i].to(dev)
            parts = [ref] + [align_to(C[m][i].to(dev), ref) for m in mem[1:]]
            clouds[p.parent.name] = mix([q.cpu().numpy() for q in parts])
        out = SUB / f"{os.environ.get('ENS_TAG', 'v28')}_{plan}" / "preds_geometry.zip"
        pack(out, clouds)
        print(f"{plan}: {len(clouds)} clouds -> {out}", flush=True)
