"""Kaggle (2x T4, internet on): IDOL (speridlabs/idol, Apache-2.0) single image -> 3D Gaussians, on the holdout photos
(and the training photos when the f3dr-trainphotos dataset is attached). One process per GPU.
Needs a Kaggle secret HF_TOKEN (read token of an account with access to facebook/dinov3-vith16plus-pretrain-lvd1689m).
The FLUX.2 autoencoder comes from black-forest-labs/FLUX.2-klein-4B (Apache-2.0, not gated) instead of FLUX.2-dev.
Output: /kaggle/working/idol_<set>.npz with ids, pts (n, 8192, 3) Gaussian centres (opacity > 0.1), camera (n, 4)."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

W = Path("/kaggle/working")
T = Path("/kaggle/temp")
T.mkdir(exist_ok=True)


def find_photo_sets():
    sets = {}
    hold = [p for p in Path("/kaggle/input").rglob("*.png") if "hold204" in str(p)]
    if hold:
        sets["hold"] = sorted(hold)
    bins = list(Path("/kaggle/input").rglob("train2042.bin"))
    if bins:
        if not (T / "train2042").exists():
            subprocess.run(["tar", "xf", str(bins[0]), "-C", str(T)], check=True)
        sets["train"] = sorted((T / "train2042").glob("*.png"))
    return sets


def encoders():
    """DINOv3 ViT-H+ from an attached private dataset when present (a local copy needs no token), else the gated repo;
    the FLUX.2 autoencoder from FLUX.2-klein-4B (Apache-2.0, not gated)."""
    enc = {"flux2_vae": "black-forest-labs/FLUX.2-klein-4B"}
    local = [p.parent for p in Path("/kaggle/input").rglob("config.json") if "dinov3-vith16plus" in str(p)]
    if local:
        enc["dinov3"] = str(local[0])
    return enc


def worker(set_name, shard, nshard):
    import torch
    from idol import IdolPipeline

    photos = find_photo_sets()[set_name][shard::nshard]
    out = W / f"idol_{set_name}_s{shard}.npz"
    pipe = IdolPipeline.from_pretrained("speridlabs/idol", dinov3_dtype="fp32",  # T4 has no bf16
                                        image_encoders=encoders())
    ids, pts, cams, secs = [], [], [], []
    rng = np.random.default_rng(0)
    for k, p in enumerate(photos):
        t1 = time.time()
        try:
            o = pipe(str(p), mode="synthetic", seed=0, steps=25, num_gaussians=65536)
            xyz = o.gaussian.xyz.float().cpu().numpy()
            op = o.gaussian.opacity.float().cpu().numpy().reshape(-1)
            keep = xyz[op > 0.1] if (op > 0.1).sum() >= 1024 else xyz
            pts.append(keep[rng.choice(len(keep), 8192, replace=len(keep) < 8192)].astype(np.float32))
            c = o.camera
            cams.append(np.array([c.get("yaw", np.nan), c.get("pitch", np.nan), c.get("radius", np.nan), c.get("fov", np.nan)], np.float32))
        except Exception as e:
            print("FAIL", p.stem, repr(e)[:300], flush=True)
            if not secs:
                import traceback
                traceback.print_exc()
            pts.append(np.full((8192, 3), np.nan, np.float32))
            cams.append(np.full(4, np.nan, np.float32))
        ids.append(p.stem)
        secs.append(time.time() - t1)
        if (k + 1) % 20 == 0 or k + 1 == len(photos):
            np.savez(out, ids=np.array(ids), pts=np.stack(pts), camera=np.stack(cams))
            print(f"{set_name} shard {shard}: {k + 1}/{len(photos)}  {np.mean(secs):.1f} s/image  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if len(sys.argv) > 1:  # child process: python run.py SET SHARD NSHARD
    worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    sys.exit()

t0 = time.time()
r = subprocess.run("pip install -q git+https://github.com/speridlabs/idol-sdk.git 2>&1 | tail -2", shell=True, capture_output=True, text=True)
print(r.stdout, flush=True)
if "dinov3" not in encoders():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        sys.exit(f"Attach the dinov3-vith16plus dataset, or add a Kaggle secret HF_TOKEN ({e!r})")
print("encoders:", encoders(), flush=True)
sets = find_photo_sets()
print({k: len(v) for k, v in sets.items()}, flush=True)
# warm the Hugging Face cache once so the two workers do not download the same files at the same time
subprocess.run([sys.executable, "-c", f"from idol import IdolPipeline; IdolPipeline.from_pretrained('speridlabs/idol', device='cpu', image_encoders={encoders()!r})"])
for name in ("hold", "train"):
    if name not in sets:
        continue
    procs = [subprocess.Popen([sys.executable, __file__, name, str(k), "2"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))) for k in (0, 1)]
    for p in procs:
        p.wait()
    parts = [np.load(W / f"idol_{name}_s{k}.npz") for k in (0, 1) if (W / f"idol_{name}_s{k}.npz").exists()]
    if parts:
        np.savez(W / f"idol_{name}.npz", **{key: np.concatenate([z[key] for z in parts]) for key in ("ids", "pts", "camera")})
        print(name, "saved", sum(len(z["ids"]) for z in parts), f"after {(time.time() - t0) / 60:.1f} min", flush=True)
print("ALL DONE", flush=True)
