"""TripoSG shapes for folders of photos, on the Colab L4.   usage: python gen_tg.py OUT.npz DIR [DIR ...]

Saves ids (n,) and pts (n, 8192, 3): surface samples of each generated mesh, in TripoSG's own frame and scale.
Masks come from ZhengPeng7/BiRefNet (MIT, float32) and are passed as an alpha channel, so TripoSG's own
RMBG-1.4 (non-commercial) is never loaded. Weights: VAST-AI/TripoSG (MIT).
"""
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("TG_ROOT", "/content/TripoSG")
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/scripts")
import cv2
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

import triposg.models.autoencoders.autoencoder_kl_triposg  # noqa: F401  (diffusers 0.3x+ only loads classes whose module is already imported)
import triposg.models.transformers.triposg_transformer  # noqa: F401
import triposg.schedulers.scheduling_rectified_flow  # noqa: F401
from image_process import prepare_image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

dev = "cuda"
STEPS = int(os.environ.get("TG_STEPS", 30))  # 50 is the default; 30 and a 256^3 mesh grid are plenty for 8192 surface points
out_path = Path(sys.argv[1])
photos = sorted(p for d in sys.argv[2:] for p in Path(d).glob("*.png"))
k, n = map(int, os.environ.get("TG_SHARD", "0/1").split("/"))
photos = photos[k::n]  # shard k of n, one per GPU
print(len(photos), "photos", flush=True)

seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).float().to(dev).eval()
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
w = snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=os.environ.get("TG_W", "/content/w_triposg"))
pipe = TripoSGPipeline.from_pretrained(w).to(dev, torch.float16)


@torch.no_grad()
def rgba(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    m = seg(tf(im)[None].to(dev))[-1].sigmoid().cpu()[0, 0]
    a = np.asarray(transforms.functional.resize(m[None], im.size[::-1])[0] * 255).clip(0, 255).astype(np.uint8)
    tmp = f"/tmp/tmp_rgba_{os.getpid()}.png"
    cv2.imwrite(tmp, cv2.cvtColor(np.dstack([np.asarray(im), a]), cv2.COLOR_RGBA2BGRA))
    return tmp


ids, pts, secs, done = [], [], [], {}
if out_path.exists():  # resume
    z = np.load(out_path, allow_pickle=False)
    done = dict(zip(z["ids"].tolist(), z["pts"]))
t0 = time.time()
for k, p in enumerate(photos):
    sid = p.stem
    if sid in done:
        ids.append(sid); pts.append(done[sid]); continue
    t1 = time.time()
    try:
        img = prepare_image(rgba(p), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
        with torch.no_grad():
            verts, faces = pipe(image=img, generator=torch.Generator(device=dev).manual_seed(0), num_inference_steps=STEPS, guidance_scale=7.0, dense_octree_depth=7, hierarchical_octree_depth=8, flash_octree_depth=8).samples[0]
        mesh = trimesh.Trimesh(np.asarray(verts, np.float32), np.ascontiguousarray(faces), process=False)
        cloud = np.asarray(trimesh.sample.sample_surface(mesh, 8192, seed=0)[0], np.float32)
    except Exception as e:
        print("FAIL", sid, repr(e)[:200], flush=True)
        if not any(np.isnan(x).any() for x in pts):
            import traceback; traceback.print_exc()
        cloud = np.full((8192, 3), np.nan, np.float32)
    ids.append(sid); pts.append(cloud); secs.append(time.time() - t1)
    torch.cuda.empty_cache()
    if (k + 1) % 20 == 0 or k + 1 == len(photos):
        np.savez(out_path, ids=np.array(ids), pts=np.stack(pts))
        print(f"{k + 1}/{len(photos)}  {np.mean(secs):.1f} s/image  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
print("GEN DONE", flush=True)
