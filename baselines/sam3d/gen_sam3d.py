"""Run SAM 3D (Fast-SAM3D, accelerated) on food photos and save each object's shape.

usage (with the .venv-sam3d python, from this folder):
  python gen_sam3d.py OUT.npz IMAGES [IMAGES ...] [--low-vram] [--ids ids.txt] [--shard 0/2]

IMAGES is a folder searched recursively. The challenge layout <group>/<category>/<sample_id>/rgb_industrial.png is
recognised (the id is the sample folder name); otherwise the id is the file name without .png.
--ids keeps only the sample ids listed in a text file (one per line).
--low-vram keeps every model in main memory and moves each one to the GPU only for its own step (depth map, rough
  shape, detailed shape, mesh). Slower, but the peak GPU memory drops. Untested on 12 GB: watch the "peak" column.

Saves, per photo (resumes from OUT.npz if it exists, saving every 10 photos):
  pts_mesh (8192, 3)  surface samples of the output mesh, SAM 3D's own frame and scale
  pts_gs   (8192, 3)  Gaussian centres, object frame
  scale (3,), rotation (4,), translation (3,)  SAM 3D's layout (object -> camera). Its size is NOT usable on these renders.
Masks come from ZhengPeng7/BiRefNet (MIT). Weights: facebook/sam-3d-objects (SAM License, gated), MoGe (MIT).
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "Fast-SAM3D"
os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")  # notebook/inference.py sets CUDA_HOME from it
os.chdir(REPO)
sys.path.insert(0, str(REPO / "notebook"))
sys.path.insert(0, str(REPO))
from unittest.mock import MagicMock

# Rendering-only libraries, not needed to produce shapes (nvdiffrast is NVIDIA non-commercial): stubbed, not installed.
for _m in ("gsplat", "nvdiffrast", "nvdiffrast.torch", "diffoctreerast", "diff_gaussian_rasterization", "torchsparse", "vox2seq"):
    sys.modules.setdefault(_m, MagicMock())

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from fft.fft2d import calculate_hfer_robust
from inference import Inference

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("images", nargs="+", type=Path)
ap.add_argument("--low-vram", action="store_true")
ap.add_argument("--ids", type=Path, help="text file of sample ids to keep")
ap.add_argument("--shard", default="0/1", help="k/n: process every n-th photo starting at k (several GPUs)")
cli = ap.parse_args()


def find_photos(roots):
    out = {}
    for r in roots:
        for p in sorted(r.rglob("*.png")):
            sid = p.parent.name if p.stem == "rgb_industrial" else p.stem
            out[sid] = p
    return out


photos = find_photos(cli.images)
if cli.ids:
    keep = set(cli.ids.read_text().split())
    photos = {k: v for k, v in photos.items() if k in keep}
k, n = map(int, cli.shard.split("/"))
items = sorted(photos.items())[k::n]
print(len(items), "photos", flush=True)

args = argparse.Namespace(tag="hf", seed=0, ss_faster_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=0.5, slat_thresh=1.5, slat_warmup=3,
                          slat_token_ratio=0.1, mesh_spectral_threshold_low=0.5, mesh_spectral_threshold_high=0.7, enable_ss_faster=True,
                          enable_slat_token=True, enable_mesh_aggregation=True, enable_acceleration=True, enable_taylor=False, enable_easy=False)
config = OmegaConf.load("checkpoints/hf/pipeline.yaml")
config.workspace_dir = "checkpoints/hf"
config["ss_generator_config_path"] = "ss_generator_faster.yaml"
config["slat_generator_config_path"] = "slat_generator_faster.yaml"
if cli.low_vram:
    config["compile_model"] = False
inference = Inference(config, compile=False, args=args)
pipe = inference._pipeline

if cli.low_vram:
    # Which models each step needs; everything else waits in main memory.
    STEPS = {
        "compute_pointmap": lambda: [pipe.depth_model.model],  # the MoGe wrapper is not a module; its network is
        "sample_sparse_structure": lambda: [pipe.models["ss_generator"], pipe.models["ss_decoder"]],
        "sample_slat": lambda: [pipe.models["slat_generator"]],
        "decode_slat": lambda: [pipe.models[m] for m in ("slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_gs_4")],
    }
    every = lambda: [m for m in [pipe.depth_model.model, *pipe.models.values(), *getattr(pipe, "condition_embedders", {}).values()] if isinstance(m, torch.nn.Module)]

    def on_gpu_only(name, fn):
        def wrapped(*a, **kw):
            need = STEPS[name]()
            for m in every():
                if not any(m is x for x in need):
                    m.to("cpu")
            torch.cuda.empty_cache()
            for m in need:
                m.to("cuda")
            return fn(*a, **kw)
        return wrapped

    for name in STEPS:
        setattr(pipe, name, on_gpu_only(name, getattr(pipe, name)))
    for m in every():
        m.to("cpu")
    torch.cuda.empty_cache()
    print("low-vram: models wait in main memory between steps", flush=True)

seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).float().eval()
seg_dev = "cuda"
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def mask_of(im: Image.Image) -> np.ndarray:
    seg.to(seg_dev)
    m = seg(tf(im)[None].to(seg_dev))[-1].sigmoid().cpu()[0, 0]
    if cli.low_vram:
        seg.to("cpu")
        torch.cuda.empty_cache()
    return np.asarray(transforms.functional.resize(m[None], im.size[::-1])[0]) > 0.5


KEYS = ("pts_mesh", "pts_gs", "scale", "rotation", "translation")
EMPTY = {"pts_mesh": (8192, 3), "pts_gs": (8192, 3), "scale": (3,), "rotation": (4,), "translation": (3,)}
done = {}
if cli.out.exists():
    z = np.load(cli.out, allow_pickle=False)
    done = {i: {key: z[key][j] for key in KEYS} for j, i in enumerate(z["ids"].tolist())}
store = {key: [] for key in KEYS}
ids, secs = [], []
t0 = time.time()
for j, (sid, p) in enumerate(items):
    if sid in done:
        r = done[sid]
    else:
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            im = Image.open(p).convert("RGB")
            m = mask_of(im)
            mp = f"/tmp/sam3d_mask_{os.getpid()}.png"
            Image.fromarray((m * 255).astype(np.uint8)).save(mp)
            if hasattr(inference, "get_hfer"):
                inference.get_hfer(calculate_hfer_robust(mp))
            if hasattr(inference, "get_params"):
                inference.get_params(args)
            out = inference(np.asarray(im), m, seed=0)
            mesh = out["glb"]
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            g = out["gaussian"][0].get_xyz.detach().float().cpu().numpy()
            r = {"pts_mesh": np.asarray(trimesh.sample.sample_surface(mesh, 8192, seed=0)[0], np.float32),
                 "pts_gs": g[np.random.default_rng(0).choice(len(g), 8192, replace=len(g) < 8192)].astype(np.float32),
                 "scale": out["scale"].reshape(-1)[:3].float().cpu().numpy(),
                 "rotation": out["rotation"].reshape(-1)[:4].float().cpu().numpy(),
                 "translation": out["translation"].reshape(-1)[:3].float().cpu().numpy()}
        except Exception as e:  # keep going; NaN marks the failure
            print("FAIL", sid, repr(e)[:300], flush=True)
            if not secs:
                import traceback
                traceback.print_exc()
            r = {key: np.full(shape, np.nan, np.float32) for key, shape in EMPTY.items()}
        secs.append(time.time() - t1)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{sid}: {secs[-1]:.1f} s, peak GPU {peak:.1f} GB", flush=True)
        torch.cuda.empty_cache()
    ids.append(sid)
    for key in KEYS:
        store[key].append(r[key])
    if (j + 1) % 10 == 0 or j + 1 == len(items):
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cli.out, ids=np.array(ids), **{key: np.stack(v) for key, v in store.items()})
        print(f"{j + 1}/{len(items)} saved, {np.mean(secs) if secs else 0:.1f} s/image, {(time.time() - t0) / 60:.1f} min", flush=True)
print("GEN DONE", flush=True)
