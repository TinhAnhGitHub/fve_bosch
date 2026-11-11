"""Kaggle (2x T4): TripoSG shapes for the 204 holdout photos, one half per GPU. Output: /kaggle/working/hold_tg.npz"""
import os, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np

W = Path("/kaggle/working"); T = Path("/kaggle/temp"); T.mkdir(exist_ok=True)
data = next(p.parent for p in Path("/kaggle/input").rglob("gen_tg.py"))
sh = lambda c: print(subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout[-1500:], flush=True)
t0 = time.time()
pngs = [p for p in Path("/kaggle/input").rglob("*.png") if "hold204" in str(p)]  # Kaggle unpacks tar files in datasets
HOLD = pngs[0].parent if pngs else None
if HOLD is None:
    sh(f"cd {T} && tar xf {data}/hold204.tar")
    HOLD = T / "hold204"
print("holdout photos:", len(list(HOLD.glob("*.png"))), "in", HOLD, flush=True)
sh(f"cd {T} && git clone -q https://github.com/VAST-AI-Research/TripoSG.git && ls TripoSG | head -3")
sh("pip install -q diffusers einops omegaconf scikit-image peft jaxtyping typeguard trimesh opencv-python-headless kornia timm 2>&1 | tail -2")
sh("pip install -q setuptools wheel ninja 2>&1 | tail -1; TORCH_CUDA_ARCH_LIST=7.5 pip install diso --no-build-isolation 2>&1 | tail -2; python -c 'import diso, torch; print(\"diso ok\", torch.__version__)' 2>&1 | tail -1")
print(f"setup {(time.time() - t0) / 60:.1f} min", flush=True)
env = dict(os.environ, TG_ROOT=str(T / "TripoSG"), TG_W=str(T / "w_triposg"), TG_STEPS="30")
subprocess.run([sys.executable, "-c", f"from huggingface_hub import snapshot_download as s; s('VAST-AI/TripoSG', local_dir='{T}/w_triposg'); s('ZhengPeng7/BiRefNet')"], env=env)
procs = [subprocess.Popen([sys.executable, str(data / "gen_tg.py"), str(W / f"hold_s{k}.npz"), str(HOLD)],
                          env=dict(env, CUDA_VISIBLE_DEVICES=str(k), TG_SHARD=f"{k}/2"), stdout=open(W / f"gen_s{k}.log", "w"), stderr=subprocess.STDOUT) for k in (0, 1)]
for p in procs:
    p.wait()
sh(f"grep -E 's/image|FAIL|Error' {W}/gen_s0.log {W}/gen_s1.log | tail -8")
parts = [np.load(W / f"hold_s{k}.npz") for k in (0, 1) if (W / f"hold_s{k}.npz").exists()]
np.savez(W / "hold_tg.npz", ids=np.concatenate([z["ids"] for z in parts]), pts=np.concatenate([z["pts"] for z in parts]))
print("saved", sum(len(z["ids"]) for z in parts), "clouds; total", f"{(time.time() - t0) / 60:.1f} min", flush=True)
