"""Run several F3DR experiments in parallel on a cloud GPU box (Kaggle or Colab), then score each.

usage: python run_exps.py WORK DATA CODE name1 "flags1" name2 "flags2" ...
  WORK  writable folder (outputs land in WORK/out)
  DATA  folder with train.npz, train_dinov3l.npz, train_mirror.npz (and train_dinov3l384.npz if used)
  CODE  folder with atlas.py, geom.py, score2.py, pl_t_s0_ema.pt
Runs are spread round-robin over the visible GPUs. Every result is scored against pl_t_s0_ema on the
real-handed shapes (RH=1). WORK/out/summary.txt collects the score lines.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

work, data, code = map(Path, sys.argv[1:4])
runs = list(zip(sys.argv[4::2], sys.argv[5::2]))
root = work / "f3dr"
atlas = root / "baselines" / "out" / "atlas"
q = atlas / "queue"
q.mkdir(parents=True, exist_ok=True)
for f in ("atlas.py", "geom.py"):
    shutil.copy(code / f, root / "baselines" / f)
shutil.copy(code / "score2.py", q / "score2.py")
shutil.copy(code / "pl_t_s0_ema.pt", atlas / "pl_t_s0_ema.pt")
for f in data.glob("*.npz"):
    if not (atlas / f.name).exists():
        os.symlink(f, atlas / f.name)
out = work / "out"
out.mkdir(exist_ok=True)
gpus = torch.cuda.device_count()
print(f"{gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(gpus)]}; runs {runs}", flush=True)
common = "--target mm --no-flip-loss --ema 0.999".split()
env = dict(os.environ, HF_HUB_OFFLINE="1", RH="1", PYTHONUNBUFFERED="1")

procs = {}
for k, (name, flags) in enumerate(runs):
    e = dict(env, CUDA_VISIBLE_DEVICES=str(k % max(gpus, 1)))
    log = open(out / f"{name}.train.log", "w")
    cmd = [sys.executable, str(root / "baselines" / "atlas.py"), "train", *common, *flags.split(), "--ckpt", str(atlas / f"{name}.pt")]
    procs[name] = (subprocess.Popen(cmd, cwd=root, env=e, stdout=log, stderr=subprocess.STDOUT), e)
    print(time.strftime("%H:%M"), "START", name, flags, "on GPU", e["CUDA_VISIBLE_DEVICES"], flush=True)
    time.sleep(20)  # stagger the cache loads so RAM peaks do not add up

summary = open(out / "summary.txt", "a")
while procs:
    for name, (p, e) in list(procs.items()):
        if p.poll() is None:
            continue
        del procs[name]
        print(time.strftime("%H:%M"), "END", name, "rc", p.returncode, flush=True)
        if p.returncode != 0:
            summary.write(f"FAILED {name} rc {p.returncode}\n"); summary.flush()
            continue
        for n in (name, name + "_ema"):
            shutil.copy(atlas / f"{n}.pt", out / f"{n}.pt")
            r = subprocess.run([sys.executable, str(q / "score2.py"), str(root), "pl_t_s0_ema", n], cwd=root, env=e, capture_output=True, text=True)
            (out / f"{n}.score.log").write_text(r.stdout + r.stderr)
            lines = [l for l in r.stdout.splitlines() if l.startswith((n + " ", "RESULT"))]
            summary.write("\n".join(lines) + "\n"); summary.flush()
            print("\n".join(lines), flush=True)
    time.sleep(30)
print("ALL DONE", flush=True)
