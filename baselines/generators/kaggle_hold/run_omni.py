"""Kaggle (2x T4, internet on): Hunyuan3D-Omni on the 204 holdout photos, controlled by RH-Mix5's own holdout cloud.

Arms, in this order (each arm finishes on all photos before the next starts, so a time-out still leaves whole arms):
  bbox   bounding box of RH-Mix5's cloud: extents [length, height, width] / largest
  voxel  RH-Mix5's cloud as points, Omni quantises them to its 16^3 grid
  point  RH-Mix5's cloud as 4096 points
RH-Mix5's cloud is put in Omni's frame by principal axes: largest -> x, smallest -> y (up), middle -> z.
Licence: Tencent Hunyuan 3D Omni Community License (not valid in the EU, UK, South Korea). Weights: tencent/Hunyuan3D-Omni
(not gated). Masks: ZhengPeng7/BiRefNet (MIT).
Needs the dataset made by make_dataset.py (hold204/<id>.png, controls.npz). Output: /kaggle/working/omni_<arm>_s<k>.npz with
ids, pts (n, 8192, 3) surface points in Omni's frame and unit, secs; NaN marks a failed photo. Last line: RUN DONE.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

W, T = Path("/kaggle/working"), Path("/kaggle/temp")
REPO_SHA = "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8"
ARMS = ("bbox", "voxel", "point")
BUDGET_H = float(os.environ.get("BUDGET_H", "7.3"))  # stop starting new photos after this many hours (account 2 has 8.25 h)
T0 = float(os.environ.get("RUN_T0", time.time()))  # workers share the run's start time, so the budget counts setup too


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if r.returncode:
        print((r.stdout + r.stderr)[-3000:], flush=True)
        raise RuntimeError(f"failed: {cmd[:200]}")


def find(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        raise RuntimeError(f"{name} not found under /kaggle/input: attach the f3dr-gen-hold dataset")
    return hits[0]


def setup() -> str:
    T.mkdir(exist_ok=True)
    venv, py = T / "venv", str(T / "venv/bin/python")
    sh("pip install -q uv")
    sh(f"uv venv {venv} --python 3.10 --python-preference only-managed -q")
    pip = f"uv pip install --python {py} -q"
    sh(f"{pip} torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124")
    sh(f"{pip} transformers==4.46.0 diffusers==0.30.0 accelerate==1.1.1 pytorch-lightning==1.9.5 timm==1.0.20 torchdiffeq==0.2.5 "
       "numpy==1.24.4 scipy==1.14.1 einops==0.8.0 opencv-python-headless==4.10.0.84 scikit-image==0.24.0 trimesh==4.8.3 "
       "pymeshlab==2023.12.post3 omegaconf==2.3.0 pyyaml==6.0.2 tqdm huggingface_hub==0.26.5 kornia safetensors pillow")
    log("environment ready")
    repo = T / "omni"
    if not repo.exists():
        sh(f"git clone -q https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git {repo} && cd {repo} && git checkout -q {REPO_SHA}")
    weights = T / "omni_weights"
    sh(f"""{py} -c "from huggingface_hub import snapshot_download as s; s('tencent/Hunyuan3D-Omni', local_dir='{weights}', """
       f"""allow_patterns=['*.json', 'model/pytorch_model.bin', 'vae/pytorch_model.bin', 'cond_encoder/*'], max_workers=8)" """)
    sh(f"""{py} -c "from transformers import AutoModelForImageSegmentation as A; A.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)" """)
    log("code and weights ready")
    return py


def to_omni_frame(cloud):
    """Principal axes of the cloud: largest -> x, smallest -> y (up), middle -> z; proper rotation."""
    import numpy as np
    c = cloud - cloud.mean(0)
    _, vecs = np.linalg.eigh(c.T @ c)  # ascending variance
    axes = vecs[:, [2, 0, 1]]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return c @ axes


def normalize(p, scale):
    """Omni's normalize_mesh on points: centre the bounds, largest extent -> [-scale, scale]."""
    lo, hi = p.min(0), p.max(0)
    return (p - (lo + hi) / 2) / (hi - lo).max() * 2 * scale


def worker(gpu: int, ngpu: int) -> None:
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    sys.path.insert(0, str(T / "omni"))
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

    if gpu > 0:  # load one after the other: each load holds the 11.6 GB fp32 checkpoint in main memory for a while
        while not (T / "loaded0").exists():
            time.sleep(10)
    pipe = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(str(T / "omni_weights"), device="cuda", dtype=torch.float16)
    seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).float().eval().cuda()
    (T / f"loaded{gpu}").touch()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with np.load(find("controls.npz")) as z:
        ids, clouds = [str(i) for i in z["ids"]], z["pts"].astype(np.float64)
    photos = find("hold204")
    mine = list(range(gpu, len(ids), ngpu))
    rgba_dir = T / f"rgba{gpu}"
    rgba_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    def rgba(sid: str) -> str:
        path = rgba_dir / f"{sid}.png"
        if not path.exists():
            im = Image.open(photos / f"{sid}.png").convert("RGB")
            with torch.no_grad():
                m = seg(tf(im)[None].cuda())[-1].sigmoid().cpu()[0, 0]
            m = transforms.functional.resize(m[None], im.size[::-1])[0].numpy()
            Image.fromarray(np.dstack([np.asarray(im), (m * 255).clip(0, 255).astype(np.uint8)]), "RGBA").save(path)
        return str(path)

    def control(arm: str, cloud):
        p = to_omni_frame(cloud)
        if arm == "bbox":
            ext = np.ptp(p, 0)
            return {"bbox": torch.tensor(ext / ext.max(), dtype=torch.float16, device="cuda").view(1, 1, 3)}
        if arm == "voxel":
            return {"voxel": torch.tensor(normalize(p, 0.9999), dtype=torch.float16, device="cuda")[None]}
        q = normalize(p, 0.98)[rng.choice(len(p), 4096, replace=False)]
        return {"point": torch.tensor(q, dtype=torch.float16, device="cuda")[None]}

    for arm in ARMS:
        out = W / f"omni_{arm}_s{gpu}.npz"
        got, pts, secs = [], [], []
        for k, n in enumerate(mine):
            if (time.time() - T0) / 3600 > BUDGET_H:
                log(f"gpu {gpu}: time budget reached in arm {arm} after {k} photos")
                break
            t1 = time.time()
            try:
                res = pipe(image=rgba(ids[n]), **control(arm, clouds[n]), num_inference_steps=50, octree_resolution=256,
                           mc_level=0, guidance_scale=4.5, generator=torch.Generator("cuda").manual_seed(1234))
                mesh = DegenerateFaceRemover()(FloaterRemover()(res["shapes"][0][0]))
                pts.append(np.asarray(trimesh.sample.sample_surface(mesh, 8192, seed=0)[0], np.float32))
            except Exception as e:  # keep going; NaN marks the failure
                print(f"FAIL {arm} {ids[n]} {e!r}"[:400], flush=True)
                pts.append(np.full((8192, 3), np.nan, np.float32))
            got.append(ids[n])
            secs.append(time.time() - t1)
            if k < 2 or (k + 1) % 10 == 0 or k + 1 == len(mine):
                np.savez(out, ids=np.array(got), pts=np.stack(pts), secs=np.array(secs))
                log(f"gpu {gpu} {arm}: {k + 1}/{len(mine)}, {np.mean(secs):.1f} s/photo, peak GPU "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            torch.cuda.empty_cache()
        if got:
            np.savez(out, ids=np.array(got), pts=np.stack(pts), secs=np.array(secs))
    log(f"gpu {gpu}: WORKER DONE")


def main() -> None:
    py = setup()
    import torch
    ngpu = max(1, torch.cuda.device_count())
    me = find("run_omni.py")
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", RUN_T0=str(T0))
        procs.append(subprocess.Popen([py, str(me), "worker", str(g), str(ngpu)], env=env))
    codes = [p.wait() for p in procs]
    log(f"worker exit codes {codes}")
    print("RUN DONE" if not any(codes) else "RUN FAILED", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
