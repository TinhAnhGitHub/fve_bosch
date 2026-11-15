"""Kaggle (2x T4, internet on): SAM 3D Objects (via Fast-SAM3D) on the holdout photos SAM 3D has not done yet.

Same install and generation as ../../sam3d (install.sh, gen_sam3d.py), with the gated weights read from a private dataset
instead of Hugging Face. One gen_sam3d.py process per GPU with --low-vram (a T4 has 16 GB; normal mode peaked at 17.6 GB).
Needs two datasets: f3dr-gen-hold (make_dataset.py: hold204/, sam3d_todo.txt, gen_sam3d.py, the two fix files) and
f3dr-sam3d-weights (the checkpoints/ folder of facebook/sam-3d-objects). Licence: Meta SAM License.
Output: /kaggle/working/sam3d_s<k>.npz (gen_sam3d.py's format) and its log. Last line: RUN DONE or RUN FAILED.
"""
import os
import subprocess
import time
from pathlib import Path

W, T = Path("/kaggle/working"), Path("/kaggle/temp")
REPO_SHA = "c76188fda755b761fe9255eabe87aaad39df829a"
T0 = time.time()


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
        raise RuntimeError(f"{name} not found under /kaggle/input")
    return hits[0]


def setup() -> str:
    T.mkdir(exist_ok=True)
    repo, venv = T / "Fast-SAM3D", T / ".venv-sam3d"
    py = str(venv / "bin/python")
    sh(f"git clone -q https://github.com/wlfeng0509/Fast-SAM3D.git {repo} && cd {repo} && git checkout -q {REPO_SHA}")
    sh("pip install -q uv")
    sh(f"uv venv {venv} --python 3.10 --python-preference only-managed -q")
    pip = f"uv pip install --python {py} -q"
    cu = "--index-url https://download.pytorch.org/whl/cu121"
    sh(f"{pip} torch==2.4.1 torchvision==0.19.1 {cu}")
    sh(f'{pip} "numpy<2" hydra-core==1.3.2 omegaconf loguru opencv-python-headless pillow trimesh plyfile optree easydict scipy '
       "imageio scikit-image timm==0.9.16 safetensors seaborn matplotlib plotly open3d==0.18.0 lightning==2.3.3 roma xatlas fvcore "
       "iopath einops transformers==4.46.3 huggingface_hub==0.26.5 pymeshfix point-cloud-utils==0.29.5 spconv-cu121==2.3.8 rtree "
       'pyrender ftfy jsonlines astor python-igraph pyvista kornia "setuptools<70"')
    sh(f"{pip} --no-deps xformers==0.0.28.post1 {cu}")
    sh(f"{pip} pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html")
    sh(f'{pip} "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/'
       'flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"')
    sh(f"{pip} kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html")
    sh(f'{pip} "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" '
       '"git+https://github.com/microsoft/MoGe.git"')
    sh(f"cd {repo} && ({py} patching/hydra || true) && uv pip install --python {py} -q -e . --no-deps")
    sh(f"{pip} --reinstall torch==2.4.1 torchvision==0.19.1 {cu}")
    sh(f'{pip} "numpy<2" "huggingface_hub==0.26.5"')
    sh(f"cp {find('fix_flow_matching_model.py')} {repo}/sam3d_objects/model/backbone/generator/flow_matching/model.py")
    sh(f"cp {find('fix_dino.py')} {repo}/sam3d_objects/model/backbone/dit/embedder/dino.py")
    sh(f"cp {find('gen_sam3d.py')} {T}/gen_sam3d.py")
    log("environment ready")
    # weights: keep the repo's yaml files (they include the accelerated variants), add the rest from the dataset
    ckpt = find("pipeline.yaml").parent
    hf = repo / "checkpoints/hf"
    hf.mkdir(parents=True, exist_ok=True)
    for f in ckpt.iterdir():
        if not (f.suffix == ".yaml" and (hf / f.name).exists()):
            (hf / f.name).unlink(missing_ok=True)
            (hf / f.name).symlink_to(f)
    moge = repo / "checkpoints/moge-vitl"
    sh(f"""{py} -c "from huggingface_hub import snapshot_download as s; s('Ruicheng/moge-vitl', local_dir='{moge}')" """)
    yaml = hf / "pipeline.yaml"
    if yaml.is_symlink():  # make it a real file before editing
        text = yaml.read_text()
        yaml.unlink()
        yaml.write_text(text)
    sh(f"sed -i 's#pretrained_model_name_or_path: .*moge-vitl/model.pt.*#pretrained_model_name_or_path: {moge}/model.pt#' {yaml}")
    sh(f"""{py} -c "from transformers import AutoModelForImageSegmentation as A; A.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)" """)
    sh(f'{py} -c "import torch, pytorch3d, kaolin, spconv, xformers, moge, kornia, astor"')
    log("weights ready")
    return py


def main() -> None:
    py = setup()
    import torch
    ngpu = max(1, torch.cuda.device_count())
    photos, todo = find("hold204"), find("sam3d_todo.txt")
    log(f"{len(todo.read_text().split())} photos to do on {ngpu} GPUs")
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
        logf = open(W / f"sam3d_s{g}.log", "w")
        procs.append(subprocess.Popen([py, str(T / "gen_sam3d.py"), str(W / f"sam3d_s{g}.npz"), str(photos), "--ids", str(todo),
                                       "--shard", f"{g}/{ngpu}", "--low-vram"], env=env, stdout=logf, stderr=subprocess.STDOUT))
        time.sleep(120)  # load one after the other
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        for g in range(ngpu):
            news = [l for l in (W / f"sam3d_s{g}.log").read_text().splitlines() if "saved" in l or "FAIL" in l or "peak" in l]
            log(f"gpu {g}: {news[-1] if news else '(no progress yet)'}")
    codes = [p.returncode for p in procs]
    for g in range(ngpu):
        print(f"--- gpu {g} log tail ---\n" + "\n".join((W / f"sam3d_s{g}.log").read_text().splitlines()[-15:]), flush=True)
    log(f"exit codes {codes}")
    print("RUN DONE" if not any(codes) else "RUN FAILED", flush=True)


if __name__ == "__main__":
    main()
