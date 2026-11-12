"""Kaggle (2x T4, internet on): TripoSG + size head on the 511 public-test photos, one shard per GPU.

Private dataset f3dr-tg-infer holds: code.tar (baselines/geom.py + baselines/generators/triposg/finetune),
eval.tar (data/f3d/eval), size.pt, and one <arm>.pt per fine-tuned shape model (none for "pretrained").
Same environment as the Colab runs: Python 3.12 venv, torch 2.5.1+cu124, finetune/requirements.txt,
TripoSG at the pinned commit, diso built for sm_75. Same predict settings: fp16, 50 steps, guidance 7, depth 7/8, seed 42.
Output: /kaggle/working/<arm>_eval.zip (+ .npz), logs in /kaggle/working/*.log. The last line is RUN DONE or RUN FAILED.
"""
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ARMS = ["lora"]  # set by the push step; arm "pretrained" needs no <arm>.pt
W, T = Path("/kaggle/working"), Path("/kaggle/temp")
T.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(T / "hf")  # one Hugging Face cache for the pre-download in setup() and for both workers
VENV = T / "venv"
PY = str(VENV / "bin" / "python")
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, tail=1500):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    out = (r.stdout + r.stderr)[-tail:]
    if r.returncode:
        print(out, flush=True)
        raise RuntimeError(f"failed ({r.returncode}): {cmd[:200]}")
    return out


def find_input(name):
    """Kaggle may unpack tars in datasets; accept the tar or its unpacked tree."""
    hits = sorted(Path("/kaggle/input").rglob(name))
    return hits[0] if hits else None


def stage():
    code = T / "f3dr"
    if not (code / "baselines/generators/triposg/finetune/predict.py").exists():
        tar = find_input("code.tar")
        if tar is not None and tar.is_file():
            code.mkdir(exist_ok=True)
            sh(f"tar -xf {tar} -C {code}")
        else:
            src = find_input("predict.py").parents[3]  # .../baselines
            sh(f"mkdir -p {code} && cp -r {src} {code}/")
    data = T / "f3d"
    pngs = sorted(Path("/kaggle/input").rglob("rgb_industrial.png"))
    if pngs and pngs[0].parents[3].name == "eval":
        data = pngs[0].parents[4]
    elif not (data / "eval").exists():
        data.mkdir(exist_ok=True)
        sh(f"tar -xf {find_input('eval.tar')} -C {data}")
    n = len(list((data / "eval").glob("*/*/*/rgb_industrial.png")))
    if n != 511:
        raise RuntimeError(f"expected 511 public-test photos, found {n} under {data}")
    log(f"code {code}, photos {n} under {data}")
    return code / "baselines/generators/triposg/finetune", data


def cuda12_home():
    """torch 2.5.1 is built for CUDA 12.4 but the image's /usr/local/cuda can be 13.x, and torch refuses to build extensions against a
    different major version. Use a CUDA 12 toolkit that is already installed, else install one from NVIDIA's apt repo."""
    have = sh("ls -d /usr/local/cuda-12* 2>/dev/null | head -1").strip()
    if have:
        return have
    script = r"""
set -e
. /etc/os-release
apt-get update -qq >/dev/null 2>&1 || true
PKGS="cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev cuda-nvtx cuda-profiler-api libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev"
# a version counts only when EVERY package has an installation candidate: the repo can list cuda-nvcc-12-4 without its libraries
have_all() { for p in $PKGS; do c=$(apt-cache policy $p-$1 2>/dev/null | sed -n 's/^ *Candidate: //p'); { [ -n "$c" ] && [ "$c" != "(none)" ]; } || return 1; done; return 0; }
find_v() { for v in 12-4 12-5 12-6 12-8 12-9; do have_all $v && { echo $v; return 0; }; done; return 0; }
v=$(find_v)
if [ -z "$v" ]; then
  n=$(echo $VERSION_ID | tr -d .)
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$n/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
  dpkg -i /tmp/ck.deb >/dev/null
  apt-get update -qq >/dev/null 2>&1 || true
  v=$(find_v)
fi
if [ -z "$v" ]; then
  echo "no complete CUDA 12 toolkit in the apt repo"; head -2 /etc/os-release; apt-cache policy cuda-nvcc-12-6 libcublas-dev-12-6 | head -8; exit 1
fi
apt-get install -y -qq --no-install-recommends $(for p in $PKGS; do echo $p-$v; done) >/tmp/apt.log 2>&1 || { tail -5 /tmp/apt.log; exit 1; }
echo /usr/local/cuda-$(echo $v | tr - .)
"""
    return sh(script, tail=300).strip().splitlines()[-1]


def setup():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    sh("pip install -q uv")
    sh(f"uv venv -q --python 3.12 --python-preference only-managed {VENV}")
    uvp = f"uv pip install -q --python {PY}"
    sh(f"{uvp} torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124")
    sh(f"{uvp} -r {FT}/requirements.txt setuptools wheel ninja")
    if not (T / "TripoSG").exists():
        sh(f"git clone -q https://github.com/VAST-AI-Research/TripoSG.git {T}/TripoSG")
    sh(f"git -C {T}/TripoSG checkout -q fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c")
    nvcc = cuda12_home()
    log(sh(f"{nvcc}/bin/nvcc --version | tail -1").strip())
    sh(f"CUDA_HOME={nvcc} PATH={nvcc}/bin:$PATH TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4 "
       f"uv pip install --python {PY} --no-build-isolation diso", tail=600)
    log(sh(f"cd /tmp && {PY} -c \"import torch, diso; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs, diso ok')\"").strip())
    # Download the pinned weights once, before two processes race for them.
    sh(f"cd {FT} && {PY} -c \"import torch; from huggingface_hub import snapshot_download as s; "
       f"m = torch.load('{SIZE}', map_location='cpu', weights_only=False)['metadata']; print(s(m['model'], revision=m['revision']))\"")


def run_arm(arm):
    shape = [] if arm == "pretrained" else ["--shape", str(find_input(f"{arm}.pt"))]
    shards = [W / f"{arm}_eval_s{k}.npz" for k in (0, 1)]
    procs = []
    for k in (0, 1):
        if shards[k].exists():
            continue
        cmd = [PY, "predict.py", "--upstream", str(T / "TripoSG"), "--data", str(DATA), "--size", str(SIZE), *shape,
               "--split", "eval", "--steps", "50", "--guidance", "7", "--precision", "fp16",
               "--dense-depth", "7", "--hierarchical-depth", "8", "--shard", str(k), "--num-shards", "2",
               "--output", str(shards[k])]
        procs.append(subprocess.Popen(cmd, cwd=FT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k), HF_HOME=str(T / "hf")),
                                      stdout=open(W / f"{arm}_s{k}.log", "w"), stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        done = [sum("radius=" in l for l in open(W / f"{arm}_s{k}.log", errors="replace")) for k in (0, 1)]
        log(f"{arm}: objects done {done[0]} + {done[1]} of 511")
    for k, p in zip((0, 1), procs):
        if p.returncode:
            print(open(W / f"{arm}_s{k}.log", errors="replace").read()[-3000:], flush=True)
            raise RuntimeError(f"{arm} shard {k} failed")
    out = W / f"{arm}_eval.npz"
    log(sh(f"cd {FT} && {PY} merge.py {shards[0]} {shards[1]} --output {out} --expect 511").strip())
    with zipfile.ZipFile(out.with_suffix(".zip")) as z:
        names = [n for n in z.namelist() if n.endswith(".ply")]
    log(f"{arm}: zip entries {len(names)}, e.g. {names[:2]}")
    for s in shards:
        s.unlink()


try:
    FT, DATA = stage()
    SIZE = find_input("size.pt")
    setup()
    log(f"setup done; arms {ARMS}")
    for arm in ARMS:
        run_arm(arm)
    log("RUN DONE")
except Exception as e:
    log(f"RUN FAILED: {e!r}")
    raise
