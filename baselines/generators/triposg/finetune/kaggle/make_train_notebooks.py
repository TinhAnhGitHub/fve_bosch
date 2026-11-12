"""Generate the Kaggle notebooks that run TripoSG + the epoch-4 LoRA + the size head on the 1,838 training photos.

usage: python make_train_notebooks.py [--selftest] [--owner NAME] [--infer OWNER/SLUG] [--photos OWNER/SLUG] [--parts 1,2,3]
Writes train_infer/partN (default owner anhanhanhanh) or train_infer_<owner>/partN next to this file. Each folder has
f3dr-tg-train-pN.ipynb and kernel-metadata.json. Push one at a time:
    kaggle kernels push -p train_infer/part1 --accelerator NvidiaTeslaT4
--infer is the dataset with code.tar, size.pt and lora.pt. --photos is the dataset with the training photos, in either layout:
  * <dataset>/train/train/<group>/<category>/<id>/rgb_industrial.png  (f3dr-tg-raw: our tar of data/f3d/train, unpacked by Kaggle)
  * a tar named train2042.bin that unpacks to train2042/<id>.png       (eiyuumiru/f3dr-trainphotos)
--selftest checks the generated notebooks and the shard and photo logic without Kaggle (standard library only).

The 24 shards of about 77 photos are rows[s::24] of the manifest's train rows (finetune/predict.py --split train --shard s
--num-shards 24). A notebook runs one worker per GPU and each worker does 4 shards in turn, so a finished shard is on disk even
if the session dies later. Merge all 24 shards afterwards with finetune/merge.py --expect 1838.
"""
import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OWNER = "anhanhanhanh"

# Pure helpers, used verbatim in the notebook and exercised by --selftest.
SHARDS_SRC = r'''NUM_PARTS, CHUNKS = 3, 4  # 3 notebooks x 2 GPU workers x 4 chunks = 24 shards of about 77 photos
NUM_SHARDS = NUM_PARTS * 2 * CHUNKS


def part_shards(part):
    """Shards that notebook `part` (1..3) does. Shard s belongs to worker s % 6; the parts take the worker pairs (0, 1), (2, 3), (4, 5)."""
    return [s for s in range(NUM_SHARDS) if (s % (NUM_PARTS * 2)) // 2 == part - 1]


def train_photos(paths):
    """Nested layout <dataset>/train/train/<group>/<category>/<id>/rgb_industrial.png: keep the paths whose inner folder is train."""
    return [p for p in paths if p.parents[3].name == "train"]


def data_root(photos):
    """The folder predict.py needs as --data: it holds train/<group>/<category>/<id>/rgb_industrial.png (the outer train folder)."""
    return photos[0].parents[4]


def link_photos(rows, flat_dir, root):
    """Flat layout: rows are (id, image path) pairs. Make root/<image path> a symlink to flat_dir/<id>.png; return the ids with no photo."""
    missing = []
    for sid, image in rows:
        src = Path(flat_dir) / f"{sid}.png"
        if not src.exists():
            missing.append(sid)
            continue
        dst = Path(root) / image
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)
    return missing
'''

MD_INTRO = r'''# LoRA inference on the training photos, part __PART__ of 3

TripoSG + the epoch-4 LoRA + our size head, run on the 1,838 training photos (the 204 holdout photos and the 511 public-test photos are done elsewhere). This notebook does one third: 8 of the 24 shards of about 77 photos, one worker per GPU.

* **Add data:** `__INFER__` (code, `size.pt`, `lora.pt`) and `__PHOTOS__` (the photos). **Internet on.** **Accelerator: GPU T4 x2.**
* **Time:** about 4 hours if a T4 takes about 50 s per photo (that figure is from an earlier TripoSG run; this notebook has not run on Kaggle yet).
* **Output:** `lora_train_sNN.npz` per shard (`ids`, `pts` in mm, `unit` clouds of radius 1, `radius_mm` from the size head), one log per shard, and `part__PART___summary.json`. The last printed line is `RUN DONE` or `RUN FAILED`.
* **Caution:** the LoRA and the size head were trained on these objects, so the outputs are in-sample: better than they would be on photos the model has not seen.
* Merge all 24 shards afterwards with `finetune/merge.py <shards> --output lora_train.npz --expect 1838`.
'''

CONFIG = r'''PART = __PART__  # 1, 2 or 3: which third of the training photos this notebook does
ARM = "lora"

import json
import os
import subprocess
import sys
import time
from pathlib import Path

__SHARDS_SRC__
W, T = Path("/kaggle/working"), Path("/kaggle/temp")
T.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(T / "hf")  # one Hugging Face cache for the pre-download and for both workers
VENV = T / "venv"
PY = str(VENV / "bin" / "python")
t0 = time.time()
FT = DATA = SIZE = SHAPE = FLAT = None  # set by stage()


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
    """Kaggle may unpack tars in datasets; accept the file or its unpacked tree."""
    hits = sorted(Path("/kaggle/input").rglob(name))
    return hits[0] if hits else None
'''

STAGE = r'''def stage():
    code = T / "f3dr"
    if not (code / "baselines/generators/triposg/finetune/predict.py").exists():
        tar = find_input("code.tar")
        if tar is not None and tar.is_file():
            code.mkdir(exist_ok=True)
            sh(f"tar -xf {tar} -C {code}")
        else:
            src = find_input("predict.py").parents[3]  # .../code/baselines
            sh(f"mkdir -p {code} && cp -r {src} {code}/")
    ft = code / "baselines/generators/triposg/finetune"
    size, shape = find_input("size.pt"), find_input(f"{ARM}.pt")
    if size is None or shape is None:
        raise RuntimeError("size.pt or lora.pt missing: add the dataset with the code and checkpoints")
    nested = train_photos(sorted(Path("/kaggle/input").rglob("rgb_industrial.png")))
    if nested:  # <dataset>/train/train/<group>/<category>/<id>/rgb_industrial.png
        if len(nested) != 2042:  # 1,838 training + 204 holdout photos live in the same folder
            raise RuntimeError(f"expected 2042 photos, found {len(nested)}")
        data, flat = data_root(nested), None
    else:  # a tar of flat <id>.png files, unpacked to train2042/
        blob = find_input("train2042.bin")
        if blob is None:
            raise RuntimeError("no training photos under /kaggle/input: add the photos dataset")
        if not (T / "train2042").exists():
            sh(f"tar xf {blob} -C {T}")
        flat = T / "train2042"
        data = T / "f3d"
        log(f"{len(list(flat.glob('*.png')))} flat photos in {flat}")
    log(f"code {ft}, photos {'flat' if flat else 'nested'} under {flat or data}, size {size}, shape {shape}")
    return ft, data, size, shape, flat
'''

SETUP = r'''def cuda12_home():
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
    log(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip().replace("\n", " | "))
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
    # download the pinned weights once, before two workers race for them
    sh(f"cd {FT} && {PY} -c \"import torch; from huggingface_hub import snapshot_download as s; "
       f"m = torch.load('{SIZE}', map_location='cpu', weights_only=False)['metadata']; print(s(m['model'], revision=m['revision']))\"")


def prepare_photos():
    """Make DATA / <image path> exist for every training row; with the flat photo tar, link the files into place."""
    rows_file = T / "train_rows.json"
    sh(f"cd {FT} && {PY} -c \"import json, torch; m = torch.load('{SIZE}', map_location='cpu', weights_only=False)['metadata']; "
       f"json.dump([[r['id'], r['image']] for r in m['rows'] if r['split'] == 'train'], open('{rows_file}', 'w'))\"")
    rows = json.loads(rows_file.read_text())
    if FLAT is not None:
        missing = link_photos(rows, FLAT, DATA)
    else:
        missing = [sid for sid, image in rows if not (DATA / image).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(rows)} training photos not found, for example {missing[:3]}")
    log(f"{len(rows)} training photos in place under {DATA}")
'''

RUN = r'''def run():
    shards = part_shards(PART)
    workers = max(1, min(2, int(sh("nvidia-smi -L | wc -l").strip() or 1)))
    log(f"part {PART}: shards {shards} on {workers} GPU worker(s)")
    procs = []
    for g in range(workers):
        cmds = []
        for s in shards[g::workers]:
            out, logf = W / f"{ARM}_train_s{s:02d}.npz", W / f"{ARM}_train_s{s:02d}.log"
            cmds.append(f"[ -f {out} ] || {PY} predict.py --upstream {T / 'TripoSG'} --data {DATA} --size {SIZE} --shape {SHAPE} "
                        f"--split train --steps 50 --guidance 7 --precision fp16 --dense-depth 7 --hierarchical-depth 8 "
                        f"--shard {s} --num-shards {NUM_SHARDS} --output {out} > {logf} 2>&1 || exit 1")
        procs.append(subprocess.Popen(["bash", "-c", "; ".join(cmds)], cwd=FT,
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        done = sum(sum("radius=" in line for line in open(f, errors="replace")) for f in sorted(W.glob(f"{ARM}_train_s*.log")))
        log(f"photos done so far: {done}; shard files: {len(list(W.glob(f'{ARM}_train_s*.npz')))} of {len(shards)}")
    for g, p in enumerate(procs):
        if p.returncode:
            for f in sorted(W.glob(f"{ARM}_train_s*.log"))[-2:]:
                print(f.name, open(f, errors="replace").read()[-2000:], flush=True)
            raise RuntimeError(f"GPU worker {g} failed")
'''

VALIDATE = r'''def validate():
    import numpy as np

    n_train = len(json.loads((T / "train_rows.json").read_text()))
    shards = part_shards(PART)
    expect = sum(1 for i in range(n_train) if i % NUM_SHARDS in set(shards))
    seen, files = set(), []
    for s in shards:
        z = np.load(W / f"{ARM}_train_s{s:02d}.npz", allow_pickle=False)
        ids = z["ids"].tolist()
        if z["pts"].shape != (len(ids), 8192, 3) or not np.isfinite(z["pts"]).all() or set(ids) & seen:
            raise RuntimeError(f"shard {s} is malformed")
        seen |= set(ids)
        files.append({"shard": s, "objects": len(ids), "radius_mm_median": float(np.median(z["radius_mm"]))})
    if len(seen) != expect:
        raise RuntimeError(f"{len(seen)} objects, expected {expect}")
    (W / f"part{PART}_summary.json").write_text(json.dumps({"part": PART, "objects": len(seen), "train_rows": n_train,
                                                              "num_shards": NUM_SHARDS, "files": files}, indent=1))
    log(f"{len(seen)} clouds in {len(shards)} shard files, all finite, ids unique")
'''

MAIN = r'''try:
    FT, DATA, SIZE, SHAPE, FLAT = stage()
    setup()
    prepare_photos()
    log(f"setup done; part {PART}")
    run()
    validate()
    log("RUN DONE")
except Exception as e:
    log(f"RUN FAILED: {e!r}")
    raise
'''


def cells_for(part, infer, photos):
    config = CONFIG.replace("__SHARDS_SRC__", SHARDS_SRC)
    cells = [("markdown", MD_INTRO), ("code", config), ("code", STAGE), ("code", SETUP), ("code", RUN), ("code", VALIDATE),
             ("code", MAIN)]
    return [(kind, src.replace("__PART__", str(part)).replace("__INFER__", infer).replace("__PHOTOS__", photos))
            for kind, src in cells]


def notebook(cells):
    def cell(kind, src):
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c

    return {"cells": [cell(k, s) for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def metadata(owner, part, infer, photos):
    return {"id": f"{owner}/f3dr-tg-train-p{part}", "title": f"f3dr-tg-train-p{part}", "code_file": f"f3dr-tg-train-p{part}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [infer, photos], "competition_sources": [], "kernel_sources": []}


def folder_for(owner, part):
    return HERE / ("train_infer" if owner == DEFAULT_OWNER else f"train_infer_{owner}") / f"part{part}"


def generate(owner, infer, photos, parts):
    for part in parts:
        out = folder_for(owner, part)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"f3dr-tg-train-p{part}.ipynb").write_text(
            json.dumps(notebook(cells_for(part, infer, photos)), indent=1) + "\n", encoding="utf-8")
        (out / "kernel-metadata.json").write_text(json.dumps(metadata(owner, part, infer, photos), indent=1) + "\n", encoding="utf-8")
        print("wrote", out)


def selftest(owner, parts, n_train=1838):
    ns = {"Path": Path}  # the notebook's config cell imports Path before the helpers
    exec(SHARDS_SRC, ns)  # noqa: S102
    part_shards, num = ns["part_shards"], ns["NUM_SHARDS"]
    everything = sorted(s for p in (1, 2, 3) for s in part_shards(p))
    assert everything == list(range(num)), "the three parts must cover shards 0..23 exactly once"
    counts = [sum(1 for i in range(n_train) if i % num in set(part_shards(p))) for p in (1, 2, 3)]
    assert sum(counts) == n_train, counts
    print("shards per part:", [part_shards(p) for p in (1, 2, 3)])
    print(f"photos per part for {n_train} training photos:", counts)
    p = Path("/kaggle/input/f3dr-tg-raw/train/train/bakery/bagel/mf3d_bagel_001/rgb_industrial.png")
    other = Path("/kaggle/input/f3dr-tg-infer/eval/eval/bakery/bagel/mf3d_bagel_006/rgb_industrial.png")
    photos = ns["train_photos"]([other, p])
    assert photos == [p], photos
    assert ns["data_root"](photos) == Path("/kaggle/input/f3dr-tg-raw/train"), ns["data_root"](photos)
    assert ns["data_root"](photos) / "train/bakery/bagel/mf3d_bagel_001/rgb_industrial.png" == p  # what predict.py opens
    print("nested photo filter and data root OK")
    with tempfile.TemporaryDirectory() as tmp:  # the flat layout: train2042/<id>.png linked into train/<group>/<category>/<id>/
        flat, root = Path(tmp) / "train2042", Path(tmp) / "f3d"
        flat.mkdir()
        for sid in ("mf3d_bagel_001", "omni_apple_002"):
            (flat / f"{sid}.png").write_bytes(b"png")
        rows = [("mf3d_bagel_001", "train/bakery/bagel/mf3d_bagel_001/rgb_industrial.png"),
                ("omni_apple_002", "train/fruits/apple/omni_apple_002/rgb_industrial.png"),
                ("omni_pear_003", "train/fruits/pear/omni_pear_003/rgb_industrial.png")]
        missing = ns["link_photos"](rows, flat, root)
        assert missing == ["omni_pear_003"], missing
        assert ns["link_photos"](rows, flat, root) == ["omni_pear_003"]  # a second run changes nothing
        for sid, image in rows[:2]:
            assert (root / image).is_symlink() and (root / image).read_bytes() == b"png", image
    print("flat photo linking OK")
    for part in parts:
        folder = folder_for(owner, part)
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        nb = json.loads((folder / meta["code_file"]).read_text())
        assert meta["id"] == f"{owner}/f3dr-tg-train-p{part}" and meta["kernel_type"] == "notebook"
        assert meta["enable_gpu"] and meta["enable_internet"] and len(meta["dataset_sources"]) == 2, meta
        for k, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]), filename=f"part{part} cell {k}")
        text = json.dumps(nb)
        assert f"PART = {part}" in "".join(nb["cells"][1]["source"]) and "__PART__" not in text
        assert "__INFER__" not in text and "__PHOTOS__" not in text and meta["dataset_sources"][0] in text
        print(f"{owner} part{part}: {len(nb['cells'])} cells parse, metadata OK ({meta['id']}, inputs {meta['dataset_sources']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--owner", default=DEFAULT_OWNER)
    ap.add_argument("--infer", help="dataset with code.tar, size.pt, lora.pt (default OWNER/f3dr-tg-infer)")
    ap.add_argument("--photos", help="dataset with the training photos (default anhanhanhanh/f3dr-tg-raw)")
    ap.add_argument("--parts", default="1,2,3")
    a = ap.parse_args()
    infer = a.infer or f"{a.owner}/f3dr-tg-infer"
    photos = a.photos or f"{DEFAULT_OWNER}/f3dr-tg-raw"
    parts = [int(x) for x in a.parts.split(",")]
    if a.selftest:
        selftest(a.owner, parts)
    else:
        generate(a.owner, infer, photos, parts)
