"""Generate the three Kaggle notebooks that run TripoSG + the epoch-4 LoRA + the size head on the 1,838 training photos.

usage: python make_train_notebooks.py [--selftest]
Writes train_infer/part1, part2 and part3 next to this file. Each folder has f3dr-tg-train-pN.ipynb and kernel-metadata.json.
Push one at a time (the account's Kaggle GPU quota resets 2026-11-14 00:00 UTC):
    kaggle kernels push -p train_infer/part1 --accelerator NvidiaTeslaT4
--selftest checks the generated notebooks and the shard and photo-path logic without Kaggle (standard library only).

The 24 shards of about 77 photos are rows[s::24] of the manifest's train rows (finetune/predict.py --split train --shard s
--num-shards 24). A notebook runs one worker per GPU and each worker does 4 shards in turn, so a finished shard is on disk even
if the session dies later. Merge all 24 shards afterwards with finetune/merge.py --expect 1838.
"""
import ast
import json
import sys
from pathlib import Path

OWNER = "anhanhanhanh"
HERE = Path(__file__).resolve().parent

# Pure helpers, used verbatim in the notebook and exercised by --selftest.
SHARDS_SRC = r'''NUM_PARTS, CHUNKS = 3, 4  # 3 notebooks x 2 GPU workers x 4 chunks = 24 shards of about 77 photos
NUM_SHARDS = NUM_PARTS * 2 * CHUNKS


def part_shards(part):
    """Shards that notebook `part` (1..3) does. Shard s belongs to worker s % 6; the parts take the worker pairs (0, 1), (2, 3), (4, 5)."""
    return [s for s in range(NUM_SHARDS) if (s % (NUM_PARTS * 2)) // 2 == part - 1]


def train_photos(paths):
    """The raw dataset is <dataset>/train/train/<group>/<category>/<id>/rgb_industrial.png: keep the paths whose inner folder is train."""
    return [p for p in paths if p.parents[3].name == "train"]


def data_root(photos):
    """The folder predict.py needs as --data: it holds train/<group>/<category>/<id>/rgb_industrial.png (the outer train folder)."""
    return photos[0].parents[4]
'''

MD_INTRO = r'''# LoRA inference on the training photos, part __PART__ of 3

TripoSG + the epoch-4 LoRA + our size head, run on the 1,838 training photos (the 204 holdout photos and the 511 public-test photos are done elsewhere). This notebook does one third: 8 of the 24 shards of about 77 photos, one worker per GPU.

* **Add data:** `anhanhanhanh/f3dr-tg-infer` (code, `size.pt`, `lora.pt`) and `anhanhanhanh/f3dr-tg-raw` (the photos). **Internet on.** **Accelerator: GPU T4 x2.**
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
FT = DATA = SIZE = SHAPE = None  # set by stage()


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
    photos = train_photos(sorted(Path("/kaggle/input").rglob("rgb_industrial.png")))
    if not photos:
        raise RuntimeError("no training photos under /kaggle/input: add the dataset anhanhanhanh/f3dr-tg-raw")
    if len(photos) != 2042:  # 1,838 training + 204 holdout photos live in the same folder
        raise RuntimeError(f"expected 2042 photos, found {len(photos)}")
    ft, data = code / "baselines/generators/triposg/finetune", data_root(photos)
    size, shape = find_input("size.pt"), find_input(f"{ARM}.pt")
    if size is None or shape is None:
        raise RuntimeError("size.pt or lora.pt missing: add the dataset anhanhanhanh/f3dr-tg-infer")
    log(f"code {ft}, {len(photos)} photos under {data}, size {size}, shape {shape}")
    return ft, data, size, shape
'''

SETUP = r'''def setup():
    log(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip().replace("\n", " | "))
    sh("pip install -q uv")
    sh(f"uv venv -q --python 3.12 --python-preference only-managed {VENV}")
    uvp = f"uv pip install -q --python {PY}"
    sh(f"{uvp} torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124")
    sh(f"{uvp} -r {FT}/requirements.txt setuptools wheel ninja")
    if not (T / "TripoSG").exists():
        sh(f"git clone -q https://github.com/VAST-AI-Research/TripoSG.git {T}/TripoSG")
    sh(f"git -C {T}/TripoSG checkout -q fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c")
    nvcc = sh("ls -d /usr/local/cuda-12* /usr/local/cuda 2>/dev/null | head -1").strip().splitlines()[-1]
    log(sh(f"{nvcc}/bin/nvcc --version | tail -1").strip())
    sh(f"CUDA_HOME={nvcc} PATH={nvcc}/bin:$PATH TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4 "
       f"uv pip install --python {PY} --no-build-isolation diso", tail=600)
    log(sh(f"cd /tmp && {PY} -c \"import torch, diso; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs, diso ok')\"").strip())
    # download the pinned weights once, before two workers race for them
    sh(f"cd {FT} && {PY} -c \"import torch; from huggingface_hub import snapshot_download as s; "
       f"m = torch.load('{SIZE}', map_location='cpu', weights_only=False)['metadata']; print(s(m['model'], revision=m['revision']))\"")
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

    n_train = int(sh(f"cd {FT} && {PY} -c \"import torch; m = torch.load('{SIZE}', map_location='cpu', weights_only=False)['metadata']; "
                     f"print(sum(r['split'] == 'train' for r in m['rows']))\"").strip().splitlines()[-1])
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
    FT, DATA, SIZE, SHAPE = stage()
    setup()
    log(f"setup done; part {PART}")
    run()
    validate()
    log("RUN DONE")
except Exception as e:
    log(f"RUN FAILED: {e!r}")
    raise
'''


def cells_for(part):
    config = CONFIG.replace("__SHARDS_SRC__", SHARDS_SRC)
    cells = [("markdown", MD_INTRO), ("code", config), ("code", STAGE), ("code", SETUP), ("code", RUN), ("code", VALIDATE),
             ("code", MAIN)]
    return [(kind, src.replace("__PART__", str(part))) for kind, src in cells]


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


def metadata(part):
    return {"id": f"{OWNER}/f3dr-tg-train-p{part}", "title": f"f3dr-tg-train-p{part}", "code_file": f"f3dr-tg-train-p{part}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{OWNER}/f3dr-tg-infer", f"{OWNER}/f3dr-tg-raw"], "competition_sources": [], "kernel_sources": []}


def generate():
    for part in (1, 2, 3):
        out = HERE / "train_infer" / f"part{part}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"f3dr-tg-train-p{part}.ipynb").write_text(json.dumps(notebook(cells_for(part)), indent=1) + "\n", encoding="utf-8")
        (out / "kernel-metadata.json").write_text(json.dumps(metadata(part), indent=1) + "\n", encoding="utf-8")
        print("wrote", out)


def selftest(n_train=1838):
    ns = {}
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
    print("photo filter and data root OK")
    for part in (1, 2, 3):
        folder = HERE / "train_infer" / f"part{part}"
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        nb = json.loads((folder / meta["code_file"]).read_text())
        assert meta["id"].endswith(f"p{part}") and meta["kernel_type"] == "notebook" and meta["enable_gpu"] and meta["enable_internet"]
        for k, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]), filename=f"part{part} cell {k}")
        assert f"PART = {part}" in "".join(nb["cells"][1]["source"]) and "__PART__" not in json.dumps(nb)
        print(f"part{part}: {len(nb['cells'])} cells parse, metadata OK ({meta['id']})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        generate()
