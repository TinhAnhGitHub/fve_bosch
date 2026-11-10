"""Write the Colab notebooks for F3DR training runs. Code only: the token caches are uploaded separately.

usage: python3 baselines/colab/make_notebooks.py
Run them with the Colab CLI (session `gpu`):
    colab exec -s gpu -f baselines/colab/f3dr_setup.ipynb --timeout 900
    colab exec -s gpu -f baselines/colab/f3dr_queue.ipynb
    colab exec -s gpu -f baselines/colab/f3dr_status.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/content/f3dr"
SIZES = {"train.npz": (1007472814, "c5b97585935e9a2a8320820f6b7bfaca"),
         "train_dinov3l.npz": (1075129838, "6f2ec68acca504073fd4b49035518b98")}
COMMON = "--target mm --no-flip-loss --ema 0.999"
BASE = "pl_t_s0_ema"
RUNS = [  # name, flags: one change per run against the baseline
    ("n_pos", "--encoder dinov3l --arch talk --pos-emb"),
    ("n_pos_s1", "--encoder dinov3l --arch talk --pos-emb --seed 1"),
    ("n_cat", "--encoder dinov3l --arch talk --cat-drop 0.15"),
    ("n_rep", "--encoder dinov3l --arch talk --repulsion 1.0"),
]


def cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip() + "\n"}


def notebook(cells: list[str]) -> dict:
    return {"cells": [cell(c) for c in cells], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}


setup = [
    f"""
# @title Join the uploaded cache parts and check them
import glob, hashlib, os
out = "{ROOT}/baselines/out/atlas"
want = {SIZES!r}
for name, (size, md5) in want.items():
    dst = f"{{out}}/{{name}}"
    parts = sorted(glob.glob(f"/content/parts/{{name}}.part*"))
    if parts and not (os.path.exists(dst) and os.path.getsize(dst) == size):
        with open(dst, "wb") as f:
            for p in parts:
                f.write(open(p, "rb").read())
    h = hashlib.md5(open(dst, "rb").read()).hexdigest()
    print(name, os.path.getsize(dst), "OK" if (os.path.getsize(dst), h) == (size, md5) else "MISMATCH")
""",
    """
# @title Packages and GPU
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trimesh"], check=True)
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
""",
    f"""
# @title Smoke test: 1 epoch with the new position embedding
import subprocess
r = subprocess.run("cd {ROOT} && python baselines/atlas.py train {COMMON} --encoder dinov3l --arch talk --pos-emb --epochs 1 --ckpt /tmp/smoke.pt",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "rc", r.returncode)
""",
]

lines = "\n".join(f'run {n} {f}' for n, f in RUNS)
queue_sh = f"""#!/bin/bash
# One change per run against {BASE}; finished runs are skipped on relaunch.
R={ROOT}; A=$R/baselines/out/atlas; Q=$A/queue; L=$Q/queue.log
cd $R
say() {{ echo "$(date +%H:%M) $*" >> $L; }}
score() {{ [ -f $A/$1.pt ] || return 0; python $Q/score2.py $R {BASE} $1 > $Q/$1.score.log 2>&1 || say "FAILED score $1"; grep -E "^$1 |^RESULT" $Q/$1.score.log >> $L; }}
run() {{
  name=$1; shift
  [ -f $Q/$name.done ] && return
  say "START $name: $*"
  python baselines/atlas.py train {COMMON} "$@" --ckpt $A/$name.pt > $Q/$name.train.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "$(grep -E '^saved' $Q/$name.train.log | head -1)"; score $name; score ${{name}}_ema
  else say "FAILED $name rc=$rc: $(tail -1 $Q/$name.train.log | cut -c1-160)"; fi
  echo $rc > $Q/$name.done
}}
{lines}
say "QUEUE DONE"
"""

queue = [
    f"""
# @title Write the run queue and start it in the background (survives this cell)
import subprocess
q = "{ROOT}/baselines/out/atlas/queue/queue.sh"
open(q, "w").write({queue_sh!r})
running = subprocess.run(["pgrep", "-fx", f"bash {{q}}"], capture_output=True, text=True).stdout.strip()
if running:
    print("already running:", running)
else:
    subprocess.Popen(["bash", q], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("started")
""",
]

status = [
    f"""
# @title Queue log, current epoch, GPU
import subprocess
Q = "{ROOT}/baselines/out/atlas/queue"
print(subprocess.run(f"tail -n 20 {{Q}}/queue.log; echo ---; for f in {{Q}}/*.train.log; do echo $(basename $f) $(grep -c '^epoch' $f) epochs: $(grep '^epoch' $f | tail -1); done; echo ---; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout)
""",
]

for name, cells in [("f3dr_setup", setup), ("f3dr_queue", queue), ("f3dr_status", status)]:
    (HERE / f"{name}.ipynb").write_text(json.dumps(notebook(cells), indent=1))
    print("wrote", HERE / f"{name}.ipynb")
