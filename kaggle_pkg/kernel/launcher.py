"""Kaggle launcher: one fold per T4 (independent processes, no inter-GPU traffic). Retries a fold in fp32 on exit code 3."""
import os, subprocess, sys, time
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.56", "peft", "timm", "einops"])
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])  # peft rejects Kaggle's old torchao
import torch, transformers, peft  # noqa: E402

print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__,
      "GPUs", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
def find(slug: str) -> Path:
    """Locate a mounted dataset: /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug> etc."""
    hits = [p for p in Path("/kaggle/input").rglob(slug) if p.is_dir()]
    hits = [p for p in hits if len(p.relative_to("/kaggle/input").parts) <= 3]
    print(slug, "->", hits, flush=True)
    if not hits:
        print("input tree:", sorted(str(x) for x in Path("/kaggle/input").glob("*"))[:20], sorted(str(x) for x in Path("/kaggle/input").glob("*/*"))[:20], flush=True)
        raise SystemExit(f"dataset {slug} not mounted")
    return hits[0]


DATA, PREP, MODEL = find("fve-bosch-data"), find("fve-b42-prep"), find("cradio-v4-h")
print({d.name: sorted(x.name for x in d.iterdir())[:6] for d in (DATA, PREP, MODEL)}, flush=True)
TRAIN = str(PREP / "b4_2_train.py")
FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1").split(",")]
EPOCHS = os.environ.get("EPOCHS", "14")
EXTRA = os.environ.get("EXTRA", "--label-smooth 0.1 --scale-aug 0.15 --lora-r 8 --head-warm 2").split()  # B4.3: regularised C-RADIO
n_gpu = max(1, torch.cuda.device_count())
Path("/kaggle/working/logs").mkdir(parents=True, exist_ok=True)


def start(fold: int, gpu: int, fp32: bool) -> subprocess.Popen:
    log = open(f"/kaggle/working/logs/f{fold}{'_fp32' if fp32 else ''}.log", "w")
    cmd = [sys.executable, "-u", TRAIN, "--fold", str(fold), "--epochs", EPOCHS, "--batch", "4", "--data", str(DATA), "--prep", str(PREP), "--model", str(MODEL), *EXTRA] + (["--fp32"] if fp32 else [])
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


jobs = {f: (start(f, i % n_gpu, False), False) for i, f in enumerate(FOLDS)}
seen: dict[int, int] = {f: 0 for f in FOLDS}
while jobs:
    time.sleep(60)
    for f, (p, fp32) in list(jobs.items()):
        log = Path(f"/kaggle/working/logs/f{f}{'_fp32' if fp32 else ''}.log").read_text().splitlines()
        for line in log[seen[f]:]:
            if "Warning" not in line and "Loading weights" not in line:
                print(f"[f{f}] {line}", flush=True)
        seen[f] = len(log)
        rc = p.poll()
        if rc is None:
            continue
        if rc == 3 and not fp32:
            print(f"[f{f}] fp16 gave non-finite loss; retrying in fp32", flush=True)
            jobs[f] = (start(f, FOLDS.index(f) % n_gpu, True), True); seen[f] = 0
        else:
            print(f"[f{f}] finished rc={rc}", flush=True)
            del jobs[f]
print("done; outputs:", sorted(str(p) for p in Path("/kaggle/working/out").rglob("*") if p.is_file()), flush=True)
