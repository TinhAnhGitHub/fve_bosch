# Colab L4 VM for M4: frozen-DINOv3 fold runs of the v26 and NoCat recipes. Blocking on purpose (an idle kernel gets
# reclaimed). Data from the private account-3 dataset eiyuumiru/f3dr-kf-cache (train.npz, train_dinov3l.npz,
# train_mirror.npz); code uploaded to /content/code by launch.sh. Env: KAGGLE_API_TOKEN, RUNS ("key:fold,...").
# Log: /content/run.log (run_exps.py). Snapshots: /content/run/f3dr/baselines/out/atlas/<run>_ema_e<E>.pt
import glob
import os
import subprocess

os.makedirs("/content/data", exist_ok=True)
subprocess.run("pip install -q -U kaggle", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if not all(os.path.exists(f"/content/data/{f}") for f in ("train.npz", "train_dinov3l.npz", "train_mirror.npz")):
    subprocess.run("kaggle datasets download eiyuumiru/f3dr-kf-cache -p /content/data --unzip", shell=True, check=True,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print({os.path.basename(f): os.path.getsize(f) for f in glob.glob("/content/data/*")}, flush=True)

# Same recipes as launch2.py (the Folds11 members were trained this way).
R = {"v26": "--arch talk --right-handed --seed 2 --epochs 32",
     "nocat": "--arch talk --no-cat --epochs 40"}
F = "--encoder dinov3l --folds baselines/folds/f3dr_k5_v1.csv"
runs = []
for item in os.environ["RUNS"].split(","):
    key, k = item.split(":")
    last = R[key].split("--epochs ")[1]
    runs += [f"kf_{key}_f{k}", f"{R[key]} {F} --fold {k} --save-epochs 12:{last}"]
with open("/content/run.log", "a") as log:
    r = subprocess.run(["python", "/content/code/run_exps.py", "/content/run", "/content/data", "/content/code", *runs],
                       cwd="/content", stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"M4 EXIT {r.returncode}\n")
print("M4 EXIT", r.returncode)
