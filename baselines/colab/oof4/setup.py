# Colab L4 VM: out-of-fold clouds + strict scores for the M4 recipes (tools/oof.py predict). Blocking on purpose.
# Data from the private account-3 dataset eiyuumiru/f3dr-kf-cache into the repo layout oof.py expects; fold snapshots
# uploaded by launch.sh to /content/ck. Env: KAGGLE_API_TOKEN, JOBS ("name:epoch,..."). Output /content/oof/<name>_e<E>.npz
import os
import subprocess

A = "/content/f3dr/baselines/out/atlas"
os.makedirs(A, exist_ok=True)
os.makedirs("/content/oof", exist_ok=True)
subprocess.run("pip install -q -U kaggle trimesh", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if not all(os.path.exists(f"{A}/{f}") for f in ("train.npz", "train_dinov3l.npz", "train_mirror.npz")):
    subprocess.run(f"kaggle datasets download eiyuumiru/f3dr-kf-cache -p {A} --unzip", shell=True, check=True,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open("/content/run.log", "a") as log:
    for job in os.environ["JOBS"].split(","):
        name, epoch = job.split(":")
        out = f"/content/oof/{name}_e{epoch}.npz"
        if os.path.exists(out):
            continue
        r = subprocess.run(["python", "/content/f3dr/baselines/tools/oof.py", "predict", "/content/ck", name, epoch, out],
                           cwd="/content/f3dr", env=dict(os.environ, RH="1", HF_HUB_OFFLINE="1"),
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"JOB {job} EXIT {r.returncode}\n")
        log.flush()
    log.write("OOF4 DONE\n")
print("OOF4 DONE")
