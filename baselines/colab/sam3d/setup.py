# Colab L4 VM: photos (Kaggle account 3 datasets), SAM 3D install + weights, then SAM 3D on shard K/N of the training
# and eval photos. Blocking on purpose (an idle kernel gets reclaimed). Env: KAGGLE_API_TOKEN, HF_TOKEN, SHARD (k/n).
# Log: /content/sam3d/run.log. Output: /content/sam3d/out/sam3d_{train,eval}_s<k>.npz (gen_sam3d.py format, saved every 10).
import glob
import os
import subprocess
import tarfile

H = "/content/sam3d"
os.makedirs(f"{H}/out", exist_ok=True)
os.makedirs("/content/photos", exist_ok=True)
LOG = open(f"{H}/run.log", "a")


def sh(cmd: str, env: dict | None = None) -> None:
    LOG.write(f"\n### {cmd[:120]}\n")
    LOG.flush()
    r = subprocess.run(cmd, shell=True, cwd=H, stdin=subprocess.DEVNULL, stdout=LOG, stderr=subprocess.STDOUT,
                       executable="/bin/bash", env={**os.environ, **(env or {})})
    if r.returncode:
        LOG.write(f"FAILED {r.returncode}: {cmd[:120]}\n")
        LOG.flush()
        raise SystemExit(f"failed: {cmd[:120]}")


sh("pip install -q -U kaggle")
if not os.path.exists("/content/photos/train2042.bin"):
    for ds in ("eiyuumiru/f3dr-trainphotos", "eiyuumiru/f3dr-photos"):
        sh(f"kaggle datasets download {ds} -p /content/photos --unzip")
    for b in glob.glob("/content/photos/**/*.bin", recursive=True):
        with tarfile.open(b) as t:
            t.extractall("/content/photos")


def photo_dir(name: str) -> str:
    """The folder named NAME that directly holds the most .png photos (datasets unpack nested)."""
    return max(glob.glob(f"/content/photos/**/{name}", recursive=True), key=lambda d: len(glob.glob(f"{d}/*.png")))


train, ev = photo_dir("train2042"), photo_dir("eval511")
LOG.write(f"photos {train} {len(glob.glob(train + '/*.png'))} {ev} {len(glob.glob(ev + '/*.png'))}\n")
if not os.path.exists(f"{H}/.venv-sam3d/bin/python"):
    sh("bash install.sh")
if not os.path.exists(f"{H}/Fast-SAM3D/checkpoints/moge-vitl"):
    sh("bash download_weights.sh")
k = os.environ["SHARD"].split("/")[0]
py = f"{H}/.venv-sam3d/bin/python"
sh(f"{py} gen_sam3d.py out/sam3d_train_s{k}.npz {train} --shard {os.environ['SHARD']}")
sh(f"{py} gen_sam3d.py out/sam3d_eval_s{k}.npz {ev} --shard {os.environ['SHARD']}")
LOG.write("SHARD DONE\n")
LOG.flush()
print("SHARD DONE")
