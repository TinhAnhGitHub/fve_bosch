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
def split_parts() -> None:
    """Every 10 min, copy each changed out/*.npz into 20 MB parts under parts/<name>/ (colab download fails >100 MB)."""
    import shutil
    import time
    seen = {}
    while True:
        for f in glob.glob(f"{H}/Fast-SAM3D/out/*.npz"):  # gen_sam3d.py chdirs into Fast-SAM3D
            m = os.path.getmtime(f)
            if seen.get(f) == m or time.time() - m < 20:  # unchanged, or possibly mid-write
                continue
            name = os.path.basename(f)
            tmp, dst = f"{H}/parts/.{name}", f"{H}/parts/{name}"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            with open(f, "rb") as src:
                n = 0
                while chunk := src.read(20 << 20):
                    with open(f"{tmp}/part{n:02d}", "wb") as out:
                        out.write(chunk)
                    n += 1
            shutil.rmtree(dst, ignore_errors=True)
            os.rename(tmp, dst)
            seen[f] = m
        time.sleep(600)


import threading  # noqa: E402

os.makedirs(f"{H}/parts", exist_ok=True)
threading.Thread(target=split_parts, daemon=True).start()
k = os.environ["SHARD"].split("/")[0]
py = f"{H}/.venv-sam3d/bin/python"
os.makedirs(f"{H}/Fast-SAM3D/out", exist_ok=True)
for name in {os.path.basename(p).split(".part")[0] for p in glob.glob(f"{H}/resume/*.part*")}:
    dst = f"{H}/Fast-SAM3D/out/{name}"  # partial results uploaded in parts by launch.sh: gen_sam3d resumes from them
    if not os.path.exists(dst):
        with open(dst, "wb") as out:
            for p in sorted(glob.glob(f"{H}/resume/{name}.part*")):
                with open(p, "rb") as src:
                    out.write(src.read())
sh(f"uv pip install --python {py} -q 'setuptools<70' && {py} -c 'import pkg_resources'")  # lightning needs it
sh(f"{py} gen_sam3d.py out/sam3d_train_s{k}.npz {train} --shard {os.environ['SHARD']}")
sh(f"{py} gen_sam3d.py out/sam3d_eval_s{k}.npz {ev} --shard {os.environ['SHARD']}")
LOG.write("SHARD DONE\n")
LOG.flush()
print("SHARD DONE")
