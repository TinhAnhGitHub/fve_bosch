# Colab VM: photos from Kaggle account 3's datasets (KAGGLE_API_TOKEN from --env), then start extract_feats.py in the
# background (HF_TOKEN from --env). Log: /content/feat/run.log; features: /content/feat/out/*.npz.
import glob
import os
import subprocess
import tarfile

os.makedirs("/content/feat/out", exist_ok=True)
os.makedirs("/content/photos", exist_ok=True)
subprocess.run("pip install -q -U kaggle", shell=True, check=True)
if not os.path.exists("/content/photos/train2042.bin"):
    for ds in ("eiyuumiru/f3dr-trainphotos", "eiyuumiru/f3dr-photos"):
        r = subprocess.run(f"kaggle datasets download {ds} -p /content/photos --unzip", shell=True, capture_output=True, text=True)
        print(ds, r.returncode, r.stdout[-200:], r.stderr[-300:])
    for b in glob.glob("/content/photos/**/*.bin", recursive=True):
        with tarfile.open(b) as t:
            t.extractall("/content/photos")


def photo_dir(name: str) -> str:
    """The folder named NAME that directly holds the most .png photos (datasets unpack nested)."""
    return max(glob.glob(f"/content/photos/**/{name}", recursive=True), key=lambda d: len(glob.glob(f"{d}/*.png")))


train, ev = photo_dir("train2042"), photo_dir("eval511")
n_tr, n_ev = len(glob.glob(f"{train}/*.png")), len(glob.glob(f"{ev}/*.png"))
print("photos", train, n_tr, ev, n_ev)
assert (n_tr, n_ev) == (2042, 511)
specs = os.environ["SPECS"].split()
cmd = ["bash", "-c", 'python /content/feat/extract_feats.py "$@" > /content/feat/run.log 2>&1', "_", train, ev,
       "/content/feat/out", *specs]
# Blocking on purpose: Colab reclaims a VM whose kernel sits idle (~20 min), so the kernel waits for the extractor.
print("RUNNING", specs, flush=True)
r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("EXIT", r.returncode)
