import os, subprocess, glob, zipfile
os.makedirs("/content/data", exist_ok=True)
subprocess.run("pip install -q kaggle trimesh", shell=True, check=True)
for f in ["train_mirror.npz", "train.npz", "train_dinov3l.npz"]:
    if os.path.exists(f"/content/data/{f}"):
        continue
    r = subprocess.run(f"kaggle datasets download anhanhanhanh/f3dr-cache -f {f} -p /content/data", shell=True, capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-300:])
    z = f"/content/data/{f}.zip"
    if os.path.exists(z):
        zipfile.ZipFile(z).extractall("/content/data"); os.remove(z)
for f in sorted(glob.glob("/content/data/*")):
    print(f, os.path.getsize(f))
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name(0))
print("SETUP DONE")
