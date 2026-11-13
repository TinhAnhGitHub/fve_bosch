import glob, hashlib, os, shutil, sys
def join(prefix, dst):
    with open(dst, "wb") as f:
        for p in sorted(glob.glob(prefix + "*")):
            f.write(open(p, "rb").read())
    return hashlib.md5(open(dst, "rb").read()).hexdigest()
src = sys.argv[1] if len(sys.argv) > 1 else ""
hub = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m")
snap = f"{hub}/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
os.makedirs(snap, exist_ok=True); os.makedirs(f"{hub}/refs", exist_ok=True)
open(f"{hub}/refs/main", "w").write("ea8dc2863c51be0a264bab82070e3e8836b02d51")
if os.path.isdir("/content/drive/MyDrive/f3dr_kf"):
    d = "/content/drive/MyDrive/f3dr_kf"
    shutil.copy(f"{d}/photos_c.npz", "/content/data/train_photos256.npz")
    shutil.copy(f"{d}/dinov3_bf16.safetensors", snap + "/model.safetensors")
    shutil.copy(f"{d}/dinov3_config.json", snap + "/config.json")
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    print("from drive: photos", md5("/content/data/train_photos256.npz"), "dino", md5(snap + "/model.safetensors"))
else:
    print("photos", join("/content/p2/photos.part", "/content/data/train_photos256.npz"))
    print("dino", join("/content/p2/dino.part", snap + "/model.safetensors"))
    shutil.copy("/content/p2/dinov3_config.json", snap + "/config.json")
print("want photos 8c1ca6f7fd33a708a265b2ec144308a4 dino 3418e417dc409d247c1fcdd3005b5c08")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/content/code")
import numpy as np, torch
from atlas import Tuner, plain_pixels
t = Tuner(4).cuda()
with np.load("/content/data/train_photos256.npz") as pz:
    x = torch.stack([plain_pixels(pz["imgs"][i]) for i in range(8)]).cuda()
with torch.inference_mode():
    tok = t(x).cpu()
with np.load("/content/data/train_dinov3l.npz") as z:
    ref = torch.tensor(z["feats"][:8], dtype=torch.float32)
cos = torch.nn.functional.cosine_similarity(tok, ref, dim=-1)
print(f"tokens vs cache: cosine mean {cos.mean():.4f} min {cos.min():.4f}; trainable {sum(p.numel() for p in t.tuned().values())/1e6:.1f}M")
import psutil; print("RAM GB", round(psutil.virtual_memory().total / 1e9), "GPU", torch.cuda.get_device_name(0))
print("ASSEMBLE DONE")
