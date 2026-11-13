# /// script
# dependencies = ["numpy", "torch", "safetensors"]
# ///
import glob, numpy as np, torch
from safetensors.torch import load_file, save_file
w = load_file(glob.glob("/home/una8hc/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/*/model.safetensors")[0])
save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in w.items()}, "/tmp/kf/dinov3_bf16.safetensors", metadata={"format": "pt"})
z = np.load("/mnt/c/tinhanh/project/ai_challenge/fve_f3dr/baselines/out/atlas/train_photos256.npz")
np.savez_compressed("/tmp/kf/photos_c.npz", ids=z["ids"], imgs=z["imgs"])
print("ok")
