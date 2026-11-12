"""Flow-matching sanity checks (run on a Colab/Kaggle GPU VM, with the cache and TripoSG checkout in place).

A. Pretrained TripoSG on the 204 val objects with the exact validation noise (seed 1729+i):
   loss with our target (clean-noise), with the flipped target, and of a zero predictor
   (= target variance), split by sigma. Gives the starting point and checks the sign.
B. Overfit one object: fixed latent + fixed noise, random sigma, LoRA lr 3e-4, 400 steps.
   The loss must fall far below A if gradients, target and time conditioning are wired correctly.
Paths below are the ones used on the Colab VM (section 18 of docs/F3DR_full_report.md).
"""
import sys
import time
from pathlib import Path

import torch

FINETUNE = "/content/f3dr/baselines/generators/triposg/finetune"
sys.path.insert(0, FINETUNE)
from common import configure_transformer, flow_batch, read_json, upstream  # noqa: E402
from train import CachedObjects, read_config  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

upstream(Path("/content/TripoSG"))
from triposg.models.transformers import TripoSGDiTModel  # noqa: E402

dev = "cuda"
meta = read_json(Path("/content/cache/metadata.json"))
va = CachedObjects(Path("/content/cache"), [r for r in meta["rows"] if r["split"] == "val"])
tr = CachedObjects(Path("/content/cache"), [r for r in meta["rows"] if r["split"] == "train"])
weights = snapshot_download(meta["model"], revision=meta["revision"])
net = TripoSGDiTModel.from_pretrained(weights, subfolder="transformer", torch_dtype=torch.bfloat16).to(dev).eval()


def forward(model, noisy, timestep, features):
    return model(noisy.to(torch.bfloat16), timestep, encoder_hidden_states=features.to(torch.bfloat16),
                 return_dict=False)[0].float()


print("A. pretrained, val objects:", len(va), flush=True)
rows = []
with torch.no_grad():
    for i in range(len(va)):
        b = {k: v[None].to(dev) for k, v in va[i].items()}
        g = torch.Generator(device=dev).manual_seed(1729 + i)
        clean = b["mean"]
        noise = torch.randn(clean.shape, device=dev, generator=g)
        sigma = torch.rand(1, device=dev, generator=g)
        noisy, target, timestep = flow_batch(clean, noise, sigma)
        pred = forward(net, noisy, timestep, b["features"])
        rows.append((sigma.item(), (pred - target).square().mean().item(),
                     (pred + target).square().mean().item(), target.square().mean().item(),
                     clean.square().mean().item()))
mean = lambda j, rs=rows: sum(r[j] for r in rs) / max(1, len(rs))  # noqa: E731
print(f"val loss, our target (clean-noise): {mean(1):.4f}")
print(f"val loss, flipped target:          {mean(2):.4f}")
print(f"zero predictor (target variance):  {mean(3):.4f}   latent mean-square {mean(4):.4f}")
for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
    rs = [r for r in rows if lo <= r[0] < lo + 0.2]
    print(f"  sigma {lo:.1f}-{lo + 0.2:.1f}: n={len(rs):3d} loss {mean(1, rs):.4f} zero-pred {mean(3, rs):.4f}")

print("B. overfit one train object, LoRA lr 3e-4", flush=True)
config = read_config(Path(FINETUNE) / "configs/lora_24gb.json")
config["lora_dropout"] = 0.0
configure_transformer(net, config)
net.train()
params = [p for p in net.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
b = {k: v[None].to(dev) for k, v in tr[0].items()}
g = torch.Generator(device=dev).manual_seed(7)
noise = torch.randn(b["mean"].shape, device=dev, generator=g)
window, t0 = [], time.time()
for step in range(1, 401):
    sigma = torch.rand(1, device=dev, generator=g)
    noisy, target, timestep = flow_batch(b["mean"], noise, sigma)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = forward(net, noisy, timestep, b["features"])
    loss = (pred - target).square().mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    window.append(loss.item())
    if step % 25 == 0:
        print(f"  step {step}: mean loss last 25 = {sum(window) / len(window):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        window = []
print("FM DIAG DONE")
