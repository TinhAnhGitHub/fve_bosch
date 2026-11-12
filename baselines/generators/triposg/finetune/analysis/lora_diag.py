"""Is LoRA moving at all? Size of the learned update B@A against the frozen weight, per layer, from a Lightning
checkpoint (`best.ckpt`). Also prints the validation losses of the full and LoRA runs. Paths are the Colab VM's."""
import csv
import glob

import torch

ck = torch.load("/content/runs/lora/best.ckpt", map_location="cpu", mmap=True, weights_only=False)
sd = ck["state_dict"]
print("ckpt epoch", ck.get("epoch"), "keys", len(sd))
ratios, bnorms = [], []
for k, b in sd.items():
    if "lora_B" not in k:
        continue
    a = sd[k.replace("lora_B", "lora_A")]
    w = sd.get(k.split(".lora_B")[0] + ".base_layer.weight")
    bnorms.append(b.float().norm().item())
    if w is not None:
        ratios.append((b.float() @ a.float()).norm().item() / w.float().norm().item())
print("lora_B tensors", len(bnorms), "zero lora_B", sum(n == 0 for n in bnorms))
if ratios:
    ratios.sort()
    print("|BA|/|W| median %.3g max %.3g" % (ratios[len(ratios) // 2], ratios[-1]))
for run in ("full", "lora"):
    for f in sorted(glob.glob(f"/content/runs/{run}/logs/version_*/metrics.csv"))[-1:]:
        v = [(r["epoch"], round(float(r["val_loss"]), 4)) for r in csv.DictReader(open(f)) if r.get("val_loss")]
        print(run, "val:", v)
