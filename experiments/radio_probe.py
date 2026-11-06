# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "transformers", "peft", "timm", "einops", "pillow", "numpy", "safetensors"]
# ///
"""Probe C-RADIOv4-H: load, list attention modules, check output shapes / VRAM / fp16 finiteness."""
import sys, os, time
from pathlib import Path
import torch
from transformers import AutoModel
P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/kaggle_stage/cradio")
m = AutoModel.from_pretrained(P, trust_remote_code=True, dtype=torch.float16).cuda().eval()
print("params M", sum(p.numel() for p in m.parameters()) / 1e6)
names = [n for n, mod in m.named_modules() if n.endswith("attn.qkv") or n.endswith("attn.proj")]
print(len(names), "attn linear modules, e.g.", names[:3])
x = torch.rand(2, 3, 512, 512, device="cuda", dtype=torch.float16)
with torch.inference_mode():
    t0 = time.time()
    summary, feats = m(x)
    print("summary", tuple(summary.shape), "feats", tuple(feats.shape), "finite", bool(torch.isfinite(summary).all() and torch.isfinite(feats).all()), f"{time.time()-t0:.2f}s")
    out = m.radio_model.forward_intermediates(x, indices=[7, 15, 23, 31], return_prefix_tokens=False, norm=False, intermediates_only=True)
    print("intermediates", [tuple(o.shape) if hasattr(o, "shape") else type(o) for o in out])
print("peak VRAM GB", torch.cuda.max_memory_allocated() / 2**30)
