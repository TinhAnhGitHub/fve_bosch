# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "numpy"]
# ///
"""B4.0 audit: does LoRA fine-tuning of DINOv3-L/16 fit in 6 GB? Measures peak VRAM + step time on random input."""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

env = Path(__file__).resolve().parent.parent / ".env"
for line in env.read_text().splitlines():
    if line.startswith("HUGGINGFACE_API_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")

free, total = torch.cuda.mem_get_info()
print(f"GPU free {free / 2**30:.2f} GB of {total / 2**30:.2f} GB (rest used by Windows/other)")


def trial(size: int, batch: int, ckpt: bool, targets: list[str]) -> None:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).cuda()
        if ckpt:
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=targets))
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        x = torch.randn(batch, 3, size, size, device="cuda", dtype=torch.bfloat16)
        times = []
        for i in range(4):
            torch.cuda.synchronize(); t = time.time()
            loss = m(pixel_values=x).last_hidden_state[:, 0].float().pow(2).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.time() - t)
        print(f"OK   size={size} batch={batch} ckpt={ckpt} lora={len(targets)} mods | peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB | {sum(times[1:]) / 3:.2f}s/step "
              f"-> {2042 * 0.8 / batch * sum(times[1:]) / 3 / 60:.1f} min/epoch (1633 imgs)")
    except torch.OutOfMemoryError:
        print(f"OOM  size={size} batch={batch} ckpt={ckpt} lora={len(targets)} mods")
    finally:
        for n in ("m", "opt", "x", "loss"):
            globals().pop(n, None)
        gc.collect(); torch.cuda.empty_cache()


QKV = ["q_proj", "k_proj", "v_proj"]
for size, batch, ckpt in [(512, 8, True), (512, 4, True), (512, 2, True), (512, 1, True), (512, 4, False),
                          (384, 8, True), (384, 4, True), (224, 16, True), (224, 8, False)]:
    trial(size, batch, ckpt, QKV)
