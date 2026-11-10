"""B4.2: LoRA fine-tune NVIDIA C-RADIOv4-H on ONE CV fold with a MoGe-2-style conv head and mesh-derived aux targets.

Standalone (no repo imports) so it runs on Kaggle. Inputs (paths overridable):
  --data  dir with train/ and val/ images           (private dataset fve-bosch-data)
  --prep  dir with rows.csv and fold{f}.npz          (b4_2_prep.py output)
  --model dir with the C-RADIOv4-H HF files          (weights + custom code)
Network: C-RADIOv4-H (ViT-H/16, fp16 frozen base, fp32 LoRA r16 on the fused qkv, per-block gradient checkpointing).
Head (MoGe-2 style, see plan/experiment_plan.md): patch tokens from blocks 8/16/24/32 -> per-layer 1x1 conv -> SUM -> +2 UV
channels -> 3x3 conv, residual conv block, stride-2 conv, residual conv block -> attention pooling (4 learned queries);
concatenated with the RADIO summary vector -> MLP -> [r, oversized logit, 8 aux]. --last-only keeps only the last block.
Outputs in --out: preds.npz (out_heldout, out_val flip-averaged, history), adapter/head state, log.
Exit code 3 if the loss is non-finite early (caller retries with --fp32).
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

SIZE, GRID = 512, 32
BLOCKS = (7, 15, 23, 31)  # 0-based block indices -> layers 8/16/24/32
N_AUX = 8


# ------------------------------------------------------------------ data
def load_images(paths: list[Path]) -> torch.Tensor:
    def one(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(one, paths)))
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def augment(x: torch.Tensor, rng: np.random.Generator, scale_aug: float = 0.0) -> torch.Tensor:
    """x float [B,3,H,W] in [0,1]: flip, +-10 deg rotation, 5% shift (backdrop fill), brightness/contrast."""
    out = []
    for img in x:
        if rng.random() < 0.5:
            img = img.flip(-1)
        border = torch.cat([img[:, :8].flatten(1), img[:, -8:].flatten(1)], 1).median(1).values.tolist()
        img = TF.affine(img, angle=float(rng.uniform(-10, 10)),
                        translate=[int(rng.uniform(-0.05, 0.05) * SIZE), int(rng.uniform(-0.05, 0.05) * SIZE)],
                        scale=float(rng.uniform(1 - scale_aug, 1 + scale_aug)), shear=[0.0], fill=border)
        b, c = rng.uniform(-0.15, 0.15), rng.uniform(0.85, 1.15)
        out.append(((img - img.mean()) * c + img.mean() + b).clamp(0, 1))
    return torch.stack(out)


# ------------------------------------------------------------------ model
class CkptBlock(nn.Module):
    """Wrap a transformer block with activation checkpointing; hooks on this module see tensors that carry gradient."""

    def __init__(self, blk: nn.Module) -> None:
        super().__init__()
        self.blk = blk

    def forward(self, *a, **k):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.blk, *a, use_reentrant=False, **k)
        return self.blk(*a, **k)


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                               nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.f(x))


class Head(nn.Module):
    def __init__(self, d_in: int, d_sum: int, n_levels: int, n_out: int, c: int = 128, p_drop: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(d_in, c, 1) for _ in range(n_levels)])
        self.stem = nn.Sequential(nn.Conv2d(c + 2, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU())
        self.res1, self.res2 = ResBlock(c), ResBlock(c)
        self.down = nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU())
        self.queries = nn.Parameter(torch.randn(4, c) * 0.02)
        self.pool = nn.MultiheadAttention(c, 4, batch_first=True)
        self.drop2d = nn.Dropout2d(p_drop)
        d = d_sum + 4 * c
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(p_drop), nn.Linear(512, n_out))
        v = torch.linspace(-1, 1, GRID)
        self.register_buffer("uv", torch.stack(torch.meshgrid(v, v, indexing="ij"))[None], persistent=False)

    def forward(self, feats: list[torch.Tensor], summary: torch.Tensor) -> torch.Tensor:
        g = sum(p(f) for p, f in zip(self.proj, feats))
        g = self.stem(torch.cat([g, self.uv.expand(g.shape[0], -1, -1, -1).to(g.dtype)], 1))
        g = self.res2(self.down(self.res1(g)))
        g = self.drop2d(g)
        tok = g.flatten(2).transpose(1, 2)
        pooled, _ = self.pool(self.queries.expand(tok.shape[0], -1, -1), tok, tok)
        return self.mlp(torch.cat([summary, pooled.flatten(1)], 1))


class XBlock(nn.Module):
    """f3dr atlas.Block (pre-norm cross-attention to the image tokens + MLP, residuals) plus self-attention among the
    queries, so the measurements can read each other (few queries, so it is cheap)."""

    def __init__(self, w: int, heads: int, p_drop: float) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(w), nn.LayerNorm(w), nn.LayerNorm(w)
        self.sa = nn.MultiheadAttention(w, heads, dropout=p_drop, batch_first=True)
        self.ca = nn.MultiheadAttention(w, heads, dropout=p_drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Dropout(p_drop), nn.Linear(2 * w, w))

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        x = x + self.ca(self.n2(x), mem, mem, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


def sincos_2d(g: int, w: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position code for a g x g token grid, [g*g, w] (w divisible by 4)."""
    v = torch.arange(g, dtype=torch.float32)
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    f = 1.0 / 10000 ** (torch.arange(w // 4, dtype=torch.float32) / (w // 4))
    a, b = yy.flatten()[:, None] * f, xx.flatten()[:, None] * f
    return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], 1)


class QueryDecoder(nn.Module):
    """Encoder-decoder head (f3dr DeepAtlas retargeted): one learned query per output value [r, oversized logit, aux...]
    attends to the fused multi-depth patch tokens (MoGe-2 style per-layer projection + sum, as Head) plus the CLS/mean
    summary token. f3dr queries are 8192 surface points; FVE queries are measurements. Zero-initialised readout: the
    first prediction is the cluster constant (r = 0, logit 0)."""

    def __init__(self, d_in: int, d_sum: int, n_levels: int, n_out: int, w: int = 256, depth: int = 4, heads: int = 8,
                 p_drop: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d_in, w) for _ in range(n_levels)])
        self.sum_in = nn.Linear(d_sum, w)
        self.norm_mem = nn.LayerNorm(w)
        self.queries = nn.Parameter(torch.randn(n_out, w) * 0.02)
        self.blocks = nn.ModuleList([XBlock(w, heads, p_drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(w)
        self.out_w = nn.Parameter(torch.zeros(n_out, w))
        self.out_b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("pos", sincos_2d(GRID, w), persistent=False)

    def forward(self, feats: list[torch.Tensor], summary: torch.Tensor) -> torch.Tensor:
        mem = sum(p(f.flatten(2).transpose(1, 2)) for p, f in zip(self.proj, feats)) + self.pos
        mem = self.norm_mem(torch.cat([self.sum_in(summary)[:, None], mem], 1))
        x = self.queries.expand(mem.shape[0], -1, -1)
        for blk in self.blocks:
            x = blk(x, mem)
        return (self.norm(x) * self.out_w).sum(-1) + self.out_b


IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def amp_dtype(dtype: torch.dtype):
    return dtype if dtype in (torch.float16, torch.bfloat16) else None


class Net(nn.Module):
    """backbone 'radio' (C-RADIOv4-H, local code) or 'dinov3' (HF DINOv3 ViT-L/16): LoRA + multi-layer conv head."""

    def __init__(self, model_dir: str, backbone: str, dtype: torch.dtype, last_only: bool, lora_r: int, drop_path: float,
                 head: str = "conv", n_out: int = 2 + N_AUX) -> None:
        super().__init__()
        self.kind = backbone
        if backbone == "radio":
            base = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype).cuda()
            for p in base.parameters():
                p.requires_grad = False
            vit = base.radio_model.model
            n_blocks, d = len(vit.blocks), getattr(vit, "embed_dim", 1280)
            for i in range(n_blocks):
                vit.blocks[i] = CkptBlock(vit.blocks[i])
            self.taps = [n_blocks - 1] if last_only else [min(b, n_blocks - 1) for b in BLOCKS]
            self._cap: list[torch.Tensor] = []
            for i in self.taps:
                vit.blocks[i].register_forward_hook(lambda m, a, o: self._cap.append(o[0] if isinstance(o, tuple) else o))
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=r".*blocks\.\d+\.blk\.attn\.qkv")
        else:
            base = AutoModel.from_pretrained(model_dir, dtype=dtype, drop_path_rate=drop_path).cuda()
            for p in base.parameters():
                p.requires_grad = False
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            n_layers, d = base.config.num_hidden_layers, base.config.hidden_size
            self.taps = [n_layers] if last_only else [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
            self.n_prefix = 1 + getattr(base.config, "num_register_tokens", 4)
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj"])
        self.backbone = get_peft_model(base, cfg)
        for n, p in self.backbone.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        with torch.no_grad():
            _, s = self.encode(torch.zeros(1, 3, SIZE, SIZE, device="cuda", dtype=dtype))
        cls = QueryDecoder if head == "decoder" else Head
        self.head = cls(d, s.shape[-1], len(self.taps), n_out).cuda()

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        n = GRID * GRID
        if self.kind == "radio":
            self._cap.clear()
            summary, _ = self.backbone(x)
            return [c[:, -n:].transpose(1, 2).reshape(x.shape[0], -1, GRID, GRID) for c in self._cap], summary
        dt = x.dtype
        x = ((x.float() - IM_MEAN.to(x.device)) / IM_STD.to(x.device)).to(dt)
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        h = out.last_hidden_state
        feats = [out.hidden_states[t][:, -n:].transpose(1, 2).reshape(x.shape[0], -1, GRID, GRID) for t in self.taps]
        return feats, torch.cat([h[:, 0], h[:, self.n_prefix:].mean(1)], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, summary = self.encode(x)
        return self.head([f.float() for f in feats], summary.float())


@torch.inference_mode()
def predict(net: Net, imgs: torch.Tensor, dtype, batch: int = 8) -> np.ndarray:
    net.eval()
    outs = []
    for i in range(0, len(imgs), batch):
        x = (imgs[i:i + batch].cuda().float() / 255).to(dtype)
        with torch.autocast("cuda", dtype=amp_dtype(dtype) or torch.float16, enabled=amp_dtype(dtype) is not None):
            o = (net(x) + net(x.flip(-1))) / 2
        outs.append(o.float().cpu().numpy())
    return np.concatenate(outs)


def decode(out: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    r = np.clip(out[:, 0], -2.5, 2.5)
    large = (1 / (1 + np.exp(-out[:, 1])) > 0.5) & np.isfinite(b1)
    return np.where(large, b1, b0) + r


def smape(y: np.ndarray, ln: np.ndarray) -> float:
    return float(np.mean(2 * np.tanh(np.abs(ln - np.log(y)) / 2)) * 100)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--head-warm", type=int, default=2, help="epochs with the LoRA lr at 0")
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--last-only", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="alias for --dtype fp32")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--backbone", default="radio", choices=["radio", "dinov3"])
    ap.add_argument("--label-smooth", type=float, default=0.0, help="smoothing on the oversized-mode BCE target")
    ap.add_argument("--scale-aug", type=float, default=0.0, help="+- scale jitter, e.g. 0.15")
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth (dinov3 backbone only)")
    ap.add_argument("--head", default="conv", choices=["conv", "decoder"], help="conv head (B4.1/B4.2) or query decoder (B6)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="/kaggle/input/fve-bosch-data")
    ap.add_argument("--prep", default="/kaggle/input/fve-b42-prep")
    ap.add_argument("--model", default="/kaggle/input/cradio-v4-h")
    ap.add_argument("--out", default="/kaggle/working/out")
    a = ap.parse_args()
    out_dir = Path(a.out) / f"f{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if a.fp32 else DTYPES[a.dtype]
    rows = pd.read_csv(Path(a.prep) / "rows.csv")
    tr_rows, va_rows = rows[rows.split == "train"].reset_index(drop=True), rows[rows.split == "val"].reset_index(drop=True)
    d = np.load(Path(a.prep) / f"fold{a.fold}.npz")
    tr_idx, va_idx = d["tr_idx"], d["va_idx"]
    if a.smoke:
        tr_idx, va_idx = tr_idx[:24], va_idx[:16]
    sl = lambda k, idx: d[k][: len(idx)] if a.smoke else d[k]
    P = lambda df, sp: [Path(a.data) / sp / r.group / r.category / r.sample_id / "rgb_industrial.png" for r in df.itertuples()]
    t0 = time.time()
    imgs_t, imgs_q = load_images(P(tr_rows.iloc[tr_idx], "train")), load_images(P(tr_rows.iloc[va_idx], "train"))
    imgs_v = load_images(P(va_rows.iloc[:16] if a.smoke else va_rows, "val"))
    print(f"fold {a.fold}: train {len(tr_idx)} held-out {len(va_idx)} val {len(imgs_v)}; images in {time.time() - t0:.0f}s", flush=True)

    torch.manual_seed(a.fold)
    rng = np.random.default_rng(a.fold)
    net = Net(a.model, a.backbone, dtype, a.last_only, a.lora_r, a.drop_path, a.head, 2 + d["aux"].shape[1])
    lora = [p for p in net.backbone.parameters() if p.requires_grad]
    head_p = list(net.head.parameters())
    print(f"taps {net.taps}; trainable LoRA {sum(p.numel() for p in lora) / 1e6:.2f}M + head "
          f"{sum(p.numel() for p in head_p) / 1e6:.2f}M; dtype {dtype}", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": 2e-4, "base": 2e-4, "is_lora": True},
                             {"params": head_p, "lr": 1e-3, "base": 1e-3, "is_lora": False}], weight_decay=0.05)
    spe = max(1, len(tr_idx) // a.batch)
    total, warm = spe * a.epochs, spe
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    amp = amp_dtype(dtype)
    ema = {n: p.detach().clone().float() for n, p in [*net.backbone.named_parameters(), *net.head.named_parameters()] if p.requires_grad}
    named = {n: p for n, p in [*net.backbone.named_parameters(), *net.head.named_parameters()] if p.requires_grad}

    r_t, ok_t = torch.tensor(sl("r", tr_idx)), torch.tensor(sl("ok", tr_idx))
    u_t, mx_t, aux_t = torch.tensor(sl("u", tr_idx)), torch.tensor(sl("mixed", tr_idx)), torch.tensor(sl("aux", tr_idx))
    q_b0, q_b1, q_y = sl("q_base0", va_idx), sl("q_base1", va_idx), sl("q_y", va_idx)

    def swap_ema() -> dict:
        backup = {n: p.detach().clone() for n, p in named.items()}
        for n, p in named.items():
            p.data.copy_(ema[n])
        return backup

    def restore(backup: dict) -> None:
        for n, p in named.items():
            p.data.copy_(backup[n])

    def evaluate() -> float:
        bk = swap_ema()
        out = predict(net, imgs_q, dtype)
        restore(bk); net.train()
        return smape(q_y, decode(out, q_b0, q_b1))

    step, t0, bad, hist = 0, time.time(), 0, []
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(len(tr_idx))
        run = []
        for b in range(spe):
            frac = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
            for g in opt.param_groups:
                g["lr"] = g["base"] * frac * (0.0 if (g["is_lora"] and ep < a.head_warm) else 1.0)
            idx = perm[b * a.batch:(b + 1) * a.batch]
            x = augment(imgs_t[idx].cuda().float() / 255, rng, a.scale_aug).to(dtype)
            with torch.autocast("cuda", dtype=amp or torch.float16, enabled=amp is not None):
                out = net(x)
            out = out.float()
            ii = torch.as_tensor(idx)
            l_r = (F.smooth_l1_loss(out[:, 0].cpu(), r_t[ii], reduction="none", beta=0.3) * ok_t[ii]).sum() / ok_t[ii].sum().clamp(min=1)
            l_u = (F.binary_cross_entropy_with_logits(out[:, 1].cpu(), u_t[ii] * (1 - a.label_smooth) + 0.5 * a.label_smooth, reduction="none") * mx_t[ii]).sum() / mx_t[ii].sum().clamp(min=1)
            l_a = F.l1_loss(out[:, 2:].cpu(), aux_t[ii])
            loss = l_r + 0.5 * l_u + 0.1 * l_a
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad >= 3 and ep == 0 and dtype == torch.float16:
                    print("NON-FINITE loss in epoch 1 -> exit 3 (retry with --fp32)", flush=True)
                    sys.exit(3)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([*lora, *head_p], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                dcy = min(a.ema, (1 + step) / (10 + step))
                for n, p in named.items():
                    ema[n].mul_(dcy).add_(p.detach().float(), alpha=1 - dcy)
            run.append((loss.item(), l_r.item(), l_u.item(), l_a.item())); step += 1
            if a.smoke and step in (2, 5):
                print(f"  step {step}: loss {np.mean(run[-2:], 0).round(3)} peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        m = np.mean(run, 0) if run else np.full(4, np.nan)
        msg = f"epoch {ep + 1}/{a.epochs} loss {m[0]:.3f} (r {m[1]:.3f} mode {m[2]:.3f} aux {m[3]:.3f}) {time.time() - t0:.0f}s " \
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
        if (ep + 1) % 2 == 0 or ep + 1 == a.epochs:
            sm = evaluate(); hist.append((ep + 1, sm)); msg += f"  | held-out sMAPE (EMA) {sm:.2f}"
        print(msg, flush=True)

    bk = swap_ema()
    out_q, out_v = predict(net, imgs_q, dtype), predict(net, imgs_v, dtype)
    sm = smape(q_y, decode(out_q, q_b0, q_b1))
    print(f"\nFINAL fold {a.fold}: held-out sMAPE (EMA weights) {sm:.2f}", flush=True)
    if "aux_mean" in d:  # aux = [3 sorted ln bbox dims (cm), ln fill, ...]: their sum is ln volume, with no category table
        geo = (out_q[:, 2:6] * d["aux_std"][:4] + d["aux_mean"][:4]).sum(1)
        print(f"geometric readout (dims + fill, category-free) sMAPE {smape(q_y, geo):.2f}", flush=True)
    np.savez(out_dir / "preds.npz", va_idx=d["va_idx"], out_heldout=out_q, out_val=out_v, history=np.array(hist))
    if not a.smoke:
        torch.save({n: p.detach().cpu() for n, p in named.items()}, out_dir / "trainable_ema.pt")
    restore(bk)


if __name__ == "__main__":
    main()
