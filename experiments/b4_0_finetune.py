# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "transformers", "peft", "pillow", "numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B4.0: LoRA fine-tune DINOv3-L/16 on ONE CV fold (seed 0, fold 0) and compare with frozen features.

Network: DINOv3-L/16 (bf16 base, fp32 LoRA r=16 on q/k/v, gradient checkpointing) + MLP head from
[CLS, mean-patch] -> size correction r, P(oversized), and mesh-derived aux targets (3 sorted ln bbox dims, ln fill).
Targets r / cluster labels / reference values come from the fold's TRAINING rows only (same as B3).

Reports, on the held-out fold only:
  net heads        : the network's own r_hat / p_large + cluster values   (clean, no leakage)
  frozen DINOv3-L  : B3 ridge+classifier on frozen features, same fold; and + kNN
  frozen B3.4      : DINOv2 + DINOv3-H+ ridge/classifier/kNN, same fold (current best)
  fine-tuned + kNN : kNN on fine-tuned CLS features (DIAGNOSTIC: train features saw their labels)

Run:  uv run experiments/b4_0_finetune.py [--smoke] [--epochs 10]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))
import b3_4_knn as knn  # noqa: E402
from b0_b1 import DATA, smape  # noqa: E402
from b3 import build_table, cluster_labels  # noqa: E402
from frozen import Config, Experiment  # noqa: E402

for line in (HERE.parent / ".env").read_text().splitlines():
    if line.startswith("HUGGINGFACE_API_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SIZE, N_REG = 512, 4
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GAP = math.log(30.0)
KNN_K, W_R, W_P = 2, 0.25, 0.25
knn.KS = (KNN_K,)


# ------------------------------------------------------------------ data
def load_images(paths: list[Path]) -> torch.Tensor:
    """uint8 [N, 3, SIZE, SIZE], resized without crop (backdrop kept)."""
    def one(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
    with ThreadPoolExecutor(16) as ex:
        arr = np.stack(list(ex.map(one, paths)))
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x: float [B,3,H,W] in [0,1] on GPU. Flip, +-10 deg rotation + 5% shift (backdrop fill), brightness/contrast."""
    out = []
    for img in x:
        if rng.random() < 0.5:
            img = img.flip(-1)
        border = torch.cat([img[:, :8].flatten(1), img[:, -8:].flatten(1)], 1).median(1).values.tolist()
        img = TF.affine(img, angle=float(rng.uniform(-10, 10)),
                        translate=[int(rng.uniform(-0.05, 0.05) * SIZE), int(rng.uniform(-0.05, 0.05) * SIZE)],
                        scale=1.0, shear=[0.0], fill=border)
        b, c = rng.uniform(-0.15, 0.15), rng.uniform(0.85, 1.15)
        img = ((img - img.mean()) * c + img.mean() + b).clamp(0, 1)
        out.append(img)
    return torch.stack(out)


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ------------------------------------------------------------------ model
class Net(nn.Module):
    """DINOv3-L/16 + LoRA, and a small head on [CLS, mean-patch]."""

    N_AUX = 4

    def __init__(self) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                        target_modules=["q_proj", "k_proj", "v_proj"]))
        d = base.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, 2 + self.N_AUX)).cuda()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(pixel_values=x.to(torch.bfloat16)).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1 + N_REG:].mean(1)], dim=1).float()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        return self.head(f), f[:, : f.shape[1] // 2]


@torch.inference_mode()
def predict(net: Net, imgs: torch.Tensor, batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Heads output and CLS features, averaged over the image and its mirror."""
    net.eval()
    outs, feats = [], []
    for i in range(0, len(imgs), batch):
        x = normalise(imgs[i:i + batch].cuda().float() / 255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (o1, f1), (o2, f2) = net(x), net(x.flip(-1))
        outs.append(((o1 + o2) / 2).float().cpu().numpy())
        feats.append(((f1 + f2) / 2).float().cpu().numpy())
    return np.concatenate(outs), np.concatenate(feats)


# ------------------------------------------------------------------ targets / decoding
def make_targets(T, rows_gt: list[dict]) -> dict[str, np.ndarray]:
    """Per training row: r, r-valid mask, oversized label, mixed-category mask, standardised aux targets."""
    u = cluster_labels(T)
    table, ref_loo = build_table(T, u)
    r = T["ln"].to_numpy() - ref_loo
    ok = np.isfinite(r) & (np.abs(r) < GAP)
    mixed_cats = {c for c, v in table.values.items() if len(v) == 2}
    mixed = T["category"].isin(mixed_cats).to_numpy()
    dims = np.log(np.sort(np.array([g["bbox_mm"] for g in rows_gt]), axis=1) / 10)  # ln cm, sorted
    fill = np.log(T["volume"].to_numpy() / np.prod(np.exp(dims), axis=1))
    aux = np.column_stack([dims, fill])
    mu, sd = aux.mean(0), aux.std(0) + 1e-6
    return {"u": u, "table": table, "r": np.nan_to_num(r), "ok": ok, "mixed": mixed, "aux": (aux - mu) / sd}


def build_o(table, T_u_rate, Q, r_hat: np.ndarray, p_large: np.ndarray) -> dict[str, np.ndarray]:
    """Assemble the arrays knn.decode expects, from heads outputs (clipped r, sigmoid p)."""
    n = len(Q)
    ln, ref_n, ref_l, pl = np.empty(n), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i, c in enumerate(Q["category"].to_numpy()):
        v = table.values.get(c)
        if v is None:
            ln[i] = table.fallback
            continue
        ref_n[i], ref_l[i] = v.get(0, np.nan), v.get(1, np.nan)
        if len(v) == 1:
            ln[i] = next(iter(v.values()))
        else:
            pl[i] = p_large[i]
            ln[i] = v[1] if p_large[i] > 0.5 else v[0]
    return {"ln": ln + r_hat, "r_hat": r_hat, "p_large": pl, "ref_n": ref_n, "ref_l": ref_l}


def score(y: np.ndarray, ln: np.ndarray, mask: np.ndarray) -> float:
    return smape(y, np.exp(ln))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    exp = Experiment()
    tr_idx, va_idx = exp.splits(0)[args.fold]
    if args.smoke:
        tr_idx, va_idx = tr_idx[:48], va_idx[:24]
    T, Q = exp.train.iloc[tr_idx].reset_index(drop=True), exp.train.iloc[va_idx].reset_index(drop=True)
    y_q = Q["volume"].to_numpy()
    print(f"fold {args.fold} of seed 0: train {len(T)}, held-out {len(Q)}", flush=True)

    def gt(row) -> dict:
        return json.loads((DATA / "train" / row.group / row.category / row.sample_id / "gt.json").read_text())
    tg = make_targets(T, [gt(r) for r in T.itertuples()])
    t0 = time.time()
    path = lambda d: [DATA / "train" / r.group / r.category / r.sample_id / "rgb_industrial.png" for r in d.itertuples()]
    imgs_t, imgs_q = load_images(path(T)), load_images(path(Q))
    print(f"images loaded in {time.time() - t0:.0f}s ({imgs_t.shape})", flush=True)

    # ---- frozen baselines on the SAME fold (computed first; cheap)
    def frozen(feats) -> tuple[float, float]:
        cfg = Config(name="frozen", features=feats, n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
        X, _ = exp.features(cfg)
        Z_tr, Z_va = exp._project(X[tr_idx], X[va_idx], None)
        o = exp.fit_predict(T, Q, Z_tr, Z_va, cfg)
        a = score(y_q, o["ln"], None)
        for k, parts in knn.knn_parts(T, Q, Z_tr, Z_va).items():
            for name, v in parts.items():
                o[f"{name}_{k}"] = v
        return a, score(y_q, knn.decode(o, KNN_K, W_R, W_P), None)
    CLS = (0, 2, 4)
    fz_l = frozen((("dinov3_vitl16_512", CLS),))
    fz_best = frozen((("dinov2l_reg_518", CLS), ("dinov3_vith16plus_512", CLS)))
    print(f"frozen DINOv3-L/16 : no-kNN {fz_l[0]:.2f} | +kNN {fz_l[1]:.2f}\n"
          f"frozen B3.4 (v2+v3H): no-kNN {fz_best[0]:.2f} | +kNN {fz_best[1]:.2f}   (this fold; B3.4 CV avg is 40.82)",
          flush=True)

    # ---- fine-tune
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Net()
    lora = [p for p in net.backbone.parameters() if p.requires_grad]
    print(f"trainable: LoRA {sum(p.numel() for p in lora) / 1e6:.2f}M + head "
          f"{sum(p.numel() for p in net.head.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": 2e-4}, {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    steps_per_epoch = len(T) // args.batch
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    r_t = torch.tensor(tg["r"], dtype=torch.float32)
    ok_t = torch.tensor(tg["ok"])
    u_t = torch.tensor(tg["u"], dtype=torch.float32)
    mx_t = torch.tensor(tg["mixed"])
    aux_t = torch.tensor(tg["aux"], dtype=torch.float32)

    def evaluate() -> tuple[float, float, np.ndarray]:
        out, feat = predict(net, imgs_q)
        r_hat = np.clip(out[:, 0], -2.5, 2.5)
        p = 1 / (1 + np.exp(-out[:, 1]))
        o = build_o(tg["table"], None, Q, r_hat, p)
        return score(y_q, o["ln"], None), float(np.corrcoef(r_hat, np.nan_to_num(
            Q["ln"].to_numpy() - np.where(np.isfinite(o["ref_n"]), o["ref_n"], o["ref_l"])))[0, 1]), feat

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(len(T))
        run = []
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x = normalise(augment(imgs_t[idx].cuda().float() / 255, rng))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = net(x)
            out = out.float()
            ii = torch.as_tensor(idx)
            l_r = (F.smooth_l1_loss(out[:, 0].cpu(), r_t[ii], reduction="none", beta=0.3) * ok_t[ii]).sum() \
                / ok_t[ii].sum().clamp(min=1)
            l_u = (F.binary_cross_entropy_with_logits(out[:, 1].cpu(), u_t[ii], reduction="none") * mx_t[ii]).sum() \
                / mx_t[ii].sum().clamp(min=1)
            l_a = F.l1_loss(out[:, 2:].cpu(), aux_t[ii])
            loss = l_r + 0.5 * l_u + 0.1 * l_a
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*lora, *net.head.parameters()], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append((loss.item(), l_r.item(), l_u.item(), l_a.item()))
            step += 1
            if args.smoke and step in (5, 10, 20):
                print(f"  step {step}: loss {np.mean(run[-5:], 0).round(3)}  peak VRAM "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB  {(time.time() - t0) / step:.2f}s/step", flush=True)
        m = np.mean(run, 0)
        msg = f"epoch {ep + 1}/{args.epochs}  loss {m[0]:.3f} (r {m[1]:.3f}, mode {m[2]:.3f}, aux {m[3]:.3f})  " \
              f"{time.time() - t0:.0f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
        if (ep + 1) % 2 == 0 or ep + 1 == args.epochs:
            sm, rho, _ = evaluate()
            msg += f"  | held-out sMAPE {sm:.2f}  corr(r_hat, r) {rho:.3f}"
        print(msg, flush=True)

    # ---- final comparison on the held-out fold
    sm, rho, feat_q = evaluate()
    print(f"\nFINE-TUNED heads, held-out fold: sMAPE {sm:.2f}  (corr r {rho:.3f})")
    _, feat_t = predict(net, imgs_t)
    out_q, _ = predict(net, imgs_q)
    o = build_o(tg["table"], None, Q, np.clip(out_q[:, 0], -2.5, 2.5), 1 / (1 + np.exp(-out_q[:, 1])))
    for k, parts in knn.knn_parts(T, Q, feat_t, feat_q).items():
        for name, v in parts.items():
            o[f"{name}_{k}"] = v
    print(f"fine-tuned + kNN (diagnostic, leaky train features): {score(y_q, knn.decode(o, KNN_K, W_R, W_P), None):.2f}")
    print(f"\nSAME FOLD:  frozen L/16 {fz_l[0]:.2f}/{fz_l[1]:.2f} (no-kNN/+kNN) | frozen B3.4 {fz_best[0]:.2f}/"
          f"{fz_best[1]:.2f} | fine-tuned heads {sm:.2f}")
    if not args.smoke:
        out = HERE / f"out_b4_0_f{args.fold}"
        net.backbone.save_pretrained(out)
        torch.save(net.head.state_dict(), out / "head.pt")


if __name__ == "__main__":
    main()
