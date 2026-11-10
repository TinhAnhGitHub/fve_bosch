#!/usr/bin/env python3
"""Food type and freshness, v2.

Detector: YOLO11 (m or l), trained on every train photo for the final run.
Freshness: timm ConvNeXt-Nano (15.6M) on padded GT crops, 3-way softmax per
Group A food. 5-fold CV by photo gives out-of-fold probabilities, which pick:
  * the epoch count for the final fit,
  * alpha: blend of a box's own probs with the mean of same-food boxes in
    the photo (a pack tends to share one grade),
  * two cuts on expected grade E = p1 + 2*p2 (QWK rewards ordinal decode).

  python baselines/v2.py eda
  python baselines/v2.py det [yolo11l.pt] [epochs]
  python baselines/v2.py cv [arch]
  python baselines/v2.py fit
  python baselines/v2.py predict [eval|test]
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ftaf"
YOLO_DIR = DATA / "yolo"
DET = ROOT / "baselines" / "out" / "v2"
OUT = ROOT / "baselines" / "out" / os.environ.get("V2", "v2")  # freshness artifacts; V2=v2_384 etc. for side-by-side runs
IMGSZ = 1280
ARCH = "convnext_nano.in12k_ft_in1k"
SIZE = int(os.environ.get("SIZE", 256))
EPOCHS = 15
FOLDS = 5


def ann():
    import fresh  # same folder

    train = fresh.find_dir("train")
    a = json.loads((train / "annotations.json").read_text())
    cats = sorted(a["categories"], key=lambda c: c["id"])
    ga = sorted(c["id"] for c in cats if fresh.is_group_a(c))
    assert len(cats) == 26 and len(ga) == 10, (len(cats), len(ga))
    for im in a["images"]:
        p = train / "images" / im["file_name"]
        im["path"] = p if p.is_file() else train / im["file_name"]
    return a, cats, ga


def qwk(t, p, n=3):
    import numpy as np

    o = np.zeros((n, n))
    for a, b in zip(t, p):
        o[int(a), int(b)] += 1
    if o.sum() == 0:
        return 0.0
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.arange(n)[:, None] - np.arange(n)[None]) ** 2 / (n - 1) ** 2
    d = (w * e).sum()
    return 1.0 if d == 0 else float(1 - (w * o).sum() / d)


def macro(cats, true, pred):
    s = [qwk(true[cats == c], pred[cats == c]) for c in sorted(set(cats.tolist())) if (cats == c).sum() >= 2]
    return float(sum(s) / len(s))


# ---------------------------------------------------------------- eda


def eda():
    a, cats, ga = ann()
    name = {c["id"]: c["name"] for c in cats}
    n = Counter(o["category_id"] for o in a["annotations"])
    print("boxes per food:", {name[k]: n[k] for k in sorted(n, key=n.get)})
    g = defaultdict(Counter)
    pack = defaultdict(list)
    for o in a["annotations"]:
        if o["category_id"] in ga and o.get("freshness_grade") is not None:
            g[o["category_id"]][o["freshness_grade"]] += 1
            pack[(o["image_id"], o["category_id"])].append(o["freshness_grade"])
    for c in ga:
        packs = [v for (i, k), v in pack.items() if k == c]
        mixed = sum(len(set(v)) > 1 for v in packs)
        print(f"{name[c]:>10} grades {dict(sorted(g[c].items()))} photos {len(packs)} mixed-grade photos {mixed}")
    # How much a box agrees with the rest of its pack: QWK of box vs pack majority (leave-one-out).
    t, p, k = [], [], []
    for (i, c), v in pack.items():
        for j, x in enumerate(v):
            rest = v[:j] + v[j + 1:]
            if rest:
                t.append(x)
                p.append(Counter(rest).most_common(1)[0][0])
                k.append(c)
    import numpy as np

    print(f"pack-majority oracle macro QWK {macro(np.array(k), np.array(t), np.array(p)):.3f} on {len(t)} boxes")


# ---------------------------------------------------------------- detector


def det(weights="yolo11l.pt", epochs=100):
    import fresh
    import torch
    from ultralytics import YOLO

    fresh.prep()  # labels, train.txt (90%), val.txt (10%)
    rows = (YOLO_DIR / "train.txt").read_text().split() + (YOLO_DIR / "val.txt").read_text().split()
    (YOLO_DIR / "all.txt").write_text("\n".join(rows) + "\n")
    y = (YOLO_DIR / "data.yaml").read_text()
    (YOLO_DIR / "all.yaml").write_text(y.replace("train: train.txt", "train: all.txt"))
    full = "--full" in sys.argv
    YOLO(weights).train(
        data=str(YOLO_DIR / ("all.yaml" if full else "data.yaml")),
        epochs=int(epochs), imgsz=IMGSZ, batch=6, device=0 if torch.cuda.is_available() else "cpu",
        workers=4, project=str(DET), name="det_full" if full else "det", exist_ok=True,
        cos_lr=True, close_mosaic=15, patience=200, degrees=5, scale=0.4, mixup=0.1,
    )


def _det_weights():
    for name in ("det_full", "det"):
        p = DET / name / "weights" / "last.pt"
        if p.is_file():
            return str(p)
    raise SystemExit("no detector")


# ---------------------------------------------------------------- freshness


def crops():
    """(padded region PIL, box xywh inside region, grade, food idx 0..9, image_id)."""
    from PIL import Image

    a, _cats, ga = ann()
    idx = {c: i for i, c in enumerate(ga)}
    by = defaultdict(list)
    for o in a["annotations"]:
        if o["category_id"] in ga and o.get("freshness_grade") is not None:
            by[o["image_id"]].append(o)
    skip = set(json.loads(Path(os.environ["EXCLUDE"]).read_text())["hold"]) if os.environ.get("EXCLUDE") else set()
    rows = []
    for im in a["images"]:
        if im["id"] not in by or im["id"] in skip:
            continue
        pic = Image.open(im["path"]).convert("RGB")
        for o in by[im["id"]]:
            region, box = _region(pic, o["bbox"])
            rows.append((region, box, int(o["freshness_grade"]), idx[o["category_id"]], im["id"]))
    print(f"{len(rows)} crops from {len(by)} photos", flush=True)
    return rows


def _region(pic, bbox, pad=0.35):
    x, y, w, h = map(float, bbox)
    x0, y0 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
    x1, y1 = min(pic.width, int(x + w * (1 + pad))), min(pic.height, int(y + h * (1 + pad)))
    return pic.crop((x0, y0, x1, y1)), (x - x0, y - y0, w, h)


class Crops:
    def __init__(self, rows, train, cfg):
        from torchvision import transforms as T

        self.rows, self.train = rows, train
        post = [T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])]
        aug = [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.15, 0.15, 0.05)] if train else []
        self.tf = T.Compose(aug + post)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        region, (x, y, w, h), g, k, _ = self.rows[i]
        if self.train:
            pad = random.uniform(0.0, 0.25)
            x += random.uniform(-0.08, 0.08) * w
            y += random.uniform(-0.08, 0.08) * h
            w *= random.uniform(0.9, 1.1)
            h *= random.uniform(0.9, 1.1)
        else:
            pad = 0.1
        c = region.crop((int(x - pad * w), int(y - pad * h), int(x + w * (1 + pad)), int(y + h * (1 + pad))))
        return self.tf(c.resize((SIZE, SIZE))), g, k


def model(arch, pretrained=True):
    import timm

    m = timm.create_model(arch, pretrained=pretrained, num_classes=30)
    cfg = timm.data.resolve_data_config({}, model=m)
    return m, {"mean": list(cfg["mean"]), "std": list(cfg["std"])}


def logits(m, x, k):
    return m(x).view(-1, 10, 3)[range(len(k)), k]


def probs(m, rows, cfg, device):
    import torch
    from torch.utils.data import DataLoader

    m.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for x, _g, k in DataLoader(Crops(rows, False, cfg), batch_size=64, num_workers=2):
            x, k = x.to(device), k.to(device)
            p = logits(m, x, k).float().softmax(-1) + logits(m, x.flip(-1), k).float().softmax(-1)
            out.append((p / 2).cpu())
    return torch.cat(out).numpy()


def train(rows, arch, epochs, device, val=None, stop=None):
    """Cosine over `epochs`; stop early at `stop` (same schedule in cv and fit)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    m, cfg = model(arch)
    m.to(device)
    n_cat = Counter(r[3] for r in rows)
    n_cg = Counter((r[3], r[2]) for r in rows)
    wts = [1 / n_cat[r[3]] / n_cg[(r[3], r[2])] ** 0.5 for r in rows]
    dl = DataLoader(Crops(rows, True, cfg), batch_size=int(os.environ.get("BS", 32)), num_workers=4, drop_last=True,
                    sampler=WeightedRandomSampler(wts, len(rows), replacement=True))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hist = []
    for ep in range(1, (stop or epochs) + 1):
        m.train()
        for x, g, k in dl:
            x, g, k = x.to(device), g.to(device), k.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(logits(m, x, k).float(), g, label_smoothing=0.05)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if val is not None:
            hist.append(probs(m, val, cfg, device))
        print(f"  epoch {ep} loss {loss.item():.3f}", flush=True)
    return m, cfg, hist


def decode(p, t1=0.5, t2=1.5):
    import numpy as np

    e = p[:, 1] + 2 * p[:, 2]
    return (e > t1).astype(int) + (e > t2).astype(int)


def blend(p, groups, alpha, w=None):
    """alpha*own + (1-alpha)*weighted mean of the same (photo, food) group."""
    import numpy as np

    w = np.ones(len(p)) if w is None else w
    out = p.copy()
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    for idx in by.values():
        ww = w[idx]
        mean = (p[idx] * ww[:, None]).sum(0) / ww.sum() if ww.sum() > 0 else p[idx].mean(0)
        out[idx] = alpha * p[idx] + (1 - alpha) * mean
    return out


def tune(p, true, cats, groups):
    import numpy as np

    best = (-1, None)
    for alpha in np.linspace(0, 1, 11):
        b = blend(p, groups, alpha)
        for t1 in np.arange(0.3, 1.01, 0.05):
            for t2 in np.arange(1.0, 1.71, 0.05):
                if t2 <= t1:
                    continue
                s = macro(cats, true, decode(b, t1, t2))
                if s > best[0]:
                    best = (s, (float(alpha), float(t1), float(t2)))
    return best


def cv(arch=ARCH):
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = crops()
    ids = sorted({r[4] for r in rows})
    random.Random(0).shuffle(ids)
    fold = {i: n % FOLDS for n, i in enumerate(ids)}
    oof = np.zeros((EPOCHS, len(rows), 3))
    for f in range(FOLDS):
        tr = [r for r in rows if fold[r[4]] != f]
        va_i = [n for n, r in enumerate(rows) if fold[r[4]] == f]
        print(f"fold {f} train {len(tr)} val {len(va_i)}", flush=True)
        _m, _c, hist = train(tr, arch, EPOCHS, device, val=[rows[n] for n in va_i])
        for e, p in enumerate(hist):
            oof[e, va_i] = p
    true = np.array([r[2] for r in rows])
    cats = np.array([r[3] for r in rows])
    groups = [(r[4], r[3]) for r in rows]
    per = [macro(cats, true, decode(oof[e])) for e in range(EPOCHS)]
    print("oof macro QWK by epoch:", " ".join(f"{s:.3f}" for s in per))
    ep = int(np.argmax(per)) + 1
    p = oof[ep - 1]
    print(f"epoch {ep}: argmax {macro(cats, true, p.argmax(1)):.4f}  E-cuts {per[ep - 1]:.4f}")
    for alpha in (1.0, 0.5, 0.0):
        print(f"  alpha {alpha}: {macro(cats, true, decode(blend(p, groups, alpha))):.4f}")
    s, (alpha, t1, t2) = tune(p, true, cats, groups)
    print(f"tuned alpha {alpha} cuts {t1:.2f} {t2:.2f} oof macro QWK {s:.4f}")
    for c in range(10):
        m = cats == c
        print(f"  food {c}: n {m.sum()} qwk {qwk(true[m], decode(blend(p, groups, alpha), t1, t2)[m]):.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fresh.json").write_text(json.dumps({"arch": arch, "size": SIZE, "epochs": ep, "alpha": alpha, "t1": t1, "t2": t2, "oof": s}))
    np.save(OUT / f"oof_{arch.split('.')[0]}.npy", p)


def fit():
    import torch

    global SIZE
    cfg = json.loads((OUT / "fresh.json").read_text())
    SIZE = cfg.get("size", 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, norm, _ = train(crops(), cfg["arch"], EPOCHS, device, stop=cfg["epochs"])
    torch.save({"state": m.state_dict(), "norm": norm, **cfg}, OUT / "fresh.pt")
    print("saved", OUT / "fresh.pt")


# ---------------------------------------------------------------- predict


def predict(split="eval"):
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    import fresh

    _a, cats, ga = ann()
    ids = [c["id"] for c in cats]
    gidx = {c: i for i, c in enumerate(ga)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global SIZE
    blob = torch.load(OUT / "fresh.pt", map_location="cpu", weights_only=False)
    SIZE = blob.get("size", 256)
    m, _ = model(blob["arch"], pretrained=False)
    m.load_state_dict(blob["state"])
    m.to(device)
    det = YOLO(_det_weights())
    paths = sorted((fresh.find_dir(split) / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    rows = []
    for path in paths:
        r = det.predict(str(path), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        pic = Image.open(path).convert("RGB")
        dets, crop_rows = [], []
        for (x1, y1, x2, y2), k, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cat = ids[k]
            box = [x1, y1, x2 - x1, y2 - y1]
            d = {"image_id": int(path.stem), "category_id": cat, "bbox": [round(v, 1) for v in box], "score": round(s, 4)}
            if cat in gidx:
                region, rbox = _region(pic, box)
                crop_rows.append((region, rbox, 0, gidx[cat], len(dets)))
            else:
                d["freshness_grade"] = None
            dets.append(d)
        if crop_rows:
            # ponytail: classifies every Group A box (up to 300); cap to top-k by score if T4 latency is tight.
            p = probs(m, crop_rows, blob["norm"], device)
            w = np.array([dets[c[4]]["score"] for c in crop_rows])
            w = np.where(w >= 0.25, w, 0.0)  # only confident boxes define the pack grade
            p = blend(p, [c[3] for c in crop_rows], blob["alpha"], w)
            for c, g, pp in zip(crop_rows, decode(p, blob["t1"], blob["t2"]), p):
                dets[c[4]]["freshness_grade"] = int(g)
                dets[c[4]]["freshness_score"] = round(float(pp[g]), 4)
        rows += dets
        print(path.stem, len(dets), flush=True)
    sub = ROOT / "submissions" / OUT.name / f"{split}.json"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(json.dumps(rows))
    print(Counter(r.get("freshness_grade") for r in rows), len(rows), "->", sub)


def _selftest():
    import numpy as np

    t = np.array([0, 1, 2, 2])
    assert qwk(t, t) == 1.0 and qwk(t, 2 - t) < 0
    p = np.array([[1, 0, 0], [0, 0, 1.0]])
    assert decode(p).tolist() == [0, 2]
    assert np.allclose(blend(p, [1, 1], 0.0), [[0.5, 0, 0.5]] * 2)
    assert np.allclose(blend(p, [1, 2], 0.0), p)
    print("ok")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    cmd, *args = sys.argv[1:] or ["test"]
    args = [a for a in args if not a.startswith("--")]
    {"eda": eda, "det": det, "cv": cv, "fit": fit, "predict": predict, "test": _selftest}[cmd](*args)
