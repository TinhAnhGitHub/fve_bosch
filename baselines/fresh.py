#!/usr/bin/env python3
"""Food type and freshness.

Boxes and the 26 food types: YOLO11m (Jocher and Qiu, Ultralytics, 2024),
COCO-pretrained. 20.1M parameters, so it stays inside the 40M / T4 budget.

Freshness on the 10 Group A foods: CORAL (Cao, Mirjalili, Raschka, 2019,
arXiv 1901.07884). Three ordered grades, fresh / semi-fresh / spoiled, become
two shared-weight thresholds. A fresh-to-spoiled miss is a longer jump than
a fresh-to-semi-fresh miss, which is what quadratic weighted kappa penalises.
One MobileNetV3-Small readout for every food. Two crops of the same food are
mixed (Zhang, Cisse, Dauphin, Lopez-Paz, 2018), so a grade can sit between
its neighbors. Hand-measured color does not survive a new photo.

Group B gets a box and a type only. Freshness is left null.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ftaf"
YOLO = DATA / "yolo"
OUT = ROOT / "baselines" / "out" / "fresh"
SUB = ROOT / "submissions" / "fresh" / "preds.json"
IMGSZ = 1280
RARE_BOXES = 100
GROUP_A = {
    "blueberry", "cherry", "grape", "lychee", "strawberry",
    "beef", "chicken", "lamb", "pork", "salmon",
}

# ponytail: fixed 1280. Drop to 960 if a T4 p95 goes over 150 ms.


def _det() -> str:
    found = list((OUT / "det").rglob("last.pt")) or list((OUT / "det").rglob("best.pt"))
    if not found:
        raise SystemExit("no detector weights")
    return str(found[0])


def find_dir(kind: str) -> Path:
    for path in DATA.rglob(kind):
        if path.is_dir() and ((path / "images").is_dir() or (path / "annotations.json").is_file()):
            return path
    raise SystemExit(f"no {kind}/ under {DATA}")


def categories(ann: dict) -> list[dict]:
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    if len(cats) != 26:
        raise SystemExit(f"expected 26 categories, got {len(cats)}")
    return cats


def is_group_a(cat: dict) -> bool:
    return str(cat.get("supercategory", "")).lower() == "group_a" or cat["name"].lower() in GROUP_A


def prep() -> None:
    train = find_dir("train")
    ann = json.loads((train / "annotations.json").read_text())
    cats = categories(ann)
    index = {c["id"]: i for i, c in enumerate(cats)}
    group_a = {c["id"] for c in cats if is_group_a(c)}
    boxes: dict[int, list] = {im["id"]: [] for im in ann["images"]}
    counts = {c["id"]: 0 for c in cats}
    for obj in ann["annotations"]:
        boxes[obj["image_id"]].append(obj)
        counts[obj["category_id"]] += 1
    rare = {cid for cid, n in counts.items() if n < RARE_BOXES}

    for path in (YOLO / "images" / "train", YOLO / "labels" / "train"):
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    for im in ann["images"]:
        src = train / "images" / im["file_name"]
        if not src.is_file():
            src = train / im["file_name"]
        if not src.is_file():
            raise SystemExit(f"missing image {im['file_name']}")
        objs = boxes[im["id"]]
        worst = min((counts[obj["category_id"]] for obj in objs), default=999)
        reps = 1 if worst >= RARE_BOXES else min(8, max(2, RARE_BOXES // worst))
        for rep in range(reps):
            stem = src.stem if rep == 0 else f"{src.stem}__{rep}"
            dst = YOLO / "images" / "train" / f"{stem}{src.suffix}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
            lines = []
            for obj in objs:
                x, y, w, h = obj["bbox"]
                cx = min(1.0, max(0.0, (x + w / 2) / im["width"]))
                cy = min(1.0, max(0.0, (y + h / 2) / im["height"]))
                lines.append(
                    f"{index[obj['category_id']]} {cx:.6f} {cy:.6f} "
                    f"{min(1.0, max(0.0, w / im['width'])):.6f} {min(1.0, max(0.0, h / im['height'])):.6f}"
                )
            (YOLO / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            rows.append((str(dst), reps > 1, im["id"]))

    rng = random.Random(0)
    rng.shuffle(rows)
    common = [row for row in rows if not row[1]]
    hold = {row[2] for row in common[: max(1, len(common) // 10)]}
    train_txt = [path for path, _rare, image_id in rows if image_id not in hold]
    val_txt = [path for path, _rare, image_id in rows if image_id in hold and "__" not in Path(path).stem]
    if not val_txt:
        cut = max(1, len(train_txt) // 10)
        val_txt, train_txt = train_txt[:cut], train_txt[cut:]
    (YOLO / "train.txt").write_text("\n".join(train_txt) + "\n")
    (YOLO / "val.txt").write_text("\n".join(val_txt) + "\n")
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))
    (YOLO / "data.yaml").write_text(f"path: {YOLO}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps({
        "ids": [c["id"] for c in cats],
        "names": [c["name"] for c in cats],
        "group_a": sorted(group_a),
        "holdout": sorted(hold),
    }))
    print(f"train {len(train_txt)} images, val {len(val_txt)}, rare {sorted(rare)}")


def train_det() -> None:
    import torch
    from ultralytics import YOLO as Detector

    gpu = torch.cuda.is_available()
    model = Detector("yolo11m.pt")
    model.train(
        data=str(YOLO / "data.yaml"),
        epochs=60,
        patience=60,
        imgsz=IMGSZ,
        batch=4 if gpu else 2,
        device=0 if gpu else "cpu",
        workers=2 if gpu else 0,
        project=str(OUT),
        name="det",
        exist_ok=True,
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,
    )


def group_a_crops():
    train = find_dir("train")
    ann = json.loads((train / "annotations.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    group_a = set(meta["group_a"])
    by_id = {im["id"]: im for im in ann["images"]}
    samples = []
    for obj in ann["annotations"]:
        if obj["category_id"] not in group_a or obj.get("freshness_grade") is None:
            continue
        im = by_id[obj["image_id"]]
        src = train / "images" / im["file_name"]
        if not src.is_file():
            src = train / im["file_name"]
        samples.append((src, obj["bbox"], int(obj["freshness_grade"]), obj["category_id"], obj["image_id"]))
    return samples


class _Crops:
    def __init__(self, samples, train: bool):
        from torchvision import transforms

        jitter = [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip()] if train else [transforms.Resize((224, 224))]
        self.tf = transforms.Compose(
            jitter
            + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        path, bbox, grade, cat = self.samples[i][:4]
        im = Image.open(path).convert("RGB")
        return self.tf(_crop(im, bbox, jitter=self.train)), grade, cat - 1


def _crop(im, bbox, pad: float = 0.1, jitter: bool = False):
    import random

    x, y, w, h = [float(v) for v in bbox]
    if jitter:
        pad = random.uniform(0.0, 0.2)
        x += random.uniform(-0.08, 0.08) * w
        y += random.uniform(-0.08, 0.08) * h
    box = (
        max(0, int(x - pad * w)),
        max(0, int(y - pad * h)),
        min(im.width, int(x + w * (1 + pad))),
        min(im.height, int(y + h * (1 + pad))),
    )
    return im.crop(box)


def _fresh(device):
    import torch
    from torchvision import models

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    dim = net.classifier[0].in_features
    net.classifier = torch.nn.Identity()
    net = net.to(device)
    head = torch.nn.Linear(dim, 1, bias=False).to(device)
    bias = torch.nn.Parameter(torch.zeros(10, 2, device=device))
    return net, head, bias


def train_fresh() -> None:
    import torch
    import torch.nn.functional as F
    from collections import Counter
    from torch.utils.data import DataLoader, WeightedRandomSampler

    meta = json.loads((OUT / "meta.json").read_text())
    hold = set(meta["holdout"])
    samples = group_a_crops() + matched_crops()
    train_samples = [s for s in samples if s[4] not in hold]
    val_samples = [s for s in samples if s[4] in hold]
    if not train_samples:
        raise SystemExit("no Group A crops")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head, bias = _fresh(device)
    opt = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": 1e-4},
            {"params": [head.weight, bias], "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    counts = Counter((s[3], s[2]) for s in train_samples)
    weights = [1.0 / counts[(s[3], s[2])] for s in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(_Crops(train_samples, True), batch_size=32, sampler=sampler, num_workers=0)
    (OUT / "thresh.json").unlink(missing_ok=True)
    state = None
    for epoch in range(1, 31):
        net.train()
        total, seen = 0.0, 0
        for x, y, k in loader:
            x, y, k = x.to(device), y.to(device), k.to(device)
            perm = torch.randperm(len(x), device=device)
            same = k[perm] == k
            lam = torch.distributions.Beta(0.4, 0.4).sample((len(x),)).to(device)
            lam = torch.where(same, lam, torch.ones_like(lam))
            x = lam[:, None, None, None] * x + (1 - lam)[:, None, None, None] * x[perm]
            logit = _logit(net, head, bias, x, k)
            target = torch.stack([(y > 0).float(), (y > 1).float()], -1)
            other = torch.stack([(y[perm] > 0).float(), (y[perm] > 1).float()], -1)
            target = lam[:, None] * target + (1 - lam)[:, None] * other
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        score = _holdout_qwk(net, head, bias, val_samples, device)
        print(f"epoch {epoch} loss {total / seen:.4f} holdout_qwk {score:.4f}", flush=True)
        state = _state(net, head, bias)
    torch.save(state, OUT / "coral.pt")
    print(f"saved {OUT / 'coral.pt'} qwk {score:.4f}")


def _holdout_qwk(net, head, bias, samples, device) -> float:
    if len(samples) < 2:
        return 0.0
    import numpy as np
    import torch

    net.eval()
    loader = _Crops(samples, False)
    preds, trues, cats = [], [], []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            x, _grade, k = loader[i]
            grade_hat = int((torch.sigmoid(_logit(net, head, bias, x[None].to(device), k)) > 0.5).sum())
            preds.append(grade_hat)
            trues.append(sample[2])
            cats.append(sample[3])
    scores = []
    for cat in set(cats):
        yt = [t for t, c in zip(trues, cats) if c == cat]
        yp = [p for p, c in zip(preds, cats) if c == cat]
        if len(yt) < 2:
            continue
        scores.append(_qwk(np.asarray(yt), np.asarray(yp)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _qwk(y_true, y_pred, n: int = 3) -> float:
    import numpy as np

    o = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        o[int(t), int(p)] += 1
    if o.sum() == 0:
        return 0.0
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    den = float((w * e).sum())
    return 1.0 if den == 0 else float(1 - (w * o).sum() / den)


def _logit(net, head, bias, x, k):
    import torch

    index = k if torch.is_tensor(k) else torch.tensor(k, device=x.device)
    bb = bias[index]
    if bb.dim() == 1:
        bb = bb.view(1, -1)
    return head(net(x)) + bb


def _state(net, head, bias):
    return {
        "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "head": head.weight.detach().cpu(),
        "bias": bias.detach().cpu(),
    }


def _load_coral(device):
    import torch
    from torchvision import models

    blob = torch.load(OUT / "coral.pt", map_location=device, weights_only=True)
    if "bias" not in blob or "color" in blob:
        raise RuntimeError("old head")
    net = models.mobilenet_v3_small(weights=None)
    net.classifier = torch.nn.Identity()
    net.load_state_dict(blob["net"])
    head = torch.nn.Linear(blob["head"].shape[1], 1, bias=False)
    head.weight.data.copy_(blob["head"])
    return net.to(device).eval(), head.to(device).eval(), blob["bias"].to(device)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _grade(p0: float, p1: float, t0: float, t1: float) -> int:
    return int(p0 > t0) + int(p1 > t1)


def _macro(rows, thresh) -> float:
    import numpy as np

    scores = []
    for cat in range(1, 11):
        sub = [r for r in rows if r[1] == cat]
        if len(sub) < 2:
            continue
        yt = np.asarray([r[2] for r in sub])
        yp = np.asarray([_grade(r[3], r[4], *thresh[cat]) for r in sub])
        scores.append(_qwk(yt, yp))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _search(rows):
    import numpy as np

    grid = (0.3, 0.4, 0.5, 0.6, 0.7)
    thresh = {}
    for cat in range(1, 11):
        sub = [r for r in rows if r[1] == cat]
        best = (0.5, 0.5)
        if len(sub) >= 8:
            yt = np.asarray([r[2] for r in sub])
            best_s = _qwk(yt, np.asarray([_grade(r[3], r[4], *best) for r in sub]))
            for t0 in grid:
                for t1 in grid:
                    s = _qwk(yt, np.asarray([_grade(r[3], r[4], t0, t1) for r in sub]))
                    if s > best_s:
                        best, best_s = (t0, t1), s
            print(f"class {cat} n {len(sub)} thresh {best} qwk {best_s:.3f}", flush=True)
        thresh[cat] = best
    return thresh


def fit() -> None:
    import torch
    from PIL import Image
    from ultralytics import YOLO

    meta = json.loads((OUT / "meta.json").read_text())
    group_a = set(meta["group_a"])
    ids = meta["ids"]
    train = find_dir("train")
    ann = json.loads((train / "annotations.json").read_text())
    gt = {}
    for obj in ann["annotations"]:
        if obj["category_id"] in group_a and obj.get("freshness_grade") is not None:
            gt.setdefault(obj["image_id"], []).append(obj)
    images = []
    for im in ann["images"]:
        src = train / "images" / im["file_name"]
        if not src.is_file():
            src = train / im["file_name"]
        if src.is_file() and im["id"] in gt:
            images.append((im["id"], src))
    det = YOLO(_det())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head, bias = _load_coral(device)
    tf = _Crops([], False).tf
    pairs = []
    for start in range(0, len(images), 16):
        chunk = images[start : start + 16]
        results = det.predict(
            [str(p) for _, p in chunk],
            imgsz=IMGSZ,
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=4,
            verbose=False,
        )
        for (image_id, path), result in zip(chunk, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            im = Image.open(path).convert("RGB")
            preds = []
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), k, score in zip(xyxy, cls, conf):
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                cat = ids[int(k)]
                if cat not in group_a:
                    continue
                x = tf(_crop(im, (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
                with torch.no_grad():
                    prob = torch.sigmoid(_logit(net, head, bias, x[None].to(device), cat - 1))[0]
                preds.append((float(score), cat, [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(prob[0]), float(prob[1])))
            used = set()
            for _score, cat, box, p0, p1 in sorted(preds, key=lambda r: -r[0]):
                best_i, best_u = None, 0.5
                for i, obj in enumerate(gt[image_id]):
                    if i in used or obj["category_id"] != cat:
                        continue
                    u = _iou(box, obj["bbox"])
                    if u >= best_u:
                        best_i, best_u = i, u
                if best_i is None:
                    continue
                used.add(best_i)
                pairs.append((image_id, cat, int(gt[image_id][best_i]["freshness_grade"]), p0, p1))
        print(f"fit {start + len(chunk)}/{len(images)} matched {len(pairs)}", flush=True)
    import numpy as np

    base = {c: (0.5, 0.5) for c in range(1, 11)}
    hold = set(meta["holdout"])
    # Meats never sit in the holdout, so a cut fit on them is just the training set.
    train_rows = [r for r in pairs if r[0] not in hold]
    hold_rows = [r for r in pairs if r[0] in hold]
    chosen = _search(train_rows)
    for cat in range(1, 11):
        sub = [r for r in pairs if r[1] == cat]
        if len(sub) < 2:
            continue
        yt = np.asarray([r[2] for r in sub])
        yp = np.asarray([_grade(r[3], r[4], 0.5, 0.5) for r in sub])
        hsub = [r for r in hold_rows if r[1] == cat]
        line = f"class {cat} n {len(sub)} base {_qwk(yt, yp):.3f} two {int((np.abs(yt - yp) == 2).sum())}"
        if len(hsub) >= 8:
            hyt = np.asarray([r[2] for r in hsub])
            hb = np.asarray([_grade(r[3], r[4], 0.5, 0.5) for r in hsub])
            hf = np.asarray([_grade(r[3], r[4], *chosen[cat]) for r in hsub])
            line += f" hold {_qwk(hyt, hb):.3f}->{_qwk(hyt, hf):.3f} n {len(hsub)}"
            if _qwk(hyt, hf) <= _qwk(hyt, hb):
                chosen[cat] = (0.5, 0.5)
        else:
            chosen[cat] = (0.5, 0.5)
        print(line, flush=True)
    base_s, fit_s = _macro(hold_rows, base), _macro(hold_rows, chosen)
    print(f"hold n {len(hold_rows)} base {base_s:.4f} fit {fit_s:.4f}", flush=True)
    if fit_s > base_s:
        (OUT / "thresh.json").write_text(json.dumps({str(k): list(v) for k, v in chosen.items()}))
    else:
        print("keep 0.5", flush=True)


def _matched(image_ids, by_image):
    import torch
    from PIL import Image
    from ultralytics import YOLO

    meta = json.loads((OUT / "meta.json").read_text())
    ids = meta["ids"]
    group_a = set(meta["group_a"])
    det = YOLO(_det())
    images = [(i, by_image[i][0][0]) for i in sorted(image_ids)]
    boxes = []
    for start in range(0, len(images), 16):
        chunk = images[start : start + 16]
        results = det.predict(
            [str(p) for _, p in chunk], imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, batch=4, verbose=False,
        )
        for (image_id, path), result in zip(chunk, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            im = Image.open(path).convert("RGB")
            preds = []
            for (x1, y1, x2, y2), k, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                cat = ids[int(k)]
                if cat not in group_a:
                    continue
                crop = _Crops([], False).tf(_crop(im, (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
                preds.append((float(score), cat, [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], crop))
            used = set()
            gt = by_image[image_id]
            for _score, cat, box, crop in sorted(preds, key=lambda r: -r[0]):
                best_i, best_u = None, 0.5
                for i, sample in enumerate(gt):
                    if i in used or sample[3] != cat:
                        continue
                    u = _iou(box, sample[1])
                    if u >= best_u:
                        best_i, best_u = i, u
                if best_i is None:
                    continue
                used.add(best_i)
                boxes.append((crop, gt[best_i][2], cat))
    return boxes


def _score_boxes(net, head, bias, boxes, device):
    import numpy as np
    import torch

    net.eval()
    x = torch.stack([b[0] for b in boxes]).to(device)
    k = torch.tensor([b[2] - 1 for b in boxes], device=device)
    grades = []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            part = _logit(net, head, bias, x[start : start + 32], k[start : start + 32])
            grades.append((torch.sigmoid(part) > 0.5).sum(1))
    grade = torch.cat(grades).cpu().numpy()
    scores, parts = [], []
    cats = np.asarray([b[2] for b in boxes])
    true = np.asarray([b[1] for b in boxes])
    for cat in range(1, 11):
        m = cats == cat
        if int(m.sum()) < 2:
            parts.append(f"{cat}:.")
            continue
        q = _qwk(true[m], grade[m])
        scores.append(q)
        parts.append(f"{cat}:{q:.2f}")
    gap = np.abs(true.astype(int) - grade.astype(int))
    bad = int((gap > 0).sum())
    parts.append(f"two:{int((gap == 2).sum())}/{bad}")
    macro = float(sum(scores) / len(scores)) if scores else 0.0
    return macro, " ".join(parts)


def matched_crops() -> list:
    """Detector boxes that land on a Group A label. QWK only scores those."""
    from ultralytics import YOLO

    meta = json.loads((OUT / "meta.json").read_text())
    ids = meta["ids"]
    group_a = set(meta["group_a"])
    ann = json.loads((find_dir("train") / "annotations.json").read_text())
    by_id = {im["id"]: im for im in ann["images"]}
    by_image: dict = {}
    for obj in ann["annotations"]:
        if obj["category_id"] in group_a and obj.get("freshness_grade") is not None:
            by_image.setdefault(obj["image_id"], []).append(obj)
    train = find_dir("train")
    paths, image_ids = [], []
    for iid, im in ((i, by_id[i]) for i in by_image):
        src = train / "images" / im["file_name"]
        if not src.is_file():
            src = train / im["file_name"]
        if src.is_file():
            paths.append(src)
            image_ids.append(iid)
    det = YOLO(_det())
    samples = []
    for start in range(0, len(paths), 8):
        chunk_p, chunk_i = paths[start : start + 8], image_ids[start : start + 8]
        results = det.predict([str(p) for p in chunk_p], imgsz=IMGSZ, conf=0.05, iou=0.7, max_det=300, verbose=False)
        for path, iid, result in zip(chunk_p, chunk_i, results):
            if result.boxes is None:
                continue
            preds = []
            for (x1, y1, x2, y2), k in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                preds.append((ids[int(k)], (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
            used = set()
            for obj in by_image[iid]:
                best_i, best_u = -1, 0.5
                for i, (cat, box) in enumerate(preds):
                    if i in used or cat != obj["category_id"]:
                        continue
                    u = _iou(box, obj["bbox"])
                    if u >= best_u:
                        best_i, best_u = i, u
                if best_i < 0:
                    continue
                used.add(best_i)
                cat, box = preds[best_i]
                samples.append((path, box, int(obj["freshness_grade"]), cat, iid))
    print(f"matched {len(samples)} detector crops", flush=True)
    return samples


def cv() -> None:
    import torch
    import torch.nn.functional as F
    from collections import Counter
    from torch.utils.data import DataLoader

    samples = group_a_crops()
    by_image = {}
    for sample in samples:
        by_image.setdefault(sample[4], []).append(sample)
    ids = sorted(by_image)
    rng = random.Random(0)
    rng.shuffle(ids)
    hold = set(ids[: max(1, len(ids) // 5)])
    for cat in range(1, 11):
        have = sum(s[3] == cat for i in hold for s in by_image[i])
        if have >= 2:
            continue
        for image_id in ids:
            if image_id in hold:
                continue
            if any(s[3] == cat for s in by_image[image_id]):
                hold.add(image_id)
                break
    boxes = _matched(hold, by_image)
    print("hold", len(hold), "matched", len(boxes), Counter(b[2] for b in boxes), flush=True)
    if len(boxes) < 10:
        raise SystemExit("holdout has too few matched boxes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (OUT / "coral.pt").exists():
        try:
            old = _load_coral(device)
            old_score, old_detail = _score_boxes(*old, boxes, device)
            print(f"current cv_qwk {old_score:.4f} {old_detail}", flush=True)
            del old
        except Exception as exc:
            print("current skipped", type(exc).__name__, flush=True)
    train_samples = [s for s in matched_crops() if s[4] not in hold]
    if len(train_samples) < 30:
        train_samples = [s for s in samples if s[4] not in hold]
    net, head, bias = _fresh(device)
    opt = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": 1e-4},
            {"params": [head.weight, bias], "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    class _Batches:
        # One crop per photo and three photos per food, so a pack cannot be memorised.
        def __init__(self, rows):
            self.by = {}
            for i, s in enumerate(rows):
                self.by.setdefault(s[3], {}).setdefault(s[4], []).append(i)
            self.steps = max(1, len(rows) // 30)

        def __len__(self):
            return self.steps

        def __iter__(self):
            for _ in range(self.steps):
                batch = []
                for images in self.by.values():
                    keys = list(images)
                    for image_id in random.sample(keys, k=min(3, len(keys))):
                        batch.append(random.choice(images[image_id]))
                random.shuffle(batch)
                yield batch

    loader = DataLoader(_Crops(train_samples, True), batch_sampler=_Batches(train_samples), num_workers=0)
    best = -1.0
    (OUT / "thresh.json").unlink(missing_ok=True)
    for epoch in range(1, 31):
        net.train()
        total, seen = 0.0, 0
        for x, y, k in loader:
            x, y, k = x.to(device), y.to(device), k.to(device)
            logit = _logit(net, head, bias, x, k)
            target = torch.stack([(y > 0).float(), (y > 1).float()], -1)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        score, detail = _score_boxes(net, head, bias, boxes, device)
        mark = ""
        if score > best:
            best = score
            mark = " saved"
            torch.save(_state(net, head, bias), OUT / "coral.pt")
        print(f"epoch {epoch} loss {total / seen:.4f} cv_qwk {score:.4f} {detail}{mark}", flush=True)
    print(f"best cv_qwk {best:.4f}", flush=True)


def predict() -> None:
    import torch
    from PIL import Image
    from ultralytics import YOLO

    meta = json.loads((OUT / "meta.json").read_text())
    group_a = set(meta["group_a"])
    ids = meta["ids"]
    det = YOLO(_det())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head, bias = _load_coral(device)
    tf = _Crops([], False).tf
    try:
        images = find_dir("eval") / "images"
    except SystemExit:
        images = find_dir("val") / "images"
    paths = sorted(images.glob("*.jpg"), key=lambda p: int(p.stem))
    rows = []
    results = det.predict([str(p) for p in paths], imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, verbose=False)
    for path, result in zip(paths, results):
        if result.boxes is None or len(result.boxes) == 0:
            print(path.stem, 0, flush=True)
            continue
        im = Image.open(path).convert("RGB")
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), k, score in zip(xyxy, cls, conf):
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            cat = ids[int(k)]
            row = {
                "image_id": int(path.stem),
                "category_id": cat,
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "score": round(float(score), 4),
            }
            if cat in group_a:
                x = tf(_crop(im, (x1, y1, x2 - x1, y2 - y1)))
                with torch.no_grad():
                    prob = torch.sigmoid(_logit(net, head, bias, x[None].to(device), cat - 1))[0]
                row["_p"] = (float(prob[0]), float(prob[1]))
            else:
                row["freshness_grade"] = None
            rows.append(row)
        print(path.stem, len(result.boxes), flush=True)
    thresh = {i: (0.5, 0.5) for i in range(1, 11)}
    saved = OUT / "thresh.json"
    if saved.is_file():
        thresh = {int(k): tuple(v) for k, v in json.loads(saved.read_text()).items()}
    from collections import Counter

    for r in rows:
        if "_p" not in r:
            continue
        p0, p1 = r.pop("_p")
        t0, t1 = thresh[r["category_id"]]
        grade = _grade(p0, p1, t0, t1)
        masses = (1 - p0, p0 - p1, p1)
        r["freshness_grade"] = grade
        r["freshness_score"] = round(float(max(0.0, masses[grade])), 4)
    print(Counter(r["freshness_grade"] for r in rows), flush=True)
    SUB.parent.mkdir(parents=True, exist_ok=True)
    SUB.write_text(json.dumps(rows))
    print(f"{len(rows)} detections -> {SUB}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prep":
        prep()
    elif cmd == "train":
        train_det()
    elif cmd == "fresh":
        train_fresh()
    elif cmd == "fit":
        fit()
    elif cmd == "cv":
        cv()
    elif cmd == "predict":
        predict()
    else:
        raise SystemExit("usage: fresh.py prep|train|fresh|fit|cv|predict")


if __name__ == "__main__":
    main()
