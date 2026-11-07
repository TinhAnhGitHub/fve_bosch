#!/usr/bin/env python3
"""Food type and freshness.

Boxes and the 26 food types: YOLO11m (Jocher and Qiu, Ultralytics, 2024),
COCO-pretrained. 20.1M parameters, so it stays inside the 40M / T4 budget.

Freshness on the 10 Group A foods: CORAL (Cao, Mirjalili, Raschka, 2019,
arXiv 1901.07884). Three ordered grades, fresh / semi-fresh / spoiled, become
two shared-weight thresholds. A fresh-to-spoiled miss is a longer jump than
a fresh-to-semi-fresh miss, which is what quadratic weighted kappa penalises.

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        path, bbox, grade = self.samples[i][:3]
        im = Image.open(path).convert("RGB")
        return self.tf(_crop(im, bbox)), grade


def _crop(im, bbox, pad: float = 0.1):
    x, y, w, h = bbox
    box = (
        max(0, int(x - pad * w)),
        max(0, int(y - pad * h)),
        min(im.width, int(x + w * (1 + pad))),
        min(im.height, int(y + h * (1 + pad))),
    )
    return im.crop(box)


def train_fresh() -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import models

    meta = json.loads((OUT / "meta.json").read_text())
    hold = set(meta["holdout"])
    samples = group_a_crops()
    train_samples = [s for s in samples if s[4] not in hold]
    val_samples = [s for s in samples if s[4] in hold]
    if not train_samples:
        raise SystemExit("no Group A crops")

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    dim = net.classifier[0].in_features
    net.classifier = torch.nn.Identity()
    head = torch.nn.Linear(dim, 1, bias=False)
    bias = torch.nn.Parameter(torch.zeros(2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, head = net.to(device), head.to(device)
    bias = bias.to(device)
    opt = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": 1e-4},
            {"params": [head.weight, bias], "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    loader = DataLoader(_Crops(train_samples, True), batch_size=32, shuffle=True, num_workers=0)
    best, best_score = None, -1.0
    for epoch in range(1, 21):
        net.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logit = head(net(x)) + bias
            target = torch.stack([(y > 0).float(), (y > 1).float()], -1)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        score = _holdout_qwk(net, head, bias, val_samples, device)
        print(f"epoch {epoch} loss {total / len(train_samples):.4f} holdout_qwk {score:.4f}", flush=True)
        if score >= best_score:
            best_score = score
            best = {
                "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "head": head.weight.detach().cpu(),
                "bias": bias.detach().cpu(),
            }
    torch.save(best, OUT / "coral.pt")
    print(f"saved {OUT / 'coral.pt'} qwk {best_score:.4f}")


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
            x, _grade = loader[i]
            grade_hat = int((torch.sigmoid(head(net(x[None].to(device))) + bias) > 0.5).sum())
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


def _load_coral(device):
    import torch
    from torchvision import models

    blob = torch.load(OUT / "coral.pt", map_location=device, weights_only=True)
    net = models.mobilenet_v3_small(weights=None)
    net.classifier = torch.nn.Identity()
    net.load_state_dict(blob["net"])
    head = torch.nn.Linear(blob["head"].shape[1], 1, bias=False)
    head.weight.data.copy_(blob["head"])
    return net.to(device).eval(), head.to(device).eval(), blob["bias"].to(device)


def predict() -> None:
    import torch
    from PIL import Image
    from ultralytics import YOLO

    meta = json.loads((OUT / "meta.json").read_text())
    group_a = set(meta["group_a"])
    ids = meta["ids"]
    det = YOLO(str(next((OUT / "det").rglob("last.pt"))))
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
                    prob = torch.sigmoid(head(net(x[None].to(device))) + bias)[0]
                grade = int((prob > 0.5).sum())
                masses = (1 - prob[0], prob[0] - prob[1], prob[1])
                row["freshness_grade"] = grade
                row["freshness_score"] = round(float(masses[grade]), 4)
            else:
                row["freshness_grade"] = None
            rows.append(row)
        print(path.stem, len(result.boxes), flush=True)
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
    elif cmd == "predict":
        predict()
    else:
        raise SystemExit("usage: fresh.py prep|train|fresh|predict")


if __name__ == "__main__":
    main()
