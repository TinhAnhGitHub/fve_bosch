#!/usr/bin/env python3
"""Food type and freshness, v3: one RF-DETR-L (Apache-2.0, DINOv2 backbone, ~34M).

One model, one pass. The class head has 46 outputs: 16 Group B foods plus
10 Group A foods x 3 grades. A grade-split label survives every box
augmentation without touching RF-DETR internals. At inference the three
grade outputs of a query are folded back into one food detection:
  score = max grade prob, E = sum(g * p_g) / sum(p_g), then pack blend + cuts.

  python baselines/v3.py prep [--full]       # 80/20 photo holdout, or train on all
  python baselines/v3.py train [epochs] [res]
  python baselines/v3.py predict valid       # holdout: local score + tuned cuts
  python baselines/v3.py predict eval        # submission JSON
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v2  # noqa: E402  (qwk, macro, decode, blend, ann)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ftaf"
RF = DATA / os.environ.get("RFDIR", "rf")
FOOD_ONLY = os.environ.get("FOOD_ONLY") == "1"  # 26 classes; grades must come from CROP
FRESH = os.environ.get("FRESH") == "1"  # ordinal freshness head on decoder queries (fresh_head.py)
OUT = ROOT / "baselines" / "out" / os.environ.get("V3", "v3")
RES = 1024  # divisible by patch 16 x 2 windows; berries are small in 1920x1080


def classes(cats, ga):
    """46 names: Group A food_g0..2 then Group B food. Returns names, (food_id, grade) per index."""
    names, meaning = [], []
    for c in cats:
        if c["id"] in ga and not FOOD_ONLY:
            for g in range(3):
                names.append(f"{c['name']}_g{g}")
                meaning.append((c["id"], g))
        else:
            names.append(c["name"])
            meaning.append((c["id"], None))
    return names, meaning


def prep():
    a, cats, ga = v2.ann()
    names, meaning = classes(cats, ga)
    index = {m: i + 1 for i, m in enumerate(meaning)}  # COCO ids 1..46
    _fill_missing(a["annotations"], ga)
    full = "--full" in sys.argv
    ids = sorted(im["id"] for im in a["images"])
    random.Random(0).shuffle(ids)
    hold = set(ids[: len(ids) // 5])
    # every (food, grade) must exist in train, or RF-DETR drops the class and shifts labels
    by_img = defaultdict(list)
    for o in a["annotations"]:
        by_img[o["image_id"]].append(o)
    for m in meaning:
        if not any(index[_key(o, ga)] == index[m] for i in ids if i not in hold for o in by_img[i]):
            hold -= {next(i for i in hold if any(_key(o, ga) == m for o in by_img[i]))}
    split = {"train": [i for i in ids if full or i not in hold], "valid": sorted(hold), "test": sorted(hold)}
    ims = {im["id"]: im for im in a["images"]}
    for name, keep in split.items():
        d = RF / name
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for i in keep:
            im = ims[i]
            dst = d / im["file_name"]
            if not dst.exists():
                dst.symlink_to(Path(im["path"]).resolve())
            images.append({"id": i, "file_name": im["file_name"], "width": im["width"], "height": im["height"]})
            for o in by_img[i]:
                anns.append({"id": o["id"], "image_id": i, "category_id": index[_key(o, ga)], "bbox": o["bbox"],
                             "area": o["bbox"][2] * o["bbox"][3], "iscrowd": 0})
        cat_list = [{"id": k + 1, "name": n, "supercategory": "food"} for k, n in enumerate(names)]
        (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cat_list}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps({"full": full, "hold": sorted(hold)}))
    print({k: len(v) for k, v in split.items()}, "full" if full else "holdout")


_FILL = {}


def _key(o, ga):
    if o["category_id"] not in ga:
        return (o["category_id"], None)
    g = o["freshness_grade"]
    return (o["category_id"], int(g) if g is not None else _FILL[(o["image_id"], o["category_id"])])


def _fill_missing(anns, ga):
    """Ungraded Group A boxes (detector-only) take their pack's majority grade, else the food's."""
    from collections import Counter

    pack, food = defaultdict(Counter), defaultdict(Counter)
    for o in anns:
        if o["category_id"] in ga and o.get("freshness_grade") is not None:
            pack[(o["image_id"], o["category_id"])][o["freshness_grade"]] += 1
            food[o["category_id"]][o["freshness_grade"]] += 1
    for o in anns:
        if o["category_id"] in ga and o.get("freshness_grade") is None:
            key = (o["image_id"], o["category_id"])
            _FILL[key] = (pack.get(key) or food[o["category_id"]]).most_common(1)[0][0]


def train(epochs=80, res=RES):
    from rfdetr import RFDETRLarge

    if FRESH:
        import fresh_head

        fresh_head.patch()
        _a, cats, ga = v2.ann()
        fresh_head.GRADE = fresh_head.grade_table(RF, set(classes(cats, ga)[0]))
        print("fresh: grade table", fresh_head.GRADE.tolist())
    m = RFDETRLarge(resolution=int(res))
    m.train(dataset_dir=str(RF), epochs=int(epochs), batch_size=int(os.environ.get("BS", 4)),
            grad_accum_steps=int(os.environ.get("ACC", 4)), lr=1e-4, output_dir=str(OUT),
            num_workers=4, checkpoint_interval=10, progress_bar="tqdm", tensorboard=False)


def _weights():
    for n in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
        if (OUT / n).is_file():
            return OUT / n
    raise SystemExit(f"no weights in {OUT}")


GROUP_A = set(range(1, 11))


def fold(det, names, meaning, image_id):
    """sv.Detections of one image -> food detections with grade probs. Same query = identical box."""
    import numpy as np

    groups = {}
    cname = det.data.get("class_name") if hasattr(det, "data") else None
    for n, (box, s, k) in enumerate(zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist())):
        if cname is not None and str(cname[n]) not in names:
            continue  # "__background__" / no-object slot
        food, g = meaning[names.index(str(cname[n])) if cname is not None else k]
        key = (food, tuple(round(v, 2) for v in box))
        e = groups.setdefault(key, {"p": np.full(3, 1e-6), "score": 0.0, "grade": food in GROUP_A})
        e["score"] = max(e["score"], s)
        if g is not None:
            e["p"][g] = max(e["p"][g], s)
    out = []
    for (food, box), e in groups.items():
        x1, y1, x2, y2 = box
        d = {"image_id": image_id, "category_id": food, "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
             "score": round(float(e["score"]), 4)}
        if e["grade"]:
            d["_p"] = (e["p"] / e["p"].sum()).tolist()
        else:
            d["freshness_grade"] = None
        out.append(d)
    return out


def predict(split="valid"):
    import numpy as np
    from PIL import Image
    from rfdetr import RFDETRLarge

    a, cats, ga = v2.ann()
    names, meaning = classes(cats, ga)
    if FRESH:
        import fresh_head

        fresh_head.patch()
    m = RFDETRLarge(pretrain_weights=str(_weights()), resolution=int(os.environ.get("RES", RES)))
    if FRESH:
        _check_fresh_loaded(fresh_head)
    try:
        m.optimize_for_inference()
    except Exception as exc:  # optional speed-up
        print("no optimize:", type(exc).__name__)
    if split == "valid":
        hold = json.loads((OUT / "split.json").read_text())["hold"]
        ims = {im["id"]: im for im in a["images"]}
        items = [(i, ims[i]["path"]) for i in hold]
    else:
        import fresh

        items = [(int(p.stem), p) for p in sorted((fresh.find_dir(split) / "images").glob("*.jpg"), key=lambda p: int(p.stem))]
    crop = _crop_model() if os.environ.get("CROP") else None
    rows = []
    for n, (iid, path) in enumerate(items):
        pic = Image.open(path).convert("RGB")
        det = m.predict(pic, threshold=0.001, include_source_image=False)
        if n == 0:
            assert "class_name" in det.data, "rfdetr did not return class names; cannot map ids safely"
            print("sample:", list(zip(det.class_id[:3].tolist(), det.data["class_name"][:3])))
        new = fold(det, names, meaning, iid)
        if FRESH:
            _add_fresh(new, pic.size, n == 0)
        if crop is not None:
            _add_crop(new, pic, crop, ga)
        rows += new
    cfg_path = OUT / "fresh.json"
    if split == "valid":
        cfg = tune_and_score(rows, a, ga)
        cfg_path.write_text(json.dumps(cfg))
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {"mix": 0.5, "alpha": 1.0, "t1": 0.5, "t2": 1.5}
    if os.environ.get("PRIORS") == "1":
        cfg["priors"] = priors(a, ga)
    print("decode cfg:", {k: v for k, v in cfg.items() if k != "priors"}, "priors" in cfg)
    finish(rows, cfg)
    sub = ROOT / "submissions" / OUT.name / f"{split}{os.environ.get('TAG', '')}.json"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(json.dumps(rows))
    print(len(rows), "->", sub)


def _check_fresh_loaded(fh):
    import torch

    ck = torch.load(_weights(), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    keys = [k for k in sd if ".fresh." in k]
    assert keys, "checkpoint has no freshness head (trained without FRESH=1?)"
    net = fh.INST[-1].state_dict()
    k = keys[0]
    assert torch.allclose(net[k].float().cpu(), sd[k].float()), "freshness head weights not loaded"
    print("fresh: head loaded,", len(keys), "tensors")


def _add_fresh(rows, size, show=False):
    """Attach ordinal-head grade probs (as _pc) to each graded row via its query box."""
    import numpy as np
    import fresh_head as fh

    b = fh.LAST["boxes"][0].float().cpu().numpy()
    f = fh.ordinal_probs(fh.LAST["fresh"][0].float().cpu()).numpy()
    W, H = size
    q = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H, (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
    graded = [r for r in rows if "_p" in r]
    if not graded:
        return
    r = np.array([[x, y, x + w, y + h] for x, y, w, h in (g["bbox"] for g in graded)])
    lt, rb = np.maximum(r[:, None, :2], q[None, :, :2]), np.minimum(r[:, None, 2:], q[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    iou = inter / (area(r)[:, None] + area(q)[None] - inter + 1e-9)
    best = iou.argmax(1)
    if show:
        print("fresh: row-to-query IoU min/median", round(float(iou.max(1).min()), 3), round(float(np.median(iou.max(1))), 3))
    for g, k in zip(graded, best):
        g["_pc"] = f[k].tolist()


def _crop_model():
    import torch

    d = Path(os.environ["CROP"])
    blob = torch.load(d / "fresh.pt", map_location="cpu", weights_only=False)
    v2.SIZE = blob.get("size", 256)
    m, _ = v2.model(blob["arch"], pretrained=False)
    m.load_state_dict(blob["state"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(dev), blob["norm"], dev


def _add_crop(rows, pic, crop, ga):
    m, norm, dev = crop
    graded = [r for r in rows if "_p" in r]
    if not graded:
        return
    gidx = {c: i for i, c in enumerate(sorted(ga))}
    items = [(*v2._region(pic, r["bbox"]), 0, gidx[r["category_id"]], 0) for r in graded]
    for r, pc in zip(graded, v2.probs(m, items, norm, dev)):
        r["_pc"] = pc.tolist()


def finish(rows, cfg):
    """Mix head/crop grade probs, pack blend within (image, food) over confident boxes, ordinal cuts. Mutates rows."""
    import numpy as np

    graded = [r for r in rows if "_p" in r]
    if not graded:
        return
    mix = cfg.get("mix", 0.0)
    p = np.array([(1 - mix) * np.array(r.pop("_p")) + mix * np.array(r.pop("_pc", r.get("_p", [0, 0, 0]))) for r in graded])
    w = np.array([r["score"] if r["score"] >= 0.25 else 0.0 for r in graded])
    p = v2.blend(p, [(r["image_id"], r["category_id"]) for r in graded], cfg["alpha"], w)
    grades = v2.decode(p, cfg["t1"], cfg["t2"])
    if cfg.get("priors"):
        grades = prior_match(p, [r["category_id"] for r in graded], w > 0, cfg["priors"])
    for r, g, pp in zip(graded, grades, p):
        r["freshness_grade"] = int(g)
        r["freshness_score"] = round(float(pp[g]), 4)


def priors(a, ga):
    from collections import Counter

    c = {f: Counter() for f in ga}
    for o in a["annotations"]:
        if o["category_id"] in ga and o.get("freshness_grade") is not None:
            c[o["category_id"]][o["freshness_grade"]] += 1
    return {f: [c[f][g] / sum(c[f].values()) for g in range(3)] for f in ga}


def prior_match(p, foods, confident, pri):
    """Per food, cut expected grade E at the quantiles that give the training grade mix on confident boxes.
    Uses no labels of the scored set; robust to a calibration shift that squeezes E toward the middle."""
    import numpy as np

    e = p[:, 1] + 2 * p[:, 2]
    foods = np.asarray(foods)
    out = np.ones(len(p), int)
    for f, (p0, _p1, p2) in pri.items():
        m = foods == f
        ref = e[m & confident] if (m & confident).sum() >= 5 else e[m]
        if len(ref) == 0:
            continue
        t1, t2 = np.quantile(ref, p0), np.quantile(ref, 1 - p2)
        out[m] = (e[m] > t1).astype(int) + (e[m] > t2).astype(int)
    return out


def score(rows, a, ga, image_ids):
    """Board metric: 0.6 * macro mAP50-95 (26 foods) + 0.4 * macro QWK on dets matched at IoU>=0.5."""
    import contextlib
    import io

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    keep = set(image_ids)
    gt = {"images": [im for im in a["images"] if im["id"] in keep],
          "annotations": [dict(o, iscrowd=0, area=o["bbox"][2] * o["bbox"][3]) for o in a["annotations"] if o["image_id"] in keep],
          "categories": a["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes([{k: r[k] for k in ("image_id", "category_id", "bbox", "score")} for r in rows])
        e = COCOeval(cg, cd, "bbox")
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        e.summarize()
    mAP = float(e.stats[0])
    q = qwk_matched(rows, gt["annotations"], ga)
    return 0.6 * mAP + 0.4 * max(0.0, q), mAP, q


def qwk_matched(rows, gt_anns, ga):
    import numpy as np

    t, p, k = [], [], []
    dets = defaultdict(list)
    for r in rows:
        dets[(r["image_id"], r["category_id"])].append(r)
    gts = defaultdict(list)
    for o in gt_anns:
        gts[(o["image_id"], o["category_id"])].append(o)
    for key, gl in gts.items():
        if key[1] not in ga:
            continue
        used = set()
        for d in sorted(dets.get(key, []), key=lambda r: -r["score"]):
            best, bu = None, 0.5
            for j, o in enumerate(gl):
                u = _iou(d["bbox"], o["bbox"])
                if j not in used and u >= bu:
                    best, bu = j, u
            if best is None:
                continue
            used.add(best)
            if gl[best].get("freshness_grade") is not None:
                t.append(gl[best]["freshness_grade"]); p.append(d["freshness_grade"]); k.append(key[1])
    return v2.macro(np.array(k), np.array(t), np.array(p)) if t else 0.0


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def tune_and_score(rows, a, ga):
    import copy

    import numpy as np

    hold = sorted({r["image_id"] for r in rows})
    base = {"alpha": 1.0, "t1": 0.5, "t2": 1.5}
    r0 = copy.deepcopy(rows)
    finish(r0, base)
    print("holdout untuned: score %.4f mAP %.4f QWK %.4f" % score(r0, a, ga, hold))
    best = (-1, base)
    mixes = (0.0, 0.5, 1.0) if any("_pc" in r for r in rows) else (0.0,)
    for mix in mixes:
        cfg0 = dict(base, mix=mix)
        rr = copy.deepcopy(rows)
        finish(rr, cfg0)
        print(f"  mix {mix} (0=RF-DETR head, 1=crop model) untuned: score %.4f mAP %.4f QWK %.4f" % score(rr, a, ga, hold))
    keep = set(hold)
    gt_anns = [o for o in a["annotations"] if o["image_id"] in keep]
    per_mix = {}
    for mix, alpha in [(x, y) for x in mixes for y in (1.0, 0.85, 0.7, 0.5)]:
        for t1 in np.arange(0.4, 1.01, 0.1):
            for t2 in np.arange(1.1, 1.71, 0.1):
                cfg = {"mix": mix, "alpha": alpha, "t1": round(float(t1), 2), "t2": round(float(t2), 2)}
                rr = copy.deepcopy(rows)
                finish(rr, cfg)
                q = qwk_matched(rr, gt_anns, ga)
                per_mix[mix] = max(per_mix.get(mix, -1), q)
                if q > best[0]:
                    best = (q, cfg)
    print("  tuned QWK per mix:", {k: round(v, 4) for k, v in per_mix.items()})
    pri = priors(a, ga)
    for mix in mixes:
        rr = copy.deepcopy(rows)
        finish(rr, dict(base, mix=mix, priors=pri))
        print(f"  mix {mix} prior-matched cuts: QWK %.4f" % qwk_matched(rr, gt_anns, ga))
    print("holdout tuned (optimistic): QWK %.4f" % best[0], best[1])
    return best[1]


def _selftest():
    import numpy as np

    a = {"categories": [{"id": i, "name": f"f{i}", "supercategory": "group_a" if i <= 10 else "group_b"} for i in range(1, 27)]}
    cats = a["categories"]
    names, meaning = classes(cats, set(range(1, 11)))
    assert len(names) == 46 and meaning[0] == (1, 0) and meaning[2] == (1, 2) and meaning[30] == (11, None)

    class D:
        xyxy = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [5, 5, 9, 9]], float)
        confidence = np.array([0.8, 0.2, 0.6])
        class_id = np.array([2, 0, 30])

    D.data = {"class_name": np.array([names[2], names[0], names[30]])}
    out = fold(D, names, meaning, 7)
    by = {d["category_id"]: d for d in out}
    assert by[1]["score"] == 0.8 and np.allclose(by[1]["_p"], [0.2, 0, 0.8], atol=1e-4) and by[11]["freshness_grade"] is None
    print("ok")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["test"]
    args = [x for x in args if not x.startswith("--")]
    {"prep": prep, "train": train, "predict": predict, "test": _selftest}[cmd](*args)
