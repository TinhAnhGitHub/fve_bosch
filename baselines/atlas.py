# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "trimesh"]
# ///
"""AtlasNet baseline for Food 3D Reconstruction.

Groueix et al., CVPR 2018: a decoder folds a 2D grid onto the surface.
A frozen DINOv2 ViT-B/14 (Oquab et al., 2023) reads the image. Each output
point attends to the patch tokens, not only the class token. Targets are the
canonical clouds from geom.py, divided by their RMS radius: the decoder
draws a unit-size shape and scale.py says how many millimetres that is.
The category vector is learned. It is not a retrieved training mesh.

  uv run baselines/atlas.py cache
  uv run baselines/atlas.py train
  uv run baselines/scale.py
  uv run baselines/atlas.py evaluate     # local grader on the 10% holdout
  uv run baselines/atlas.py predict

cache and train use whatever GPU is present. ViT-B/14 at 224 plus this
decoder fits an L4. CPU still runs, only slower. An A100 is not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import MIN_POINTS, canonicalize, hedge, pack

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "baselines" / "out" / "atlas"
CKPT = OUT / "atlas.pt"
SCALE = ROOT / "baselines" / "out" / "scale" / "scale.npz"
HEDGE = ROOT / "baselines" / "out" / "scale" / "hedge.json"
IMAGE = 224
N = MIN_POINTS
PATCHES = 8
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    if not torch.cuda.is_available():
        print("no GPU, using CPU")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"gpu {name}  free {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    return torch.device("cuda")


def data_root() -> Path:
    for path in (ROOT / "data" / "f3d", ROOT / "data" / "fve"):
        if path.is_dir():
            return path
    raise SystemExit("put the challenge files in data/f3d or data/fve")


def rows(split: str) -> list[Path]:
    root = data_root() / split
    if not root.is_dir() and split == "val":
        root = data_root() / "eval"
    return sorted(root.rglob("rgb_industrial.png"))


def image_tensor(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((IMAGE, IMAGE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    return (x - mean) / std


def target_points(mesh_path: Path, n: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    except Exception:
        v = np.asarray(mesh.vertices, np.float64)
        rng = np.random.default_rng(abs(hash(mesh_path.name)) % (2**32))
        take = rng.choice(len(v), n, replace=len(v) < n)
        pts = v[take]
    return canonicalize(pts).astype(np.float32)


def encoder(dev: torch.device) -> nn.Module:
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    net.eval().to(dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def embed(net: nn.Module, paths: list[Path], dev: torch.device, batch: int = 32) -> np.ndarray:
    chunks = []
    for i in range(0, len(paths), batch):
        x = torch.stack([image_tensor(p) for p in paths[i : i + batch]]).to(dev)
        chunks.append(image_tokens(net, x).half().cpu())
        print(f"embed {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(chunks).numpy()


def image_tokens(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Class token plus patch tokens. The class token alone scored 733."""
    out = net.forward_features(x)
    return torch.cat([out["x_norm_clstoken"][:, None], out["x_norm_patchtokens"]], 1).float()


class Atlas(nn.Module):
    """Fold PATCHES grids into N millimetre points, attending to image tokens."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES) -> None:
        super().__init__()
        g = int((n // patches) ** 0.5)
        if g * g * patches != n:
            raise ValueError(f"{n} is not {patches} square grids")
        axis = torch.linspace(-1.0, 1.0, g)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], -1).reshape(-1, 2))
        self.patch = nn.Parameter(torch.randn(patches, 16) * 0.02)
        self.cat = nn.Embedding(n_cat, 32)
        width = 256
        self.q = nn.Linear(32 + 16 + 2, width)
        self.k = nn.Linear(dim, width)
        self.v = nn.Linear(dim, width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, tok: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        b, p, g = tok.shape[0], self.patch.shape[0], self.grid.shape[0]
        parts = [
            self.cat(cat)[:, None, None, :].expand(b, p, g, -1),
            self.patch[None, :, None, :].expand(b, p, g, -1),
            self.grid[None, None].expand(b, p, g, -1),
        ]
        q = self.q(torch.cat(parts, -1)).reshape(b, p * g, -1)
        key = self.k(tok)
        attn = torch.softmax(q @ key.transpose(1, 2) * (key.shape[-1] ** -0.5), dim=-1)
        return self.mlp(attn @ self.v(tok))


def cd_l2(pred: torch.Tensor, gt: torch.Tensor, k: int = 2048) -> torch.Tensor:
    if pred.shape[1] > k:
        ip = torch.randint(0, pred.shape[1], (k,), device=pred.device)
        ig = torch.randint(0, gt.shape[1], (k,), device=gt.device)
        pred, gt = pred[:, ip], gt[:, ig]
    d2 = torch.cdist(pred, gt).square()
    return d2.min(2).values.mean() + d2.min(1).values.mean()


def _mesh_row(img: str) -> tuple[str, str, np.ndarray]:
    path = Path(img)
    return path.parent.name, path.parent.parent.name, target_points(path.parent / "mesh.ply", N)


def cache() -> None:
    from concurrent.futures import ProcessPoolExecutor

    paths = rows("train")
    if not paths:
        raise SystemExit(f"no training images under {data_root()}")
    dev = device()
    saved = OUT / "train.npz"
    if saved.exists():
        z = np.load(saved, allow_pickle=False)
        if z["ids"].tolist() == [p.parent.name for p in paths]:
            ids, cats, vocab, pts = (np.array(z["ids"]), np.array(z["cats"]), np.array(z["vocab"]), np.array(z["pts"]))
            z.close()
            print(f"reuse {len(paths)} clouds", flush=True)
            feats = embed(encoder(dev), paths, dev)
            np.savez(saved, ids=ids, cats=cats, vocab=vocab, feats=feats, pts=pts)
            print(f"cached {len(paths)}  tokens {feats.shape[1]}  -> {saved}")
            return
        z.close()
    ids, cats, pts = [], [], []
    with ProcessPoolExecutor() as pool:
        for i, (sample_id, cat, cloud) in enumerate(pool.map(_mesh_row, map(str, paths), chunksize=16), 1):
            ids.append(sample_id)
            cats.append(cat)
            pts.append(cloud)
            if i % 50 == 0 or i == len(paths):
                print(f"mesh {i}/{len(paths)}", flush=True)
    vocab = sorted(set(cats))
    feats = embed(encoder(dev), paths, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "train.npz",
        ids=np.array(ids),
        cats=np.array(cats),
        vocab=np.array(vocab),
        feats=feats,
        pts=np.stack(pts),
    )
    print(f"cached {len(ids)}  dim {feats.shape[1]}  -> {OUT / 'train.npz'}")


def _load_cache():
    path = OUT / "train.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run: uv run baselines/atlas.py cache")
    z = np.load(path, allow_pickle=False)
    vocab = {c: i for i, c in enumerate(z["vocab"].tolist())}
    return z, vocab


def holdout(n: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """The fixed 10% holdout (seed 0) shared by train and evaluate."""
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    cut = max(1, int(n * 0.1))
    return order[:cut], order[cut:], rng


def radii(pts: torch.Tensor) -> torch.Tensor:
    return pts.square().sum(-1).mean(-1).sqrt()


def train(ckpt: Path, epochs: int) -> None:
    z, vocab = _load_cache()
    dev = device()
    feats = torch.tensor(z["feats"], dtype=torch.float32)
    if feats.ndim != 3:
        raise SystemExit("cache has no patch tokens. Run: python baselines/atlas.py cache")
    pts = torch.tensor(z["pts"], dtype=torch.float32)
    r = radii(pts)
    unit = pts / r[:, None, None]
    cat = torch.tensor([vocab[c] for c in z["cats"].tolist()], dtype=torch.long)
    val, tr, rng = holdout(len(feats))
    net = Atlas(feats.shape[-1], len(vocab)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 4 if dev.type == "cuda" else 2
    best, best_score = None, float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(tr)
        total = 0.0
        seen = 0
        for i in range(0, len(tr), batch):
            b = tr[i : i + batch]
            pred = net(feats[b].to(dev), cat[b].to(dev))
            gt = unit[b].to(dev)
            loss = cd_l2(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            seen += len(b)
        net.eval()
        vtotal, vseen = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val), batch):
                b = val[i : i + batch]
                pred = net(feats[b].to(dev), cat[b].to(dev))
                # shape error in mm^2 at the true size: CD scales with r^2
                for j, k in enumerate(b):
                    vtotal += float(cd_l2(pred[j : j + 1], unit[k : k + 1].to(dev), k=8192)) * float(r[k]) ** 2
                vseen += len(b)
        score = vtotal / vseen
        if score < best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch}  train {total / seen:.4f} unit^2  holdout {score:.1f} mm^2 at true size", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best, "vocab": vocab, "dim": feats.shape[-1], "unit": True}, ckpt)
    print(f"saved {ckpt}  holdout {best_score:.1f}")


def load_scale() -> dict[str, float]:
    if not SCALE.exists():
        raise SystemExit(f"missing {SCALE}. Run: uv run baselines/scale.py")
    z = np.load(SCALE, allow_pickle=False)
    return dict(zip(z["ids"].tolist(), z["s"].tolist()))


def load_sizes() -> tuple[dict[str, tuple[float, tuple[float, float]]], dict]:
    """Per id: P(oversized) and the expected radius of each cluster, plus the tuned hedge."""
    import json

    load_scale()
    z = np.load(SCALE, allow_pickle=False)
    sizes = np.exp(z["mu"] + z["sig"][None, :] ** 2 / 2)
    table = {i: (float(p), (float(a), float(b))) for i, p, (a, b) in zip(z["ids"].tolist(), z["p"], sizes)}
    rule = json.loads(HEDGE.read_text()) if HEDGE.exists() else {"q_min": 1.0, "frac": 0.0}
    return table, rule


def to_mm(unit: np.ndarray, sample_id: str, table: dict, rule: dict) -> np.ndarray:
    p, sizes = table[sample_id]
    return hedge(unit, p, sizes, rule["q_min"], rule["frac"])


def load_net(ckpt: Path, dev: torch.device) -> tuple[nn.Module, dict]:
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}. Run: uv run baselines/atlas.py train")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = Atlas(blob["dim"], len(blob["vocab"])).to(dev).eval()
    net.load_state_dict(blob["state"])
    return net, blob


def predict(ckpt: Path) -> None:
    dev = device()
    net, blob = load_net(ckpt, dev)
    table, rule = load_sizes() if blob.get("unit") else (None, None)
    if rule is not None:
        print(f"hedge q_min {rule['q_min']} frac {rule['frac']}")
    enc = encoder(dev)
    paths = rows("val")
    if not paths:
        raise SystemExit("no eval images")
    vocab = blob["vocab"]
    unk = 0
    clouds = {}
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            chunk = paths[i : i + 8]
            tok = image_tokens(enc, torch.stack([image_tensor(p) for p in chunk]).to(dev))
            cat = torch.tensor(
                [vocab.get(p.parent.parent.name, unk) for p in chunk],
                dtype=torch.long,
                device=dev,
            )
            pred = net(tok, cat).cpu().numpy()
            for p, cloud in zip(chunk, pred):
                clouds[p.parent.name] = to_mm(cloud, p.parent.name, table, rule) if table is not None else cloud
            print(f"predict {min(i + len(chunk), len(paths))}/{len(paths)}", flush=True)
    out = ROOT / "submissions" / "atlas" / "preds_geometry.zip"
    pack(out, clouds)
    print(f"{len(clouds)} clouds -> {out}")


FLIPS = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32)


def nearest(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance and index of the nearest b for every a. a (n, 3), b (m, 3)."""
    d, j = [], []
    for s in range(0, len(a), chunk):
        v, k = torch.cdist(a[s : s + chunk], b).square().min(1)
        d.append(v)
        j.append(k)
    return torch.cat(d), torch.cat(j)


def kabsch(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation and translation that move a onto b, no scale."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = torch.linalg.svd((a - ca).T @ (b - cb))
    fix = torch.ones(3, device=a.device)
    fix[2] = torch.sign(torch.det(vt.T @ u.T))
    rot = vt.T @ torch.diag(fix) @ u.T
    return rot, cb - rot @ ca


@torch.no_grad()
def aligned_cd(pred: torch.Tensor, gt: torch.Tensor, iters: int = 10) -> float:
    """Local copy of the grader: best rotation and translation, scale kept. Unit mm^2.

    Starts from the 4 proper sign flips of the PCA frame, then point-to-point ICP.
    """
    best = float("inf")
    for flip in FLIPS.to(pred.device):
        q = pred * flip
        q = q - q.mean(0) + gt.mean(0)
        for _ in range(iters):
            _, j = nearest(q, gt)
            rot, t = kabsch(q, gt[j])
            q = q @ rot.T + t
        best = min(best, float(nearest(q, gt)[0].mean() + nearest(gt, q)[0].mean()))
    return best


def evaluate(ckpts: list[Path]) -> None:
    """Score holdout predictions the way the grader does, with the predicted and the true size."""
    z, vocab = _load_cache()
    dev = device()
    val, _, _ = holdout(len(z["ids"]))
    ids = z["ids"][val].tolist()
    feats = torch.tensor(z["feats"][val], dtype=torch.float32)
    gt = torch.tensor(z["pts"][val], dtype=torch.float32)
    r = radii(gt)
    cat = torch.tensor([vocab[c] for c in z["cats"][val].tolist()], dtype=torch.long)
    size = torch.tensor([load_scale()[i] for i in ids]) if SCALE.exists() else None
    big = torch.zeros(len(ids), dtype=torch.bool)
    if SCALE.exists():
        zs = np.load(SCALE, allow_pickle=False)
        u = dict(zip(zs["ids"].tolist(), zs["u"].tolist()))
        big = torch.tensor([u[i] == 1 for i in ids])
        err = (size - r).square()
        print(f"size head on holdout: mean (s - s_true)^2 {err.mean():.1f} mm^2, oversized {int(big.sum())}")
    for ckpt in ckpts:
        net, blob = load_net(ckpt, dev)
        with torch.inference_mode():
            pred = torch.cat([net(feats[i : i + 8].to(dev), cat[i : i + 8].to(dev)).cpu() for i in range(0, len(ids), 8)])
        own = radii(pred - pred.mean(1, keepdim=True))
        variants = {}
        if blob.get("unit"):
            variants["true size"] = pred * r[:, None, None]
            if size is not None:
                variants["size head"] = pred * size[:, None, None]
                table, rule = load_sizes()
                if rule["frac"] > 0:
                    variants["size head + hedge"] = [
                        torch.from_numpy(to_mm(pred[k].numpy(), i, table, rule)).float() for k, i in enumerate(ids)
                    ]
        else:
            variants["as trained, mm"] = pred
            variants["rescaled to true size"] = pred * (r / own)[:, None, None]
            if size is not None:
                variants["rescaled to size head"] = pred * (size / own)[:, None, None]
        for name, clouds in variants.items():
            cd = torch.tensor([aligned_cd(clouds[k].to(dev), gt[k].to(dev)) for k in range(len(ids))])
            line = f"{ckpt.name:18s} {name:24s} CD-L2 {cd.mean():8.1f}  median {cd.median():7.1f}"
            if big.any():
                line += f"  normal {cd[~big].mean():8.1f}  oversized {cd[big].mean():9.1f}"
            print(line, flush=True)


def fit_check() -> None:
    torch.manual_seed(0)
    net = Atlas(8, 1, n=128, patches=2)
    feat = torch.zeros(1, 4, 8)
    cat = torch.zeros(1, dtype=torch.long)
    target = torch.zeros(1, 128, 3)
    target[0, :, 0] = torch.linspace(0, 40, 128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = float(cd_l2(net(feat, cat), target, k=128).detach())
    for _ in range(40):
        loss = cd_l2(net(feat, cat), target, k=128)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = float(cd_l2(net(feat, cat), target, k=128).detach())
    assert last < first * 0.5, (first, last)
    print(f"fit check ok  {first:.1f} -> {last:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["cache", "train", "predict", "evaluate", "check"])
    ap.add_argument("--ckpt", type=Path, nargs="+", default=[CKPT])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if args.cmd == "cache":
        cache()
    elif args.cmd == "train":
        train(args.ckpt[0], args.epochs)
    elif args.cmd == "predict":
        predict(args.ckpt[0])
    elif args.cmd == "evaluate":
        evaluate(args.ckpt)
    else:
        fit_check()
