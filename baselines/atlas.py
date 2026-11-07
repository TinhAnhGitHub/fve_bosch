# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "trimesh"]
# ///
"""Category-template baseline for Food 3D Reconstruction.

A frozen DINOv2 ViT-B/14 (Oquab et al., 2023) reads the image. Each category
has one learned millimetre template (initialized from the median-sized
training cloud). The image adds a low-rank offset and a scale. Targets are
the canonical clouds from geom.py, so the loss sees shape and size and not
the scan pose. The template is one cloud per category, not a mesh retrieved
by appearance.

  uv run baselines/atlas.py cache
  uv run baselines/atlas.py train
  uv run baselines/atlas.py predict

cache and train use whatever GPU is present. ViT-B/14 at 224 plus this
decoder fits an L4. CPU still runs, only slower. An A100 is not required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import MIN_POINTS, canonicalize, pack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "baselines" / "out" / "atlas"
CKPT = OUT / "atlas.pt"
IMAGE = 224
N = MIN_POINTS
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
        chunks.append(net(x).float().cpu())
        print(f"embed {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(chunks).numpy()


class Atlas(nn.Module):
    """One template per category, plus an image offset and a scale.

    The folded 768-d decoder scored 733 and then memorised the training
    clouds. A 64-d bottleneck scored 864. The template holds the category
    shape, so the image only has to move it.
    """

    def __init__(self, dim: int, n_cat: int, n: int = N, k: int = 32) -> None:
        super().__init__()
        self.tmpl = nn.Parameter(torch.zeros(n_cat, n, 3))
        self.basis = nn.Parameter(torch.randn(k, n, 3) * 0.01)
        self.coeff = nn.Linear(dim, k)
        self.scale = nn.Linear(dim, 1)
        nn.init.zeros_(self.coeff.weight)
        nn.init.zeros_(self.coeff.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, feat: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        delta = torch.einsum("bk,knd->bnd", self.coeff(feat), self.basis)
        # ponytail: tanh caps scale at about 7x. Wider tails need a bigger multiplier.
        size = torch.exp(self.scale(feat).tanh() * 2.0)
        return (self.tmpl[cat] + delta) * size[:, None, :]


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


def _median_templates(pts: torch.Tensor, cat: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One cloud per category: the sample whose size is closest to the median."""
    tmpl = torch.zeros(n_cat, pts.shape[1], pts.shape[2])
    rms = pts.square().mean(dim=(1, 2)).sqrt()
    for c in range(n_cat):
        idx = (cat == c).nonzero().flatten()
        if len(idx) == 0:
            continue
        r = rms[idx]
        tmpl[c] = pts[idx[(r - r.median()).abs().argmin()]]
    return tmpl


def train() -> None:
    z, vocab = _load_cache()
    dev = device()
    feats = torch.tensor(z["feats"], dtype=torch.float32)
    pts = torch.tensor(z["pts"], dtype=torch.float32)
    cat = torch.tensor([vocab[c] for c in z["cats"].tolist()], dtype=torch.long)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(feats))
    cut = max(1, int(len(order) * 0.1))
    val, tr = order[:cut], order[cut:]
    net = Atlas(feats.shape[1], len(vocab)).to(dev)
    with torch.no_grad():
        net.tmpl.copy_(_median_templates(pts, cat, len(vocab)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 4 if dev.type == "cuda" else 2
    best, best_score = None, float("inf")
    for epoch in range(1, 61):
        net.train()
        rng.shuffle(tr)
        total = 0.0
        seen = 0
        for i in range(0, len(tr), batch):
            b = tr[i : i + batch]
            pred = net(feats[b].to(dev), cat[b].to(dev))
            gt = pts[b].to(dev)
            loss = cd_l2(pred, gt) + (pred.square().mean((1, 2)).sqrt() - gt.square().mean((1, 2)).sqrt()).square().mean()
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
                vtotal += float(cd_l2(net(feats[b].to(dev), cat[b].to(dev)), pts[b].to(dev), k=8192)) * len(b)
                vseen += len(b)
        score = vtotal / vseen
        if score < best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch}  train {total / seen:.1f}  holdout {score:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best, "vocab": vocab, "dim": feats.shape[1]}, CKPT)
    print(f"saved {CKPT}  holdout {best_score:.1f}")


def predict() -> None:
    if not CKPT.exists():
        raise SystemExit(f"missing {CKPT}. Run: uv run baselines/atlas.py train")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    dev = device()
    net = Atlas(blob["dim"], len(blob["vocab"])).to(dev).eval()
    net.load_state_dict(blob["state"])
    enc = encoder(dev)
    paths = rows("val")
    if not paths:
        raise SystemExit("no eval images")
    vocab = blob["vocab"]
    unk = 0
    clouds = {}
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            chunk = paths[i : i + 32]
            feat = enc(torch.stack([image_tensor(p) for p in chunk]).to(dev))
            cat = torch.tensor(
                [vocab.get(p.parent.parent.name, unk) for p in chunk],
                dtype=torch.long,
                device=dev,
            )
            pred = net(feat, cat).cpu().numpy()
            for p, cloud in zip(chunk, pred):
                clouds[p.parent.name] = cloud
            print(f"predict {min(i + 32, len(paths))}/{len(paths)}", flush=True)
    out = ROOT / "submissions" / "atlas" / "preds_geometry.zip"
    pack(out, clouds)
    print(f"{len(clouds)} clouds -> {out}")


def fit_check() -> None:
    torch.manual_seed(0)
    net = Atlas(8, 1, n=128, k=4)
    with torch.no_grad():
        net.tmpl[0, :, 0] = torch.linspace(0, 10, 128)
    feat = torch.zeros(1, 8)
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
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "cache":
        cache()
    elif cmd == "train":
        train()
    elif cmd == "predict":
        predict()
    elif cmd == "check":
        fit_check()
    else:
        raise SystemExit("use cache, train, predict, or check")
