# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
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

--encoder dinov3l on cache and train swaps DINOv2-B/14 at 224 for DINOv3-L/16 at
256 (Simeoni et al., 2025; 256 patches + class token, registers dropped). Its tokens
go to out/atlas/train_dinov3l.npz; the checkpoint records the encoder.

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
ENCODERS = {"dinov2b": 224, "dinov3l": 256}  # name -> input size; both give 257 tokens
DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
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


def image_tensor(path: Path, size: int = IMAGE) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
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


class DinoV3(nn.Module):
    """Hugging Face DINOv3 returning [class token, patch tokens], registers dropped."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(DINOV3)
        self.registers = self.net.config.num_register_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x).last_hidden_state
        return torch.cat([out[:, :1], out[:, 1 + self.registers :]], 1)


def encoder(dev: torch.device, name: str = "dinov2b") -> nn.Module:
    net = DinoV3() if name == "dinov3l" else torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    net.eval().to(dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def embed(net: nn.Module, paths: list[Path], dev: torch.device, batch: int = 32, size: int = IMAGE) -> np.ndarray:
    chunks = []
    for i in range(0, len(paths), batch):
        x = torch.stack([image_tensor(p, size) for p in paths[i : i + batch]]).to(dev)
        chunks.append(image_tokens(net, x).half().cpu())
        print(f"embed {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(chunks).numpy()


def image_tokens(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Class token plus patch tokens. The class token alone scored 733."""
    if isinstance(net, DinoV3):
        return net(x).float()
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


class Block(nn.Module):
    """Pre-norm cross-attention from points to image tokens, then an MLP, each with a residual."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        h = self.heads
        q = self.q(self.norm_q(x)).reshape(b, n, h, w // h).transpose(1, 2)
        k, v = self.kv(tok).reshape(b, tok.shape[1], 2, h, w // h).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, w))
        return x + self.mlp(self.norm_mlp(x))


class DeepAtlas(Atlas):
    """Same atlas queries, but depth blocks of multi-head cross-attention with residuals.

    The original decoder is one single-head attention and a 2-layer MLP, with no
    residual or normalisation: each point gets one look at the image. Here every block
    lets a point re-read the tokens with what it has already decided, the usual
    transformer-decoder layout (as in Point-E / LRM-style point and triplane decoders),
    without point-to-point self-attention, which 8192 queries would make costly.
    With no self-attention each point depends only on its own query, so training can
    draw a random subset of the queries (`keep`) and still match all target points.
    """

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8) -> None:
        super().__init__(dim, n_cat, n, patches)
        width = self.q.out_features
        del self.k, self.v, self.mlp
        self.norm_tok = nn.LayerNorm(dim)
        self.tok = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, tok: torch.Tensor, cat: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        b, p, g = tok.shape[0], self.patch.shape[0], self.grid.shape[0]
        parts = [
            self.cat(cat)[:, None, None, :].expand(b, p, g, -1),
            self.patch[None, :, None, :].expand(b, p, g, -1),
            self.grid[None, None].expand(b, p, g, -1),
        ]
        x = torch.cat(parts, -1).reshape(b, p * g, -1)
        if keep is not None:
            x = x[:, keep]
        with torch.autocast(tok.device.type, dtype=torch.bfloat16, enabled=tok.is_cuda):
            x = self.q(x)
            t = self.tok(self.norm_tok(tok))
            for block in self.blocks:
                x = block(x, t)
            return self.out(x).float()


ARCHS = {"atlas": Atlas, "deep": DeepAtlas}


def cd_l2(pred: torch.Tensor, gt: torch.Tensor, k: int = 2048) -> torch.Tensor:
    if pred.shape[1] > k:
        ip = torch.randint(0, pred.shape[1], (k,), device=pred.device)
        ig = torch.randint(0, gt.shape[1], (k,), device=gt.device)
        pred, gt = pred[:, ip], gt[:, ig]
    d2 = torch.cdist(pred, gt).square()
    return d2.min(2).values.mean() + d2.min(1).values.mean()


FLIPS = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32)


def nearest(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance and index of the nearest b for every a. a (n, 3), b (m, 3)."""
    d, j = [], []
    for s in range(0, len(a), chunk):
        v, k = torch.cdist(a[s : s + chunk], b).square().min(1)
        d.append(v)
        j.append(k)
    return torch.cat(d), torch.cat(j)


def chamfer(pred: torch.Tensor, gt: torch.Tensor, flips: bool = True, chunk: int = 2048, pick: int = 1024) -> torch.Tensor:
    """Training loss: exact CD-L2 over every point, at the best PCA sign flip of each target.

    PCA fixes the axes but not their signs: geom.canonicalize picks them by skewness, and
    a near-symmetric axis flips between similar objects. The grader undoes any rotation,
    so the loss should not punish a flip either. It takes the minimum over the 4 proper
    flips, chosen on a 1024-point subsample. Exact nearest neighbours over all 8192
    points then come from one chunked pass without gradient, and the loss is the plain
    squared distance to them.
    """
    cand = FLIPS.to(gt.device) if flips else FLIPS[:1].to(gt.device)
    nb, n, m = len(pred), pred.shape[1], gt.shape[1]
    with torch.no_grad():
        q = pred.detach()
        if len(cand) > 1:  # pick the flip on a subsample, all flips in one batched pass
            ip = torch.randperm(n, device=gt.device)[:pick]
            ig = torch.randperm(m, device=gt.device)[:pick]
            g = gt[:, None, ig] * cand[None, :, None, :]
            d = torch.cdist(q[:, None, ip].expand(-1, len(cand), -1, -1), g).square()
            best = (d.min(3).values.mean(2) + d.min(2).values.mean(2)).argmin(1)
        else:
            best = torch.zeros(nb, dtype=torch.long, device=gt.device)
        gf = gt * cand[best][:, None, :]
        j1 = torch.empty((nb, n), dtype=torch.long, device=gt.device)
        d2 = torch.full((nb, m), float("inf"), device=gt.device)
        j2 = torch.zeros((nb, m), dtype=torch.long, device=gt.device)
        for s in range(0, n, chunk):  # exact neighbours over every point, both directions
            d = torch.cdist(q[:, s : s + chunk], gf).square()
            j1[:, s : s + chunk] = d.argmin(2)
            v, k = d.min(1)
            better = v < d2
            d2 = torch.where(better, v, d2)
            j2 = torch.where(better, k + s, j2)
    to_gt = (pred - torch.gather(gf, 1, j1[..., None].expand(-1, -1, 3))).square().sum(-1).mean(1)
    to_pred = (gf - torch.gather(pred, 1, j2[..., None].expand(-1, -1, 3))).square().sum(-1).mean(1)
    return (to_gt + to_pred).mean()


def _mesh_row(img: str) -> tuple[str, str, np.ndarray]:
    path = Path(img)
    return path.parent.name, path.parent.parent.name, target_points(path.parent / "mesh.ply", N)


def cache_path(enc: str) -> Path:
    return OUT / ("train.npz" if enc == "dinov2b" else f"train_{enc}.npz")


def cache(enc: str = "dinov2b") -> None:
    from concurrent.futures import ProcessPoolExecutor

    paths = rows("train")
    if not paths:
        raise SystemExit(f"no training images under {data_root()}")
    dev = device()
    saved = OUT / "train.npz"
    if enc != "dinov2b":
        if not saved.exists():
            raise SystemExit(f"run: uv run baselines/atlas.py cache   (the {enc} cache reuses its clouds)")
        ids = np.load(saved, allow_pickle=False)["ids"]
        if ids.tolist() != [p.parent.name for p in paths]:
            raise SystemExit("train.npz is stale. Run: uv run baselines/atlas.py cache")
        feats = embed(encoder(dev, enc), paths, dev, size=ENCODERS[enc])
        np.savez(cache_path(enc), ids=ids, feats=feats)
        print(f"cached {len(ids)}  tokens {feats.shape[1:]}  -> {cache_path(enc)}")
        return
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


def _load_cache(enc: str = "dinov2b") -> tuple[dict, dict[str, int]]:
    """Clouds, categories and ids from train.npz; image tokens from the encoder's cache."""
    for path in {OUT / "train.npz", cache_path(enc)}:
        if not path.exists():
            raise SystemExit(f"missing {path}. Run: uv run baselines/atlas.py cache --encoder {enc}")
    with np.load(OUT / "train.npz", allow_pickle=False) as base:
        z = {k: base[k] for k in base.files if enc == "dinov2b" or k != "feats"}
    if enc != "dinov2b":
        with np.load(cache_path(enc), allow_pickle=False) as extra:
            if extra["ids"].tolist() != z["ids"].tolist():
                raise SystemExit(f"{cache_path(enc)} is stale. Run: uv run baselines/atlas.py cache --encoder {enc}")
            z["feats"] = extra["feats"]
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


def train(ckpt: Path, epochs: int, arch: str, enc: str) -> None:
    z, vocab = _load_cache(enc)
    dev = device()
    feats = torch.tensor(z["feats"], dtype=torch.float32)
    if feats.ndim != 3:
        raise SystemExit("cache has no patch tokens. Run: python baselines/atlas.py cache")
    pts = torch.tensor(z["pts"], dtype=torch.float32)
    r = radii(pts)
    unit = pts / r[:, None, None]
    cat = torch.tensor([vocab[c] for c in z["cats"].tolist()], dtype=torch.long)
    val, tr, rng = holdout(len(feats))
    net = ARCHS[arch](feats.shape[-1], len(vocab)).to(dev)
    print(f"{arch}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")
    lr = 1e-3 if arch == "atlas" else 3e-4
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * -(-len(tr) // (4 if dev.type == "cuda" else 2))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05) if arch != "atlas" else None
    batch = 4 if dev.type == "cuda" else 2
    best, best_score = None, float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(tr)
        total = 0.0
        seen = 0
        for i in range(0, len(tr), batch):
            b = tr[i : i + batch]
            if arch == "atlas":
                pred = net(feats[b].to(dev), cat[b].to(dev))
            else:
                pred = net(feats[b].to(dev), cat[b].to(dev), keep=torch.randperm(N, device=dev)[: N // 2])
            gt = unit[b].to(dev)
            loss = chamfer(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
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
                    vtotal += float(chamfer(pred[j : j + 1], unit[k : k + 1].to(dev))) * float(r[k]) ** 2
                vseen += len(b)
        score = vtotal / vseen
        if score < best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch}  train {total / seen:.4f} unit^2  holdout {score:.1f} mm^2 at true size", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best, "vocab": vocab, "dim": feats.shape[-1], "unit": True, "arch": arch, "encoder": enc}, ckpt)
    print(f"saved {ckpt}  holdout {best_score:.1f}")


def descriptor(tok: torch.Tensor) -> torch.Tensor:
    """Retrieval key: class token and mean patch token, unit length. tok (b, 257, dim)."""
    v = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], -1).float()
    return torch.nn.functional.normalize(v, dim=-1)


def nearest_shape(query: torch.Tensor, q_cat: list[str], bank: torch.Tensor, bank_cat: list[str]) -> np.ndarray:
    """Index of the most similar bank image, inside the query's category when the bank has it.

    Tatarchenko et al. (CVPR 2019) show that this retrieval baseline matches or beats
    single-view decoders on ShapeNet, because those decoders mostly recognise and recall.
    """
    sim = query @ bank.T
    cats = np.array(bank_cat)
    for i, c in enumerate(q_cat):
        same = torch.from_numpy(cats == c)
        if same.any():
            sim[i, ~same] = -float("inf")
    return sim.argmax(1).numpy()


def bank_descriptors(feats: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    return torch.cat([descriptor(torch.from_numpy(feats[idx[i : i + 64]])) for i in range(0, len(idx), 64)])


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
    net = ARCHS[blob.get("arch", "atlas")](blob["dim"], len(blob["vocab"])).to(dev).eval()
    net.load_state_dict(blob["state"])
    return net, blob


def predict(ckpt: Path, shape: str) -> None:
    dev = device()
    if shape == "retrieve":
        z, vocab = _load_cache()
        everyone = np.arange(len(z["ids"]))
        bank, bank_cat = bank_descriptors(z["feats"], everyone), z["cats"].tolist()
        pts = z["pts"].astype(np.float32)
        units = pts / np.sqrt((pts**2).sum(-1).mean(-1))[:, None, None]
        net, blob = None, {"unit": True, "vocab": vocab}
    else:
        net, blob = load_net(ckpt, dev)
    table, rule = load_sizes() if blob.get("unit") else (None, None)
    if rule is not None:
        print(f"hedge q_min {rule['q_min']} frac {rule['frac']}")
    name = blob.get("encoder", "dinov2b")
    enc = encoder(dev, name)
    paths = rows("val")
    if not paths:
        raise SystemExit("no eval images")
    vocab = blob["vocab"]
    unk = 0
    clouds = {}
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            chunk = paths[i : i + 8]
            tok = image_tokens(enc, torch.stack([image_tensor(p, ENCODERS[name]) for p in chunk]).to(dev))
            cat = torch.tensor(
                [vocab.get(p.parent.parent.name, unk) for p in chunk],
                dtype=torch.long,
                device=dev,
            )
            if net is None:
                pred = units[nearest_shape(descriptor(tok.cpu()), [p.parent.parent.name for p in chunk], bank, bank_cat)]
            else:
                pred = net(tok, cat).cpu().numpy()
            for p, cloud in zip(chunk, pred):
                clouds[p.parent.name] = to_mm(cloud, p.parent.name, table, rule) if table is not None else cloud
            print(f"predict {min(i + len(chunk), len(paths))}/{len(paths)}", flush=True)
    out = ROOT / "submissions" / ("atlas" if net is not None else "retrieve") / "preds_geometry.zip"
    pack(out, clouds)
    print(f"{len(clouds)} clouds -> {out}")


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


def score(tag: str, variants: dict, gt: torch.Tensor, big: torch.Tensor, dev: torch.device) -> None:
    for name, clouds in variants.items():
        cd = torch.tensor([aligned_cd(clouds[k].to(dev), gt[k].to(dev)) for k in range(len(gt))])
        line = f"{tag:18s} {name:24s} CD-L2 {cd.mean():8.1f}  median {cd.median():7.1f}"
        if big.any():
            line += f"  normal {cd[~big].mean():8.1f}  oversized {cd[big].mean():9.1f}"
        print(line, flush=True)


def sized(unit: torch.Tensor, ids: list[str], r: torch.Tensor, size: torch.Tensor | None) -> dict:
    """A unit-shape prediction at the true size, the size head's size, and hedged."""
    out = {"true size": unit * r[:, None, None]}
    if size is not None:
        out["size head"] = unit * size[:, None, None]
        table, rule = load_sizes()
        if rule["frac"] > 0:
            out["size head + hedge"] = [torch.from_numpy(to_mm(unit[k].numpy(), i, table, rule)).float() for k, i in enumerate(ids)]
    return out


def evaluate(ckpts: list[Path], retrieval: bool) -> None:
    """Score holdout predictions the way the grader does, with the predicted and the true size."""
    z, vocab = _load_cache()
    dev = device()
    val, tr, _ = holdout(len(z["ids"]))
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
    if retrieval:
        pick = nearest_shape(descriptor(feats), z["cats"][val].tolist(), bank_descriptors(z["feats"], tr), z["cats"][tr].tolist())
        bank = torch.tensor(z["pts"][tr[pick]], dtype=torch.float32)
        score("retrieval", sized(bank / radii(bank)[:, None, None], ids, r, size), gt, big, dev)
    for ckpt in ckpts:
        net, blob = load_net(ckpt, dev)
        if blob.get("encoder", "dinov2b") != "dinov2b":
            feats = torch.tensor(_load_cache(blob["encoder"])[0]["feats"][val], dtype=torch.float32)
        else:
            feats = torch.tensor(z["feats"][val], dtype=torch.float32)
        with torch.inference_mode():
            pred = torch.cat([net(feats[i : i + 8].to(dev), cat[i : i + 8].to(dev)).cpu() for i in range(0, len(ids), 8)])
        if blob.get("unit"):
            variants = sized(pred, ids, r, size)
        else:
            own = radii(pred - pred.mean(1, keepdim=True))
            variants = {"as trained, mm": pred, "rescaled to true size": pred * (r / own)[:, None, None]}
            if size is not None:
                variants["rescaled to size head"] = pred * (size / own)[:, None, None]
        score(ckpt.name, variants, gt, big, dev)


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
    deep = DeepAtlas(8, 1, n=128, patches=2, depth=2, heads=4)
    opt = torch.optim.Adam(deep.parameters(), lr=3e-3)
    d0 = float(chamfer(deep(feat, cat), target).detach())
    for _ in range(60):
        loss = chamfer(deep(feat, cat), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    d1 = float(chamfer(deep(feat, cat), target).detach())
    assert d1 < d0 * 0.5, (d0, d1)
    flipped = target * torch.tensor([-1.0, -1.0, 1.0])
    assert float(chamfer(target, flipped)) < 1e-9 < float(chamfer(target, flipped, flips=False))
    print(f"fit check ok  {first:.1f} -> {last:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["cache", "train", "predict", "evaluate", "check"])
    ap.add_argument("--ckpt", type=Path, nargs="+", default=[CKPT])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="deep", help="train: decoder (atlas = the original)")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="dinov2b", help="cache, train: image tokens")
    ap.add_argument("--shape", choices=["decoder", "retrieve"], default="decoder", help="predict: decoder or nearest training shape")
    ap.add_argument("--retrieval", action="store_true", help="evaluate: also score the nearest-training-shape baseline")
    args = ap.parse_args()
    if args.cmd == "cache":
        cache(args.encoder)
    elif args.cmd == "train":
        train(args.ckpt[0], args.epochs, args.arch, args.encoder)
    elif args.cmd == "predict":
        predict(args.ckpt[0], args.shape)
    elif args.cmd == "evaluate":
        evaluate(args.ckpt if args.ckpt != [CKPT] or CKPT.exists() else [], args.retrieval)
    else:
        fit_check()
