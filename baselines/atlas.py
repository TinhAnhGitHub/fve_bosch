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
import math
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
ENCODERS = {"dinov2b": 224, "dinov3l": 256, "dinov3l_ml": 256, "dinov3l384": 384, "dav2s": 224}  # name -> input size; 257 tokens, 577 at 384 px
DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
N = MIN_POINTS
DENSE = 8 * MIN_POINTS
DENSE_CACHE = OUT / "train_dense.npz"
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


def image_tensor(path: Path, size: int = IMAGE, flip: bool = False) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    x = (x - mean) / std
    return x.flip(2) if flip else x


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


def half_dtype(dev: torch.device) -> torch.dtype:
    """bf16 on Ampere and newer; fp16 on older GPUs such as the T4, which have no bf16 (train() scales the loss then)."""
    return torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16


class DinoV3(nn.Module):
    """Hugging Face DINOv3 returning [class token, patch tokens], registers dropped."""

    def __init__(self, layers: tuple[int, ...] = ()) -> None:
        super().__init__()
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(DINOV3)
        self.registers = self.net.config.num_register_tokens
        self.layers = layers  # also average these hidden layers (each layer-normalised) with the last one

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.net(pixel_values=x, output_hidden_states=bool(self.layers))
        out = res.last_hidden_state
        if self.layers:
            out = torch.stack([out, *(self.net.norm(res.hidden_states[i]) for i in self.layers)]).mean(0)
        return torch.cat([out[:, :1], out[:, 1 + self.registers :]], 1)


class DepthTok(nn.Module):
    """Depth Anything V2 Small (Apache-2.0) at 224 px: [class, 16 x 16 patch] tokens of 384 depth-trained features,
    plus one channel with each patch's relative depth (z-scored per image; 0 on the class token). Same 16 x 16 grid
    as DINOv3 at 256 px, so its tokens can be appended to the DINOv3 tokens channel-wise (train --extra dav2s)."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.net = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x, output_hidden_states=True)
        tok = out.hidden_states[-1]  # (b, 1 + 256, 384) from the ViT-S/14 backbone
        d = torch.nn.functional.adaptive_avg_pool2d(out.predicted_depth[:, None], 16).flatten(1)  # (b, 256)
        d = (d - d.mean(1, keepdim=True)) / (d.std(1, keepdim=True) + 1e-6)
        d = torch.cat([torch.zeros_like(d[:, :1]), d], 1)[..., None]
        return torch.cat([tok, d], -1)


def encoder(dev: torch.device, name: str = "dinov2b") -> nn.Module:
    if name == "dav2s":
        net = DepthTok()
        net.eval().to(dev)
        return net
    net = DinoV3((12, 18) if name == "dinov3l_ml" else ()) if name.startswith("dinov3l") else torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    net.eval().to(dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def embed(net: nn.Module, paths: list[Path], dev: torch.device, batch: int = 32, size: int = IMAGE, flip: bool = False) -> np.ndarray:
    chunks = []
    for i in range(0, len(paths), batch):
        x = torch.stack([image_tensor(p, size, flip) for p in paths[i : i + batch]]).to(dev)
        chunks.append(image_tokens(net, x).half().cpu())
        print(f"embed {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(chunks).numpy()


def image_tokens(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Class token plus patch tokens. The class token alone scored 733."""
    if isinstance(net, (DinoV3, DepthTok)):
        return net(x).float()
    out = net.forward_features(x)
    return torch.cat([out["x_norm_clstoken"][:, None], out["x_norm_patchtokens"]], 1).float()


def fourier_uv(uv: torch.Tensor, harmonics: int) -> torch.Tensor:
    """uv, r^2 and sin/cos(k * pi * uv) for k <= harmonics: (n, 2) -> (n, 3 + 4 * harmonics).

    Low frequencies only: the decoder already overfits, and high ones can fit mesh noise.
    """
    if not harmonics:
        return uv
    wave = uv[..., None] * torch.arange(1, harmonics + 1) * torch.pi
    return torch.cat([uv, uv.square().sum(-1, keepdim=True), wave.sin().flatten(1), wave.cos().flatten(1)], 1)


class Atlas(nn.Module):
    """Fold PATCHES grids into N millimetre points, attending to image tokens."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, fourier: int = 0) -> None:
        super().__init__()
        g = int((n // patches) ** 0.5)
        if g * g * patches != n:
            raise ValueError(f"{n} is not {patches} square grids")
        axis = torch.linspace(-1.0, 1.0, g)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", fourier_uv(torch.stack([xx, yy], -1).reshape(-1, 2), fourier))
        self.patch = nn.Parameter(torch.randn(patches, 16) * 0.02)
        self.cat = nn.Embedding(n_cat, 32)
        width = 256
        self.q = nn.Linear(32 + 16 + self.grid.shape[1], width)
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

    attn_drop = 0.0  # training only, set by train(); neither changes the checkpoint format
    drop_path = 0.0

    def path(self, r: torch.Tensor) -> torch.Tensor:
        """DropPath: while training, skip this residual for a random share of the objects."""
        if not self.training or not self.drop_path:
            return r
        keep = (torch.rand((r.shape[0], 1, 1), device=r.device) >= self.drop_path).to(r.dtype)
        return r * keep / (1 - self.drop_path)

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def attend(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        h = self.heads
        q = self.q(self.norm_q(x)).reshape(b, n, h, w // h).transpose(1, 2)
        k, v = self.kv(tok).reshape(b, tok.shape[1], 2, h, w // h).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj(a.transpose(1, 2).reshape(b, n, w))

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        x = x + self.path(self.attend(x, tok))
        return x + self.path(self.mlp(self.norm_mlp(x)))


class SheetBlock(Block):
    """Block plus a zero-initialised 3x3 depthwise convolution along each sheet and a per-sheet offset.

    Both start as the identity and add about 2.8k parameters per block: the decoder overfits already.
    """

    def __init__(self, width: int, heads: int, sheets: int, side: int) -> None:
        super().__init__(width, heads)
        self.sheets, self.side = sheets, side
        self.gate = nn.Parameter(torch.zeros(width))
        self.conv = nn.Conv2d(width, width, 3, padding=1, groups=width)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        s, g = self.sheets, self.side
        x = x + self.path(self.attend(x, tok))
        sheet = x.reshape(b, s, g * g, w).mean(2, keepdim=True)
        x = x + (self.gate.to(x.dtype) * (sheet - sheet.mean(1, keepdim=True))).expand(-1, -1, g * g, -1).reshape(b, n, w)
        x = x + self.path(self.mlp(self.norm_mlp(x)))
        near = self.conv(x.reshape(b * s, g, g, w).permute(0, 3, 1, 2))
        return x + self.path(near.permute(0, 2, 3, 1).reshape(b, n, w))


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

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, fourier)
        width = self.q.out_features
        del self.k, self.v, self.mlp
        self.norm_tok = nn.LayerNorm(dim)
        self.tok = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        self.feedback = None  # CoordAtlas: each block also sees where the previous blocks put the points
        self.hint_in = None  # add_hint(): a per-object size hint added to every query
        self.pos_tok = None  # add_pos(): where each image token sits in the image
        self.scale_mlp = None  # add_scale(): the object's size comes from its own head
        self.img_query = None  # use_image_query(): the queries carry the image's global features, not the category

    def add_hint(self, size: int = 3) -> None:
        self.hint_in = nn.Linear(size, self.q.out_features)
        nn.init.zeros_(self.hint_in.weight)
        nn.init.zeros_(self.hint_in.bias)

    def add_pos(self, tokens: int = 257) -> None:
        """A learned embedding per image token (class token + 16 x 16 patches), added after projection.

        Cross-attention treats the tokens as an unordered set, so without it the decoder
        knows where a patch lies only if the frozen features happen to encode it.
        Zero start: the model begins exactly as the version without it.
        """
        self.pos_tok = nn.Parameter(torch.zeros(tokens, self.q.out_features))

    def use_image_query(self, dim: int) -> None:
        """Replace the category embedding in every query by a projection of the image's class token and mean patch
        token, so the model needs no category label (train --no-cat)."""
        self.img_query = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, self.cat.embedding_dim))

    def add_scale(self, dim: int, cat_logr: torch.Tensor, unit_mm: float) -> None:
        """Split size from shape: the points are rescaled to a size predicted by a separate head.

        The head reads the class token, the mean patch token and the category, and predicts
        log RMS radius (mm) as an offset from the category's mean, with a spread sigma.
        Squared Chamfer is lowest at the mean size, so the output uses exp(mu + sigma^2 / 2),
        not the median exp(mu). The last layer starts at zero: the first guess is the
        category mean.
        """
        width = self.q.out_features
        self.scale_mlp = nn.Sequential(nn.LayerNorm(2 * dim + self.cat.embedding_dim), nn.Linear(2 * dim + self.cat.embedding_dim, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)
        with torch.no_grad():
            self.scale_mlp[-1].bias[1] = math.log(0.3)  # start with a 30% spread, not 100%
        self.register_buffer("cat_logr", cat_logr.float().clone())
        self.unit_mm = float(unit_mm)

    scale_out: tuple[torch.Tensor, torch.Tensor] | None = None  # (mu, log sigma) of log radius in mm, for the size loss

    def rescale(self, y: torch.Tensor, tok: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        c = y.mean(1, keepdim=True)
        u = y - c
        rms = u.square().sum(-1).mean(1).sqrt().clamp_min(1e-6)[:, None, None]
        feat = torch.cat([tok[:, 0].float(), tok[:, 1:].float().mean(1), self.cat(cat).float()], -1)
        mu, log_sig = self.scale_mlp(feat).unbind(-1)
        mu = mu + self.cat_logr[cat]
        log_sig = log_sig.clamp(-4.0, 1.0)
        self.scale_out = (mu, log_sig)
        size = torch.exp(mu + 0.5 * torch.exp(2 * log_sig)) / self.unit_mm
        return c + u / rms * size[:, None, None]

    grid_jitter = 0.0  # training only, in grid steps: set by train()
    deep_sup = 0.0  # training only: when set, forward() leaves the earlier blocks' outputs in aux_out
    aux_out: list[torch.Tensor] = []

    def forward(self, tok: torch.Tensor, cat: torch.Tensor, keep: torch.Tensor | None = None, hint: torch.Tensor | None = None) -> torch.Tensor:
        b, p, g = tok.shape[0], self.patch.shape[0], self.grid.shape[0]
        grid = self.grid[None, None].expand(b, p, g, -1)
        if self.training and self.grid_jitter:
            step = 2.0 / (int(g**0.5) - 1)
            grid = grid + (torch.rand_like(grid) - 0.5) * 2.0 * self.grid_jitter * step
        cond = self.cat(cat) if self.img_query is None else self.img_query(torch.cat([tok[:, 0], tok[:, 1:].mean(1)], -1).float())
        parts = [
            cond[:, None, None, :].expand(b, p, g, -1),
            self.patch[None, :, None, :].expand(b, p, g, -1),
            grid,
        ]
        x = torch.cat(parts, -1).reshape(b, p * g, -1)
        if keep is not None:
            x = x[:, keep]
        with torch.autocast(tok.device.type, dtype=half_dtype(tok.device), enabled=tok.is_cuda):
            x = self.q(x)
            if hint is not None:
                x = x + self.hint_in(hint)[:, None, :]
            t = self.tok(self.norm_tok(tok))
            if self.pos_tok is not None:
                t = t + self.pos_tok[: t.shape[1]]
            aux = []
            for i, block in enumerate(self.blocks):
                if self.feedback is not None and i:
                    with torch.autocast(tok.device.type, enabled=False):
                        pos = self.out(x.float()).detach()
                    x = x + self.feedback[i - 1](pos)
                x = block(x, t)
                if self.training and self.deep_sup and i < len(self.blocks) - 1:  # same head as the final output
                    with torch.autocast(tok.device.type, enabled=False):
                        aux.append(self.out(x.float()))
            self.aux_out = aux
        y = self.out(x.float())  # fp32 head: millimetre targets need the precision
        if self.scale_mlp is not None:
            y = self.rescale(y, tok, cat)
        return y


class SheetAtlas(DeepAtlas):
    """DeepAtlas with SheetBlocks. Needs the whole grid, so it cannot subsample queries."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, depth, heads, fourier)
        side = int((n // patches) ** 0.5)
        self.blocks = nn.ModuleList(SheetBlock(self.q.out_features, heads, patches, side) for _ in range(depth))


class TalkBlock(SheetBlock):
    """SheetBlock whose 8 sheet summaries also attend to each other, so sheets can divide the surface.

    The conversation runs at 64 dimensions and its output starts at zero, so the block begins as a SheetBlock.
    """

    def __init__(self, width: int, heads: int, sheets: int, side: int, talk: int = 64) -> None:
        super().__init__(width, heads, sheets, side)
        self.talk_norm = nn.LayerNorm(width)
        self.talk_in = nn.Linear(width, talk)
        self.talk = nn.MultiheadAttention(talk, 4, batch_first=True)
        self.talk_out = nn.Linear(talk, width)
        nn.init.zeros_(self.talk_out.weight)
        nn.init.zeros_(self.talk_out.bias)

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        s, g = self.sheets, self.side
        x = x + self.path(self.attend(x, tok))
        sheet = x.reshape(b, s, g * g, w).mean(2, keepdim=True)
        x = x + (self.gate.to(x.dtype) * (sheet - sheet.mean(1, keepdim=True))).expand(-1, -1, g * g, -1).reshape(b, n, w)
        summary = self.talk_in(self.talk_norm(x.reshape(b, s, g * g, w).mean(2)))
        said, _ = self.talk(summary, summary, summary, need_weights=False)
        x = x + self.path(self.talk_out(said)[:, :, None].expand(-1, -1, g * g, -1).reshape(b, n, w))
        x = x + self.path(self.mlp(self.norm_mlp(x)))
        near = self.conv(x.reshape(b * s, g, g, w).permute(0, 3, 1, 2))
        return x + self.path(near.permute(0, 2, 3, 1).reshape(b, n, w))


class TalkAtlas(SheetAtlas):
    """SheetAtlas with TalkBlocks: sheet-level attention between the 8 sheets in every block."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, depth, heads, fourier)
        side = int((n // patches) ** 0.5)
        self.blocks = nn.ModuleList(TalkBlock(self.q.out_features, heads, patches, side) for _ in range(depth))


class CoordAtlas(TalkAtlas):
    """TalkAtlas where every block after the first is told the 3D positions the decoder has reached so far."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, depth, heads, fourier)
        self.feedback = nn.ModuleList(nn.Linear(3, self.q.out_features) for _ in range(depth - 1))
        for layer in self.feedback:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


class PatchTalkBlock(SheetBlock):
    """SheetBlock whose sheets talk through 4x4 regions each, 128 tokens in all, not one summary per sheet."""

    def __init__(self, width: int, heads: int, sheets: int, side: int, talk: int = 64, cells: int = 4) -> None:
        super().__init__(width, heads, sheets, side)
        self.cells = cells
        self.talk_norm = nn.LayerNorm(width)
        self.talk_in = nn.Linear(width, talk)
        self.talk = nn.MultiheadAttention(talk, 4, batch_first=True)
        self.talk_out = nn.Linear(talk, width)
        nn.init.zeros_(self.talk_out.weight)
        nn.init.zeros_(self.talk_out.bias)

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        s, g, c = self.sheets, self.side, self.cells
        x = x + self.path(self.attend(x, tok))
        sheet = x.reshape(b, s, g * g, w).mean(2, keepdim=True)
        x = x + (self.gate.to(x.dtype) * (sheet - sheet.mean(1, keepdim=True))).expand(-1, -1, g * g, -1).reshape(b, n, w)
        y = self.talk_in(self.talk_norm(x)).reshape(b * s, g, g, -1).permute(0, 3, 1, 2)
        cell = torch.nn.functional.avg_pool2d(y, g // c).flatten(2).transpose(1, 2).reshape(b, s * c * c, -1)
        said, _ = self.talk(cell, cell, cell, need_weights=False)
        said = said.reshape(b * s, c, c, -1).permute(0, 3, 1, 2)
        said = torch.nn.functional.interpolate(said.float(), size=(g, g), mode="bilinear", align_corners=False).to(y.dtype)
        x = x + self.path(self.talk_out(said.permute(0, 2, 3, 1).reshape(b, n, -1)))
        x = x + self.path(self.mlp(self.norm_mlp(x)))
        near = self.conv(x.reshape(b * s, g, g, w).permute(0, 3, 1, 2))
        return x + self.path(near.permute(0, 2, 3, 1).reshape(b, n, w))


class PatchTalkAtlas(SheetAtlas):
    """SheetAtlas with PatchTalkBlocks."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, depth, heads, fourier)
        side = int((n // patches) ** 0.5)
        self.blocks = nn.ModuleList(PatchTalkBlock(self.q.out_features, heads, patches, side) for _ in range(depth))


class SphereBlock(Block):
    """Block for one closed 64 x 128 latitude-longitude grid: region attention, MLP, seam-free 3x3 filter."""

    def __init__(self, width: int, heads: int, rows: int = 64, cols: int = 128, talk: int = 64, cell: int = 16) -> None:
        super().__init__(width, heads)
        self.rows, self.cols, self.cell = rows, cols, cell
        self.talk_norm = nn.LayerNorm(width)
        self.talk_in = nn.Linear(width, talk)
        self.talk = nn.MultiheadAttention(talk, 4, batch_first=True)
        self.talk_out = nn.Linear(talk, width)
        self.conv = nn.Conv2d(width, width, 3, groups=width)
        for layer in (self.talk_out, self.conv):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        r, c, k = self.rows, self.cols, self.cell
        x = x + self.path(self.attend(x, tok))
        y = self.talk_in(self.talk_norm(x)).reshape(b, r, c, -1).permute(0, 3, 1, 2)
        cell = torch.nn.functional.avg_pool2d(y, k)
        cr, cc = cell.shape[2:]
        cell = cell.flatten(2).transpose(1, 2)
        said, _ = self.talk(cell, cell, cell, need_weights=False)
        said = said.transpose(1, 2).reshape(b, -1, cr, cc)
        said = torch.nn.functional.interpolate(said.float(), size=(r, c), mode="bilinear", align_corners=False).to(y.dtype)
        x = x + self.path(self.talk_out(said.permute(0, 2, 3, 1).reshape(b, n, -1)))
        x = x + self.path(self.mlp(self.norm_mlp(x)))
        z = x.reshape(b, r, c, w).permute(0, 3, 1, 2)
        z = torch.nn.functional.pad(torch.nn.functional.pad(z, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
        return x + self.path(self.conv(z).permute(0, 2, 3, 1).reshape(b, n, w))


class SphereAtlas(DeepAtlas):
    """One closed grid on a unit sphere in place of 8 flat sheets: queries are 3D sphere positions."""

    def __init__(self, dim: int, n_cat: int, n: int = N, patches: int = PATCHES, depth: int = 4, heads: int = 8, fourier: int = 0) -> None:
        super().__init__(dim, n_cat, n, patches, depth, heads, fourier)
        rows, cols = 64, 128
        theta = (torch.arange(rows) + 0.5) / rows * torch.pi
        phi = (torch.arange(cols) + 0.5) / cols * 2.0 * torch.pi
        tt, pp = torch.meshgrid(theta, phi, indexing="ij")
        self.register_buffer("grid", torch.stack([tt.sin() * pp.cos(), tt.sin() * pp.sin(), tt.cos()], -1).reshape(-1, 3))
        self.patch = nn.Parameter(torch.randn(1, 16) * 0.02)
        width = self.q.out_features
        self.q = nn.Linear(32 + 16 + 3, width)
        self.blocks = nn.ModuleList(SphereBlock(width, heads, rows, cols) for _ in range(depth))


ARCHS = {"atlas": Atlas, "deep": DeepAtlas, "sheet": SheetAtlas, "talk": TalkAtlas, "coord": CoordAtlas, "patch": PatchTalkAtlas, "sphere": SphereAtlas}


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


def chamfer(pred: torch.Tensor, gt: torch.Tensor, flips: bool = True, chunk: int = 2048, pick: int = 1024, weight: torch.Tensor | None = None, sides: tuple[float, float] = (1.0, 1.0), l1: bool = False) -> torch.Tensor:
    """Training loss: exact CD-L2 (l1=True: plain distances, a robust variant) over every point, at the best PCA sign flip of each target.

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
    to_gt = (pred - torch.gather(gf, 1, j1[..., None].expand(-1, -1, 3))).square().sum(-1)
    to_pred = (gf - torch.gather(pred, 1, j2[..., None].expand(-1, -1, 3))).square().sum(-1)
    if l1:  # un-squared distances: far outliers pull less, near pairs more (HyperCD-like)
        to_gt, to_pred = (to_gt + 1e-8).sqrt(), (to_pred + 1e-8).sqrt()
    to_gt, to_pred = to_gt.mean(1), to_pred.mean(1)
    cd = sides[0] * to_gt + sides[1] * to_pred
    return (cd if weight is None else cd * weight).mean()


def sliced_w2(pred: torch.Tensor, gt: torch.Tensor, dirs: int = 64) -> torch.Tensor:
    """Sliced Wasserstein-2: both clouds projected on random directions, sorted, matched one to one.

    Chamfer lets many predicted points sit on one target point and leave others unmatched;
    an equal-count match cannot, so it pushes the cloud out to cover the whole surface
    (our predictions crowd inward: RMS radius 0.77x the truth). pred (b, n, 3), gt (b, m, 3).
    """
    n = min(pred.shape[1], gt.shape[1])
    if gt.shape[1] != n:
        gt = gt[:, torch.randperm(gt.shape[1], device=gt.device)[:n]]
    if pred.shape[1] != n:
        pred = pred[:, torch.randperm(pred.shape[1], device=pred.device)[:n]]
    d = torch.randn(3, dirs, device=pred.device)
    d = d / d.norm(dim=0, keepdim=True)
    a = torch.sort(pred.float() @ d, dim=1).values
    b = torch.sort(gt.float() @ d, dim=1).values
    return (a - b).square().mean((1, 2)).mean()


def repulsion(pred: torch.Tensor, k: int = 4, pick: int = 1024) -> torch.Tensor:
    """Penalise points closer to their neighbours than an even cover of the object would put them.

    8192 points spread evenly over a sphere of radius r sit about 0.04 r apart. Most predicted
    points are nobody's nearest, so clumps waste points that could cover the surface.
    Neighbours come from a random subset of `pick` points to keep this cheap.
    """
    b, n = pred.shape[:2]
    q = pred[:, torch.randperm(n, device=pred.device)[:pick]]
    with torch.no_grad():
        j = torch.cdist(q.detach(), pred.detach()).topk(k + 1, largest=False).indices[..., 1:]
        r = (pred.detach() - pred.detach().mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt()
    near = torch.gather(pred, 1, j.reshape(b, -1, 1).expand(-1, -1, 3)).reshape(b, pick, k, 3)
    d2 = (q[:, :, None] - near).square().sum(-1)
    return ((0.04 * r)[:, None, None].square() - d2).clamp(min=0).mean()


def size_hints(query: torch.Tensor, qcat: torch.Tensor, bank: torch.Tensor, bcat: torch.Tensor, blogr: torch.Tensor, self_idx: torch.Tensor | None = None, k: int = 2) -> torch.Tensor:
    """Size of the k most similar bank objects: normalised mean log radius, nearest log radius, mean similarity.

    Similarity is cosine between class tokens, inside the query's category when it has at least k other
    members, else over the whole bank. self_idx leaves each query's own row out, so training objects
    cannot read their own size.
    """
    sim = torch.nn.functional.normalize(query, dim=-1) @ torch.nn.functional.normalize(bank, dim=-1).T
    same = qcat[:, None] == bcat[None, :]
    if self_idx is not None:
        here = torch.arange(len(query), device=query.device)
        sim[here, self_idx] = -float("inf")
        same[here, self_idx] = False
    pick = torch.where(same.sum(1, keepdim=True) >= k, torch.where(same, sim, torch.full_like(sim, -float("inf"))), sim)
    top = pick.topk(k, 1)
    r = (blogr[top.indices] - np.log(30.0)) / 0.5
    return torch.stack([r.mean(1), r[:, 0], top.values.mean(1)], 1)


def _mesh_row(img: str) -> tuple[str, str, np.ndarray]:
    path = Path(img)
    return path.parent.name, path.parent.parent.name, target_points(path.parent / "mesh.ply", N)


def _dense_row(img: str) -> np.ndarray:
    return target_points(Path(img).parent / "mesh.ply", DENSE).astype(np.float32)


def cache_dense() -> None:
    """DENSE surface samples per training object, so --resample can draw fresh targets every step."""
    from concurrent.futures import ProcessPoolExecutor

    paths = rows("train")
    with np.load(OUT / "train.npz", allow_pickle=False) as z:
        ids = z["ids"]
    if ids.tolist() != [p.parent.name for p in paths]:
        raise SystemExit("train.npz is stale. Run: uv run baselines/atlas.py cache")
    with ProcessPoolExecutor() as pool:
        pts = list(pool.map(_dense_row, map(str, paths), chunksize=16))
    np.savez(DENSE_CACHE, ids=ids, pts=np.stack(pts))
    print(f"cached {len(ids)} objects x {DENSE} points -> {DENSE_CACHE}")


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


def augment_image(path: Path, size: int, rng: np.random.Generator) -> torch.Tensor:
    """A label-preserving view of one training image.

    In-plane rotation (+-20 deg), zoom (0.9-1.1) and colour jitter (+-20%). The target is a
    PCA-canonical cloud, which does not depend on how the object sits in the picture, so it
    stays the same. The camera auto-frames every object, so zoom does not change its size.
    """
    return augment_pil(Image.open(path).convert("RGB"), size, rng)


def augment_pil(im: Image.Image, size: int, rng: np.random.Generator) -> torch.Tensor:
    """augment_image on an image already in memory (train --finetune reads the photo pack)."""
    from PIL import ImageEnhance

    w, h = im.size
    bg = tuple(int(v) for v in np.median(np.asarray(im)[:8].reshape(-1, 3), 0))  # backdrop colour along the top edge
    im = im.rotate(float(rng.uniform(-20, 20)), resample=Image.BICUBIC, fillcolor=bg)
    z = float(rng.uniform(0.9, 1.1))
    cw, ch = int(w / z), int(h / z)
    canvas = Image.new("RGB", (cw, ch), bg)
    canvas.paste(im, ((cw - w) // 2, (ch - h) // 2))
    im = canvas.resize((size, size), Image.BICUBIC)
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = enh(im).enhance(float(rng.uniform(0.8, 1.2)))
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def aug_cache_path(enc: str, k: int) -> Path:
    return OUT / f"train_{enc}_aug{k}.npz"


def augcache(enc: str, sets: int, seed: int = 0) -> None:
    """Token sets for augmented views of every training image, one file per set; finished sets are skipped."""
    paths = rows("train")
    ids = np.load(OUT / "train.npz", allow_pickle=False)["ids"]
    if ids.tolist() != [p.parent.name for p in paths]:
        raise SystemExit("train.npz does not match the training images")
    dev = device()
    net = encoder(dev, enc)
    for k in range(sets):
        if aug_cache_path(enc, k).exists():
            continue
        rng = np.random.default_rng(seed * 1000 + k)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(paths), 32):
                x = torch.stack([augment_image(p, ENCODERS[enc], rng) for p in paths[i : i + 32]]).to(dev)
                chunks.append(image_tokens(net, x).half().cpu())
                print(f"aug set {k}: {min(i + 32, len(paths))}/{len(paths)}", flush=True)
        np.savez(aug_cache_path(enc, k), ids=ids, feats=torch.cat(chunks).numpy())
        print(f"saved {aug_cache_path(enc, k)}", flush=True)


def flip_cache_path(enc: str) -> Path:
    return OUT / f"train_{enc}_flip.npz"


def cache_flip(enc: str) -> None:
    """Tokens of every horizontally flipped training image."""
    paths = rows("train")
    dev = device()
    feats = embed(encoder(dev, enc), paths, dev, size=ENCODERS[enc], flip=True)
    np.savez(flip_cache_path(enc), ids=np.array([p.parent.name for p in paths]), feats=feats)
    print(f"cached flipped tokens {feats.shape} -> {flip_cache_path(enc)}")


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


def split(ids: list[str], folds: Path | None, fold: str) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """(val, train, rng): the fixed 10% holdout without a fold file; else val = fold `fold` of the file
    (baselines/tools/make_folds.py) and train = the other folds; fold "all" trains on every object."""
    if folds is None:
        return holdout(len(ids))
    import csv

    with open(folds, newline="") as f:
        table = {r["id"]: int(r["fold"]) for r in csv.DictReader(f)}
    if sorted(table) != sorted(ids):
        raise SystemExit(f"{folds} does not list the same ids as train.npz")
    of = np.array([table[i] for i in ids])
    if fold == "all":
        return np.array([], dtype=np.int64), np.arange(len(ids)), np.random.default_rng(0)
    return np.flatnonzero(of == int(fold)), np.flatnonzero(of != int(fold)), np.random.default_rng(0)


def epoch_set(spec: str) -> set[int]:
    """'12:32' -> epochs 12..32; '14:32:2' -> every second one."""
    if not spec:
        return set()
    a, b, *s = (int(v) for v in spec.split(":"))
    return set(range(a, b + 1, s[0] if s else 1))


PHOTOS = OUT / "train_photos256.npz"


def photo_pack() -> None:
    """The training photos resized to 256 px exactly as image_tensor does, as one uint8 array (train --finetune)."""
    paths = rows("train")
    ids = np.load(OUT / "train.npz", allow_pickle=False)["ids"]
    if ids.tolist() != [p.parent.name for p in paths]:
        raise SystemExit("train.npz does not match the training images")
    size = ENCODERS["dinov3l"]
    imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)) for p in paths])
    np.savez(PHOTOS, ids=ids, imgs=imgs)
    print(f"packed {imgs.shape} -> {PHOTOS}")


def plain_pixels(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


class Tuner(nn.Module):
    """DINOv3-L/16 whose last `n` blocks and final norm train with the decoder; the rest stays frozen.

    Kept in eval mode (DINOv3 has no dropout to switch). bf16 where the GPU has it: DINOv3 gives NaN in fp16,
    so older GPUs run it in fp32.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.enc = DinoV3()
        net = self.enc.net
        blocks = net.layer if hasattr(net, "layer") else net.model.layer  # older / newer transformers
        self.enc.requires_grad_(False)
        for blk in list(blocks)[-n:]:
            blk.requires_grad_(True)
        net.norm.requires_grad_(True)
        self.n = n
        self.eval()

    def train(self, mode: bool = True) -> "Tuner":
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bf16 = x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 8
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=bf16):
            return self.enc(x).float()

    def tuned(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.named_parameters() if v.requires_grad}

    def load_tuned(self, state: dict[str, torch.Tensor]) -> None:
        own = self.tuned()
        if set(own) != set(state):
            raise SystemExit("the checkpoint's encoder blocks do not match this Tuner")
        with torch.no_grad():
            for k, v in state.items():
                own[k].copy_(v)


def tuned_encoder(blob: dict, dev: torch.device) -> Tuner | None:
    """The fine-tuned encoder a checkpoint needs (train --finetune), or None for cached-token checkpoints."""
    if not blob.get("finetune"):
        return None
    tuner = Tuner(blob["finetune"]).to(dev)
    tuner.load_tuned(blob["enc_state"])
    return tuner


def radii(pts: torch.Tensor) -> torch.Tensor:
    return pts.square().sum(-1).mean(-1).sqrt()


def right_handed(pts: np.ndarray, ids: list[str]) -> np.ndarray:
    """Undo the mirror that canonicalize's skewness signs give about half the objects.

    train_mirror.npz marks objects whose canonical frame has det -1 (from the meshes).
    Flipping their least certain axis (smallest |skewness|) makes every target a pure
    rotation of the real object, so the decoder no longer has to learn both handednesses.
    """
    with np.load(OUT / "train_mirror.npz") as m:
        mark = dict(zip(m["ids"].tolist(), m["mirror"].tolist()))
    out = pts.copy()
    for k, i in enumerate(ids):
        if mark[i]:
            c = out[k] - out[k].mean(0)
            axis = int(np.argmin(np.abs((c**3).mean(0))))
            out[k, :, axis] *= -1
    return out


def train(ckpt: Path, epochs: int, arch: str, enc: str, target: str, seed: int = 0, drop_oversized: bool = False, flips: bool = True, fourier: int = 0, resample: bool = False, token_drop: float = 0.0, all_queries: bool = False, reweight_oversized: bool = False, drop_path: float = 0.0, attn_drop: float = 0.0, grid_jitter: float = 0.0, repulse: float = 0.0, ema: float = 0.0, deep_sup: float = 0.0, flip_aug: bool = False, size_hint: bool = False, cat_drop: float = 0.0, sides: tuple[float, float] = (1.0, 1.0), pos: bool = False, right: bool = False, aug_sets: int = 0, size_head: float = 0.0, swd: float = 0.0, l1cd: bool = False, feat_noise: float = 0.0, no_cat: bool = False, extra_enc: str = "", folds: Path | None = None, fold: str = "", save_epochs: str = "", finetune: int = 0, ft_lr: float = 1e-5) -> None:
    """target unit: shape only, size from scale.py. target mm: shape and size end to end.

    On the 10% holdout the end-to-end model beat every unit shape x predicted size
    (457 vs >= 566 mm^2): trained in mm, the decoder learns to spread its points over
    the sizes it is unsure of, which squared Chamfer rewards.
    """
    z, vocab = _load_cache(enc)
    if cat_drop:
        vocab["<unk>"] = len(vocab)
    dev = device()
    feats = torch.tensor(z["feats"], dtype=torch.float32)
    if extra_enc:  # e.g. dav2s: depth tokens appended channel-wise (same 257-token grid)
        with np.load(cache_path(extra_enc), allow_pickle=False) as ez:
            if ez["ids"].tolist() != z["ids"].tolist():
                raise SystemExit(f"{cache_path(extra_enc)} does not match the training ids")
            feats = torch.cat([feats, torch.tensor(ez["feats"], dtype=torch.float32)], -1)
    flipped = None
    if flip_aug:
        with np.load(flip_cache_path(enc), allow_pickle=False) as fz:
            if fz["ids"].tolist() != z["ids"].tolist():
                raise SystemExit("flipped tokens are stale. Run: uv run baselines/atlas.py flipcache")
            flipped = torch.tensor(fz["feats"], dtype=torch.float32)
    if feats.ndim != 3:
        raise SystemExit("cache has no patch tokens. Run: python baselines/atlas.py cache")
    pts = torch.tensor(right_handed(z["pts"], z["ids"].tolist()) if right else z["pts"], dtype=torch.float32)
    r = radii(pts)
    # the deep decoder's normalised head cannot reach millimetres quickly: it predicts
    # in units of 40 mm (about the median radius). Still end to end, one global constant.
    unit_mm = 40.0 if target == "mm" and arch != "atlas" else 1.0
    unit = pts / r[:, None, None] if target == "unit" else pts / unit_mm
    cat = torch.tensor([vocab[c] for c in z["cats"].tolist()], dtype=torch.long)
    val, tr, rng = split(z["ids"].tolist(), folds, fold)
    if (folds is None) != (not fold):
        raise SystemExit("--folds and --fold go together")
    if folds is not None:
        print(f"fold {fold} of {folds.name}: train {len(tr)}, check {len(val)}")
    photos, tuner = None, None
    if finetune:  # the encoder's last blocks train too, so tokens come from the photos every step
        if enc != "dinov3l" or extra_enc or flip_aug or aug_sets or feat_noise or token_drop or size_hint or resample:
            raise SystemExit("--finetune needs --encoder dinov3l and none of --extra, --flip-aug, --aug-sets, --feat-noise, --token-drop, --size-hint, --resample")
        with np.load(PHOTOS, allow_pickle=False) as pz:
            if pz["ids"].tolist() != z["ids"].tolist():
                raise SystemExit(f"{PHOTOS} does not match train.npz. Run: uv run baselines/atlas.py photos")
            photos = pz["imgs"]
    if seed:  # same holdout, different initialisation and data order: an ensemble member
        rng = np.random.default_rng(1000 + seed)
        torch.manual_seed(seed)
    if drop_oversized:  # the size head's per-object labels; the holdout check then covers normal objects only
        with np.load(SCALE, allow_pickle=False) as s:
            flag = dict(zip(s["ids"].tolist(), s["u"].tolist()))
        normal = np.array([flag[i] == 0 for i in z["ids"].tolist()])
        tr, val = tr[normal[tr]], val[normal[val]]
        print(f"drop oversized: train {len(tr)}, holdout {len(val)} normal objects")
    weight = None
    if reweight_oversized:  # true millimetres, but flagged objects pull on the loss like a typical object
        with np.load(SCALE, allow_pickle=False) as s:
            flag = dict(zip(s["ids"].tolist(), s["u"].tolist()))
        over = np.array([flag[i] != 0 for i in z["ids"].tolist()])
        typical = float(r[torch.from_numpy(~over)].median())
        weight = torch.where(torch.from_numpy(over), (typical / r).square().clamp(max=1.0), torch.ones_like(r))
        val = val[~over[val]]
        print(f"reweight oversized: {int(over[tr].sum())} of {len(tr)} training objects, holdout {len(val)} normal objects")
    dense = None
    if resample:  # fresh surface points every step instead of the same 8192
        with np.load(DENSE_CACHE, allow_pickle=False) as d:
            if d["ids"].tolist() != z["ids"].tolist():
                raise SystemExit(f"{DENSE_CACHE} is stale. Run: uv run baselines/atlas.py dense")
            dense = torch.from_numpy(d["pts"]) / unit_mm
    net = ARCHS[arch](feats.shape[-1], len(vocab), fourier=fourier)
    hints, hint_bank = None, None
    if size_hint:
        if arch == "atlas":
            raise SystemExit("--size-hint needs a block decoder")
        net.add_hint()
        tr_t = torch.from_numpy(np.sort(tr))
        logr = r.log()
        hint_bank = {"cls": feats[tr_t, 0].clone(), "cat": cat[tr_t].clone(), "logr": logr[tr_t].clone()}
        hints = torch.zeros(len(feats), 3)
        hints[tr_t] = size_hints(hint_bank["cls"], hint_bank["cat"], hint_bank["cls"], hint_bank["cat"], hint_bank["logr"], self_idx=torch.arange(len(tr_t)))
        val_t = torch.from_numpy(val)
        hints[val_t] = size_hints(feats[val_t, 0], cat[val_t], hint_bank["cls"], hint_bank["cat"], hint_bank["logr"])
    cat_logr = None
    if size_head:
        if arch == "atlas" or target != "mm":
            raise SystemExit("--size-head needs a block decoder and --target mm")
        logr = torch.log(r)
        cat_logr = torch.full((len(vocab),), float(logr[tr].mean()))
        for c_i in torch.unique(cat[tr]).tolist():  # category means from the training split only
            cat_logr[c_i] = logr[tr][cat[tr] == c_i].mean()
        net.add_scale(feats.shape[-1], cat_logr, unit_mm)
    if no_cat:
        if arch == "atlas" or size_head:
            raise SystemExit("--no-cat needs a block decoder and no --size-head")
        net.use_image_query(feats.shape[-1])
    if pos:
        if arch == "atlas" or token_drop:
            raise SystemExit("--pos-emb needs a block decoder and all tokens")
        net.add_pos(feats.shape[1])
    net = net.to(dev)
    print(f"{arch}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")
    if deep_sup:
        if arch == "atlas":
            raise SystemExit("--deep-sup needs a block decoder")
        net.deep_sup = deep_sup
    if drop_path or attn_drop or grid_jitter:
        if arch == "atlas" or (grid_jitter and fourier):
            raise SystemExit("--drop-path, --attn-drop and --grid-jitter need the deep or sheet decoder; --grid-jitter also needs raw uv")
        net.grid_jitter = grid_jitter
        for i, blk in enumerate(net.blocks):  # stochastic depth rises from 0 at the first block
            blk.attn_drop, blk.drop_path = attn_drop, drop_path * i / (len(net.blocks) - 1)
    lr = 1e-3 if arch == "atlas" else 3e-4
    groups = [{"params": list(net.parameters()), "lr": lr}]
    if finetune:
        if arch == "atlas":
            raise SystemExit("--finetune needs a block decoder")
        tuner = Tuner(finetune).to(dev)
        groups.append({"params": list(tuner.tuned().values()), "lr": ft_lr})
        print(f"finetune: last {finetune} DINOv3 blocks + norm, {sum(p.numel() for p in tuner.tuned().values()) / 1e6:.1f}M parameters at lr {ft_lr:g}")
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda" and half_dtype(dev) == torch.float16)  # fp16 needs loss scaling; bf16 does not
    steps = epochs * -(-len(tr) // (4 if dev.type == "cuda" else 2))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.05) if arch != "atlas" else None
    batch = 4 if dev.type == "cuda" else 2
    best, best_score = None, float("inf")
    shadow, ema_best, ema_score, ema_step = None, None, float("inf"), 0
    if ema:  # a second decoder holds the running average of the weights; checked alongside the first
        shadow = ARCHS[arch](feats.shape[-1], len(vocab), fourier=fourier)
        if size_hint:
            shadow.add_hint()
        if pos:
            shadow.add_pos(feats.shape[1])
        if no_cat:
            shadow.use_image_query(feats.shape[-1])
        if size_head:
            shadow.add_scale(feats.shape[-1], cat_logr, unit_mm)
        shadow = shadow.to(dev).eval()
        shadow.load_state_dict(net.state_dict())
    enc_shadow = {k: v.detach().clone() for k, v in tuner.tuned().items()} if tuner is not None and shadow is not None else None
    blob = {"vocab": vocab, "dim": feats.shape[-1], "unit": target == "unit", "arch": arch, "encoder": enc, "unit_mm": unit_mm, "fourier": fourier, "hint_bank": hint_bank, "pos": pos, "tokens": feats.shape[1], "size_head": size_head, "n_vocab": len(vocab), "no_cat": no_cat, "extra": extra_enc, "finetune": finetune, "folds": folds.name if folds else "", "fold": fold}
    saves = epoch_set(save_epochs)
    best_enc, ema_best_enc = None, None

    def cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in state.items()}

    def check(model: nn.Module, enc_state: dict[str, torch.Tensor] | None = None) -> float:
        if not len(val):  # --fold all: nothing held out
            return float("nan")
        kept = None
        if tuner is not None and enc_state is not None:  # score with the averaged encoder blocks, then put the live ones back
            kept = {k: v.detach().clone() for k, v in tuner.tuned().items()}
            tuner.load_tuned(enc_state)
        vtotal, vseen = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val), batch):
                b = val[i : i + batch]
                tok = feats[b].to(dev) if tuner is None else tuner(torch.stack([plain_pixels(photos[k]) for k in b]).to(dev))
                pred = model(tok, cat[b].to(dev), **({"hint": hints[b].to(dev)} if hints is not None else {}))
                # shape error in mm^2 at the true size: CD scales with r^2
                for j, k in enumerate(b):
                    gain = float(r[k]) ** 2 if target == "unit" else unit_mm**2
                    vtotal += float(chamfer(pred[j : j + 1], unit[k : k + 1].to(dev), flips=flips)) * gain
                vseen += len(b)
        if kept is not None:
            tuner.load_tuned(kept)
        return vtotal / vseen

    import time

    start = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(tr)
        src = feats  # the holdout check always reads the plain tokens
        if aug_sets and (k := (epoch - 1) % (aug_sets + 1)):  # epochs cycle: plain, view 0, view 1, ...
            with np.load(aug_cache_path(enc, k - 1), allow_pickle=False) as az:
                src = torch.tensor(az["feats"], dtype=torch.float32)
        total = 0.0
        seen = 0
        for i in range(0, len(tr), batch):
            b = tr[i : i + batch]
            if tuner is not None:  # a fresh augmented view of each photo through the partly trainable encoder
                tok = tuner(torch.stack([augment_pil(Image.fromarray(photos[k]), ENCODERS[enc], rng) for k in b]).to(dev))
            else:
                tok = src[b]
            if feat_noise:  # Gaussian noise on the frozen tokens, scaled per token
                tok = tok + feat_noise * tok.std(-1, keepdim=True) * torch.randn_like(tok)
            if flipped is not None:  # the mirrored image has the same canonical target (geom.canonicalize fixes signs by skewness)
                pick_flip = torch.from_numpy(rng.random(len(b)) < 0.5)
                tok = torch.where(pick_flip[:, None, None], flipped[b], tok)
            if token_drop:  # the class token stays; a random share of the patch tokens does not
                kept = torch.randperm(tok.shape[1] - 1)[: int((tok.shape[1] - 1) * (1 - token_drop))] + 1
                tok = tok[:, torch.cat([torch.zeros(1, dtype=torch.long), kept])]
            c_b = cat[b]
            if cat_drop:  # learn a fallback for unseen categories, and lean less on the label
                c_b = torch.where(torch.rand(len(b)) < cat_drop, torch.full_like(c_b, vocab["<unk>"]), c_b)
            extra = {"hint": hints[b].to(dev)} if hints is not None else {}
            if arch == "deep" and not all_queries:
                pred = net(tok.to(dev), c_b.to(dev), keep=torch.randperm(N, device=dev)[: N // 2], **extra)
            else:
                pred = net(tok.to(dev), c_b.to(dev), **extra)
            gt = (unit[b] if dense is None else dense[b][:, torch.randperm(DENSE)[:N]]).to(dev)
            loss = chamfer(pred, gt, flips=flips, weight=None if weight is None else weight[b].to(dev), sides=sides, l1=l1cd)
            if repulse:
                loss = loss + repulse * repulsion(pred)
            if swd:  # equal-count transport term next to Chamfer, same units
                loss = loss + swd * sliced_w2(pred, gt)
            if size_head:  # Gaussian NLL of the true log radius under the head's (mu, sigma)
                mu, log_sig = net.scale_out
                z_ = (torch.log(r[b]).to(dev) - mu) / torch.exp(log_sig)
                loss = loss + size_head * (log_sig + 0.5 * z_.square()).mean()
            if deep_sup:  # each earlier block also has to draw the object, at a smaller weight
                for early in net.aux_out:
                    loss = loss + deep_sup * chamfer(early, gt, flips=flips)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            if shadow is not None:  # early on, the average follows the weights more closely
                ema_step += 1
                d = min(ema, (1 + ema_step) / (10 + ema_step))
                with torch.no_grad():
                    for s, v in zip(shadow.state_dict().values(), net.state_dict().values()):
                        s.lerp_(v, 1 - d) if s.is_floating_point() else s.copy_(v)
                    if enc_shadow is not None:
                        for k, v in tuner.tuned().items():
                            enc_shadow[k].lerp_(v.detach(), 1 - d)
            total += float(loss.detach()) * len(b)
            seen += len(b)
        net.eval()
        score = check(net)
        if score < best_score:
            best_score = score
            best = cpu(net.state_dict())
            best_enc = cpu(tuner.tuned()) if tuner is not None else None
        note = ""
        if shadow is not None:
            now = check(shadow, enc_shadow)
            note = f"  ema {now:.1f}"
            if now < ema_score:
                ema_score = now
                ema_best = cpu(shadow.state_dict())
                ema_best_enc = cpu(enc_shadow) if enc_shadow is not None else None
        if epoch in saves:  # fixed-epoch snapshots of the averaged weights (the raw ones without --ema)
            snap = ckpt.with_name(f"{ckpt.stem}_{'ema_' if shadow is not None else ''}e{epoch}.pt")
            enc_now = enc_shadow if shadow is not None else (tuner.tuned() if tuner is not None else None)
            snap.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": cpu((shadow if shadow is not None else net).state_dict()), **blob, "epoch": epoch, **({"enc_state": cpu(enc_now)} if enc_now is not None else {})}, snap)
        print(f"epoch {epoch}  train {total / seen * (unit_mm**2 if target == 'mm' else 1):.4f}  holdout {score:.1f} mm^2" + (" at true size" if target == "unit" else "") + note + f"  {time.time() - start:.0f}s", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if best is None:  # nothing held out (--fold all): the final weights
        best, best_enc = cpu(net.state_dict()), (cpu(tuner.tuned()) if tuner is not None else None)
        if shadow is not None:
            ema_best, ema_best_enc = cpu(shadow.state_dict()), (cpu(enc_shadow) if enc_shadow is not None else None)
    torch.save({"state": best, **blob, **({"enc_state": best_enc} if best_enc is not None else {})}, ckpt)
    print(f"saved {ckpt}  holdout {best_score:.1f}")
    if shadow is not None:
        path = ckpt.with_name(f"{ckpt.stem}_ema.pt")
        torch.save({"state": ema_best, **blob, **({"enc_state": ema_best_enc} if ema_best_enc is not None else {})}, path)
        print(f"saved {path}  holdout {ema_score:.1f}")


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
    net = ARCHS[blob.get("arch", "atlas")](blob["dim"], len(blob["vocab"]), fourier=blob.get("fourier", 0))
    if blob.get("hint_bank") is not None:
        net.add_hint()
    if blob.get("pos"):
        net.add_pos(blob["tokens"])
    if blob.get("no_cat"):
        net.use_image_query(blob["dim"])
    if blob.get("size_head"):  # the category means come back with the weights
        net.add_scale(blob["dim"], torch.zeros(len(blob["vocab"])), blob["unit_mm"])
    net = net.to(dev).eval()
    net.load_state_dict(blob["state"])
    if blob.get("hint_bank") is not None:
        net = Hinted(net, blob["hint_bank"]).to(dev).eval()
    if blob.get("unit_mm", 1.0) != 1.0:
        net = Scaled(net, blob["unit_mm"])
    return net, blob


class Hinted(nn.Module):
    """Look up the size hint from the training objects stored in the checkpoint, so callers need not."""

    def __init__(self, net: nn.Module, bank: dict) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("cls", bank["cls"])
        self.register_buffer("cat", bank["cat"])
        self.register_buffer("logr", bank["logr"])

    def forward(self, tok: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.net(tok, cat, hint=size_hints(tok[:, 0].float(), cat, self.cls, self.cat, self.logr))


class Scaled(nn.Module):
    """Multiply a decoder's output by the constant it was trained in (see train)."""

    def __init__(self, net: nn.Module, factor: float) -> None:
        super().__init__()
        self.net, self.factor = net, factor

    def forward(self, tok: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.net(tok, cat) * self.factor


def mix(clouds: list[np.ndarray], n: int = N) -> np.ndarray:
    """Point-mix ensemble: an equal share of every model's points, n in total.

    Under squared Chamfer a mix of models that err differently beats each of them:
    on the holdout, original + new-loss + deep decoder scored 400 mm^2 against 457
    for the best single model, mostly on the oversized objects (2450 vs 3059).
    """
    k = len(clouds)
    return np.concatenate([c[m::k] for m, c in enumerate(clouds)])[:n]


def predict_ensemble(ckpts: list[Path]) -> None:
    """Millimetre checkpoints only (train --target mm); writes submissions/ensemble."""
    dev = device()
    paths = rows("val")
    if not paths:
        raise SystemExit("no eval images")
    tokens: dict[str, list[torch.Tensor]] = {}
    per_model = []
    for ckpt in ckpts:
        net, blob = load_net(ckpt, dev)
        if blob.get("unit"):
            raise SystemExit(f"{ckpt} predicts a unit shape; the ensemble needs millimetre models")
        name = blob.get("encoder", "dinov2b")
        if name not in tokens:
            enc = encoder(dev, name)
            with torch.inference_mode():
                tokens[name] = [
                    image_tokens(enc, torch.stack([image_tensor(p, ENCODERS[name]) for p in paths[i : i + 8]]).to(dev)).cpu()
                    for i in range(0, len(paths), 8)
                ]
            del enc
        vocab = blob["vocab"]
        out = []
        with torch.inference_mode():
            for i, tok in zip(range(0, len(paths), 8), tokens[name]):
                cat = torch.tensor([vocab.get(p.parent.parent.name, vocab.get("<unk>", 0)) for p in paths[i : i + 8]], dtype=torch.long, device=dev)
                out.append(net(tok.to(dev), cat).cpu().numpy())
        per_model.append(np.concatenate(out))
        print(f"{ckpt.name}: {len(paths)} clouds", flush=True)
    clouds = {p.parent.name: mix([m[i] for m in per_model]) for i, p in enumerate(paths)}
    out = ROOT / "submissions" / "ensemble" / "preds_geometry.zip"
    pack(out, clouds)
    print(f"{len(clouds)} clouds from {len(ckpts)} models -> {out}")


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
    enc_x = encoder(dev, blob["extra"]) if blob.get("extra") else None
    paths = rows("val")
    if not paths:
        raise SystemExit("no eval images")
    vocab = blob["vocab"]
    unk = vocab.get("<unk>", 0)
    clouds = {}
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            chunk = paths[i : i + 8]
            tok = image_tokens(enc, torch.stack([image_tensor(p, ENCODERS[name]) for p in chunk]).to(dev))
            if blob.get("extra"):  # e.g. depth tokens, appended channel-wise as in training
                tok = torch.cat([tok, image_tokens(enc_x, torch.stack([image_tensor(p, ENCODERS[blob['extra']]) for p in chunk]).to(dev))], -1)
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
    ap.add_argument("cmd", nargs="?", default="train", choices=["cache", "augcache", "dense", "flipcache", "photos", "train", "predict", "evaluate", "check"])
    ap.add_argument("--ckpt", type=Path, nargs="+", default=[CKPT])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="deep", help="train: decoder (atlas = the original)")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="dinov2b", help="cache, train: image tokens")
    ap.add_argument("--target", choices=["mm", "unit"], default="mm", help="train: millimetre clouds, or unit shape for scale.py")
    ap.add_argument("--seed", type=int, default=0, help="train: init and shuffle seed (holdout split is fixed)")
    ap.add_argument("--no-flip-loss", action="store_true", help="train: score the target as canonicalised, without searching its PCA sign flips")
    ap.add_argument("--fourier", type=int, default=0, help="train: low harmonics of the sheet coordinates, plus r^2 (0 = raw uv)")
    ap.add_argument("--resample", action="store_true", help="train: draw fresh target points from the dense cache every step (run: dense)")
    ap.add_argument("--token-drop", type=float, default=0.0, help="train: share of patch tokens hidden from the decoder each step")
    ap.add_argument("--all-queries", action="store_true", help="train: deep decoder on all queries instead of a random half")
    ap.add_argument("--reweight-oversized", action="store_true", help="train: keep oversized objects, capping their loss at a typical object's")
    ap.add_argument("--drop-oversized", action="store_true", help="train: leave oversized objects out of training and of the holdout check")
    ap.add_argument("--drop-path", type=float, default=0.0, help="train: stochastic depth, rising from 0 at the first block to this at the last")
    ap.add_argument("--attn-drop", type=float, default=0.0, help="train: dropout on the cross-attention weights")
    ap.add_argument("--grid-jitter", type=float, default=0.0, help="train: uniform noise on the sheet coordinates, half-width in grid steps")
    ap.add_argument("--repulsion", type=float, default=0.0, help="train: weight of the loss that pushes clumped points apart")
    ap.add_argument("--ema", type=float, default=0.0, help="train: decay of a running average of the weights, saved as <ckpt>_ema.pt")
    ap.add_argument("--deep-sup", type=float, default=0.0, help="train: weight of the extra Chamfer losses on the earlier decoder blocks")
    ap.add_argument("--flip-aug", action="store_true", help="train: use horizontally flipped image tokens for a random half of each batch (run: flipcache)")
    ap.add_argument("--size-hint", action="store_true", help="train: give every query the size of the most similar training objects")
    ap.add_argument("--no-cat", action="store_true", help="train: no category input; the queries carry the image's global features instead")
    ap.add_argument("--extra", default="", help="train: a second token cache appended channel-wise, e.g. dav2s (cache --encoder dav2s first)")
    ap.add_argument("--l1cd", action="store_true", help="train: Chamfer on plain (not squared) distances")
    ap.add_argument("--feat-noise", type=float, default=0.0, help="train: Gaussian noise on the image tokens, as a share of each token's spread (try 0.1)")
    ap.add_argument("--swd", type=float, default=0.0, help="train: weight of a sliced Wasserstein term next to Chamfer (try 0.5)")
    ap.add_argument("--size-head", type=float, default=0.0, help="train: separate size head; value = weight of its log-size NLL (try 0.01)")
    ap.add_argument("--aug-sets", type=int, default=0, help="augcache: views to cache; train: cycle through that many augmented views (augcache first)")
    ap.add_argument("--right-handed", action="store_true", help="train: un-mirror the targets canonicalize mirrored (needs train_mirror.npz)")
    ap.add_argument("--pos-emb", action="store_true", help="train: learned position embedding on the image tokens")
    ap.add_argument("--cat-drop", type=float, default=0.0, help="train: share of objects whose category is replaced by an unknown token")
    ap.add_argument("--stray-weight", type=float, default=1.0, help="train: weight of the predicted-to-true half of the loss")
    ap.add_argument("--miss-weight", type=float, default=1.0, help="train: weight of the true-to-predicted half of the loss")
    ap.add_argument("--folds", type=Path, default=None, help="train: fold file from baselines/tools/make_folds.py (needs --fold); default the fixed 10%% holdout")
    ap.add_argument("--fold", default="", help="train: fold to hold out (0..4), or 'all' to train on every object")
    ap.add_argument("--save-epochs", default="", help="train: also save the EMA weights at these epochs, 'A:B' or 'A:B:step', as <ckpt>_ema_e<E>.pt")
    ap.add_argument("--finetune", type=int, default=0, help="train: also train the last N DINOv3 blocks, reading augmented photos (run: photos)")
    ap.add_argument("--ft-lr", type=float, default=1e-5, help="train: peak learning rate of the fine-tuned DINOv3 blocks")
    ap.add_argument("--shape", choices=["decoder", "retrieve"], default="decoder", help="predict: decoder or nearest training shape")
    ap.add_argument("--retrieval", action="store_true", help="evaluate: also score the nearest-training-shape baseline")
    args = ap.parse_args()
    if args.cmd == "cache":
        cache(args.encoder)
    elif args.cmd == "dense":
        cache_dense()
    elif args.cmd == "flipcache":
        cache_flip(args.encoder)
    elif args.cmd == "photos":
        photo_pack()
    elif args.cmd == "augcache":
        augcache(args.encoder, args.aug_sets, args.seed)
    elif args.cmd == "train":
        train(args.ckpt[0], args.epochs, args.arch, args.encoder, args.target, args.seed, args.drop_oversized, not args.no_flip_loss, args.fourier, args.resample, args.token_drop, args.all_queries, args.reweight_oversized, args.drop_path, args.attn_drop, args.grid_jitter, args.repulsion, args.ema, args.deep_sup, args.flip_aug, args.size_hint, args.cat_drop, (args.stray_weight, args.miss_weight), args.pos_emb, args.right_handed, args.aug_sets, args.size_head, args.swd, args.l1cd, args.feat_noise, args.no_cat, args.extra, args.folds, args.fold, args.save_epochs, args.finetune, args.ft_lr)
    elif args.cmd == "predict":
        if len(args.ckpt) > 1:
            predict_ensemble(args.ckpt)
        else:
            predict(args.ckpt[0], args.shape)
    elif args.cmd == "evaluate":
        evaluate(args.ckpt if args.ckpt != [CKPT] or CKPT.exists() else [], args.retrieval)
    else:
        fit_check()
