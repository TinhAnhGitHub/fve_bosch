# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "trimesh"]
# ///
"""PSGN baseline for Food 3D Reconstruction.

Fan, Su, Guibas, CVPR 2017: an image encoder predicts a point set and the
loss is Chamfer distance. Targets are canonical millimetre clouds from
`baselines/geom.py`, because the training meshes have no pose and the grader
keeps scale.

  uv run baselines/baseline.py train
  uv run baselines/baseline.py predict
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import MIN_POINTS, canonicalize, pack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "baselines" / "out" / "psgn.pt"
IMAGE = 128


class PointSet(nn.Module):
    """Small conv encoder and a linear point decoder. Output is (B, N, 3) mm."""

    def __init__(self, n: int = MIN_POINTS) -> None:
        super().__init__()
        self.n = n
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=4, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dec = nn.Linear(128, n * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x).flatten(1)
        return self.dec(z).view(-1, self.n, 3)


def cd_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    d2 = torch.cdist(pred, gt).square()
    return d2.min(2).values.mean() + d2.min(1).values.mean()


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
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0


def target_points(mesh_path: Path, n: int) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return canonicalize(pts)


def fit_check() -> None:
    """One cloud, no dataset: the Chamfer loss must fall, and the zip must open."""
    torch.manual_seed(0)
    net = PointSet(n=128)
    target = torch.zeros(128, 3)
    target[:, 0] = torch.linspace(0, 40, 128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.rand(1, 3, IMAGE, IMAGE)
    first = float(cd_l2(net(x)[0], target).detach())
    for _ in range(40):
        loss = cd_l2(net(x)[0], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = float(cd_l2(net(x)[0], target).detach())
    assert last < first * 0.5, (first, last)
    out = Path("/tmp/f3dr_fit.zip")
    pack(out, {"toy": net(x)[0].detach().numpy()}, min_points=128)
    print(f"fit check ok  {first:.1f} -> {last:.1f}")


def train() -> None:
    net = PointSet().cuda() if torch.cuda.is_available() else PointSet()
    device = next(net.parameters()).device
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    paths = rows("train")
    if not paths:
        raise SystemExit(f"no training images under {data_root()}")
    for epoch in range(1, 21):
        total = 0.0
        for img in paths:
            mesh = img.parent / "mesh.ply"
            gt = torch.tensor(target_points(mesh, net.n), dtype=torch.float32, device=device)
            pred = net(image_tensor(img).unsqueeze(0).to(device))[0]
            loss = cd_l2(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {epoch}  cd_l2 {total / len(paths):.2f}", flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "n": net.n}, CKPT)
    print(f"saved {CKPT}")


def predict() -> None:
    if not CKPT.exists():
        raise SystemExit(f"missing {CKPT}. Run: uv run baselines/baseline.py train")
    blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = PointSet(blob["n"]).eval()
    net.load_state_dict(blob["state"])
    clouds = {}
    with torch.inference_mode():
        for img in rows("val"):
            pred = net(image_tensor(img).unsqueeze(0))[0].numpy()
            clouds[img.parent.name] = pred
    if not clouds:
        raise SystemExit("no eval images")
    out = ROOT / "submissions" / "psgn" / "preds_geometry.zip"
    pack(out, clouds)
    print(f"{len(clouds)} clouds -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "predict":
        predict()
    elif cmd == "check":
        fit_check()
    else:
        raise SystemExit("use train, predict, or check")
