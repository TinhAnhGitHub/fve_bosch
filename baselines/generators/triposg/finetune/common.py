"""Shared contracts for experimental TripoSG food adaptation."""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

UPSTREAM_COMMIT = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
MODEL_REVISION = "2c1c516"
SCHEMA = 1


def upstream(path):
    root = Path(path).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    sys.path.insert(0, str(root))
    for module in ("models.autoencoders.autoencoder_kl_triposg",
                   "models.transformers.triposg_transformer", "schedulers.scheduling_rectified_flow"):
        importlib.import_module("triposg." + module)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prepared_image(path):
    """Same CPU preprocessing in cache/predict. No implicit segmentation downloads.

    RGBA: crop alpha foreground, white composite, 10% square padding.
    RGB: full image, white square padding. Masked inputs are recommended.
    """
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        if "A" in source.getbands():
            bounds = rgba.getchannel("A").getbbox()
            if bounds is None:
                raise ValueError(f"Empty alpha: {path}")
            rgba = rgba.crop(bounds)
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        rgb = white.convert("RGB")
    side = max(rgb.size)
    canvas = Image.new("RGB", (side + 2 * round(0.1 * side),) * 2, "white")
    canvas.paste(rgb, ((canvas.width - rgb.width) // 2, (canvas.height - rgb.height) // 2))
    return canvas


def normalized_surface(mesh, count, seed, unit_mm, pose):
    """Return XYZ+normals within decoder bounds, physical surface RMS separately."""
    import trimesh
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[faces], dtype=np.float64)
    center = points.mean(0)
    points = points - center
    radius_mm = float(np.sqrt(np.square(points).sum(1).mean()) * unit_mm)
    rotation = np.eye(3)
    if pose == "pca":
        _, _, vt = np.linalg.svd(points, full_matrices=False)
        rotation = vt.T
        signs = np.where(np.mean((points @ rotation) ** 3, axis=0) < 0, -1.0, 1.0)
        rotation = rotation * signs
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
    elif pose != "mesh":
        raise ValueError(pose)
    points, normals = points @ rotation, normals @ rotation
    extent = float(np.abs((np.asarray(mesh.vertices) - center) @ rotation).max())
    if not np.isfinite(radius_mm) or radius_mm <= 0 or extent <= 0:
        raise ValueError("Degenerate/nonfinite mesh")
    surface = np.concatenate([points * (0.95 / extent), normals], -1).astype(np.float32)
    if not np.isfinite(surface).all():
        raise ValueError("Nonfinite surface")
    return surface, radius_mm


def flow_batch(clean, noise, sigma, num_timesteps=1000):
    """Released scheduler: sigma=1 noise; adds (sigma-sigma_next)*velocity.

    Hence the velocity target is clean-noise, and model time is sigma*1000.
    """
    s = sigma[:, None, None]
    return (1 - s) * clean + s * noise, clean - noise, sigma * num_timesteps


def pooled(features):
    return torch.cat([features[:, 0], features[:, 1:].mean(1)], -1).float()


class SizeHead(nn.Module):
    def __init__(self, width=1024, hidden=256, bias=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, hidden),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, features):
        return self.net(pooled(features)).squeeze(-1)


def configure_transformer(model, config):
    if config["mode"] == "full":
        model.requires_grad_(True)
    elif config["mode"] == "lora":
        from peft import LoraConfig
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                                     lora_dropout=config["lora_dropout"], bias="none",
                                     target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        if not any("lora_" in n and p.requires_grad for n, p in model.named_parameters()):
            raise RuntimeError("No adapters installed")
        if any("lora_" not in n and p.requires_grad for n, p in model.named_parameters()):
            raise RuntimeError("Base weights unexpectedly trainable")
    else:
        raise ValueError(config["mode"])
    # Upstream forward implements non-reentrant torch checkpointing directly.
    model.gradient_checkpointing = config["gradient_checkpointing"]
    return model


def model_state(model, mode):
    if mode == "lora":
        from peft.utils import get_peft_model_state_dict
        state = get_peft_model_state_dict(model)
    else:
        state = model.state_dict()
    return {k: v.detach().cpu() for k, v in state.items()}


def restore_model(model, state, mode):
    if mode == "lora":
        from peft.utils import set_peft_model_state_dict
        result = set_peft_model_state_dict(model, state)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        missing = [k for k in result.missing_keys if "lora_" in k]
        if missing:
            raise ValueError(f"{len(missing)} LoRA weights not restored, e.g. {missing[:3]}")
    else:
        model.load_state_dict(state, strict=True)


def metric_points(points, radius_mm):
    points = np.asarray(points, np.float64)
    points = points - points.mean(0)
    radius = np.sqrt(np.square(points).sum(1).mean())
    if not np.isfinite(points).all() or not np.isfinite(radius_mm) or radius_mm <= 0 or radius <= 0:
        raise ValueError("Invalid generated cloud or radius")
    return (points / radius * radius_mm).astype(np.float32)
