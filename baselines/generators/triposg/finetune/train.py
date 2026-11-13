"""Single-GPU cached-latent training. Phase shape: LoRA/full. Phase size: MLP."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # Initialize before Diffusers/Lightning on torch 2.5.
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (SCHEMA, SizeHead, configure_transformer, digest, flow_batch,
                    model_state, read_json, upstream, write_json)


class CachedObjects(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / f"{row['id']}.npz", allow_pickle=False) as z:
            data = {k: torch.from_numpy(np.asarray(z[k], np.float32).copy())
                    for k in ("features", "mean", "logvar", "radius_mm")}
        if any(not torch.isfinite(v).all() for v in data.values()) or data["radius_mm"] <= 0:
            raise ValueError(f"Nonfinite cache/radius for {row['id']}")
        return data


def read_config(path):
    config = read_json(path)
    config.setdefault("size_loss", "log_mse")
    # stop_epoch < epochs: the learning-rate schedule spans `epochs`, training stops after `stop_epoch`
    # (reproduces an early-stopped run's exact trajectory without a validation set)
    config.setdefault("stop_epoch", config.get("epochs"))
    required = {"mode", "epochs", "batch_size", "accumulation", "lr", "precision",
                "seed", "gradient_checkpointing", "weight_decay", "warmup_fraction",
                "clip_grad", "condition_dropout", "rank", "alpha", "lora_dropout",
                "posterior_sample", "patience", "sigma_sampling", "hidden", "size_loss", "stop_epoch"}
    if set(config) != required:
        raise ValueError(f"Config keys: missing {required - set(config)}, unknown {set(config) - required}")
    if config["mode"] not in {"lora", "full", "size"}:
        raise ValueError("mode must be lora, full, or size")
    if config["size_loss"] not in {"log_mse", "radius_mse"}:
        raise ValueError("size_loss must be log_mse or radius_mse")
    for key in ("epochs", "batch_size", "accumulation", "rank", "alpha", "hidden", "patience", "stop_epoch"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["stop_epoch"] > config["epochs"]:
        raise ValueError("stop_epoch cannot exceed epochs")
    if config["precision"] not in {"bf16", "fp16", "fp32"} or config["sigma_sampling"] not in {"uniform", "logit_normal"}:
        raise ValueError("Invalid precision or sigma_sampling")
    for key in ("condition_dropout", "lora_dropout", "warmup_fraction"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["lr"] <= 0 or config["clip_grad"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    return config


def objective(network, batch, config, phase, generator=None, training=True):
    """Model-specific math only. Trainer owns autocast, backward and stepping."""
    features = batch["features"]
    if phase == "size":
        target = batch["radius_mm"].log()
        pred = network(features)
        radius_mse = (pred.float().exp() - batch["radius_mm"]).square().mean()
        # Fixed units change loss magnitude, not the radius-MSE optimum.
        loss = radius_mse / 40.0**2 if config["size_loss"] == "radius_mse" else nn.functional.mse_loss(pred.float(), target.float())
        return loss, radius_mse.detach()
    clean = batch["mean"]
    device = clean.device
    if training and config["posterior_sample"]:
        clean = clean + (0.5 * batch["logvar"]).exp() * torch.randn(clean.shape, device=device, generator=generator)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    if config["sigma_sampling"] == "logit_normal":
        sigma = torch.randn(len(clean), device=device, generator=generator).sigmoid()
    else:
        sigma = torch.rand(len(clean), device=device, generator=generator)
    noisy, target, timestep = flow_batch(clean, noise, sigma)
    if training:
        drop = torch.rand(len(clean), device=device, generator=generator) < config["condition_dropout"]
        features = features.masked_fill(drop[:, None, None], 0)
    dtype = network.proj_in.weight.dtype
    prediction = network(noisy.to(dtype), timestep,
                         encoder_hidden_states=features.to(dtype), return_dict=False)[0]
    loss = nn.functional.mse_loss(prediction.float(), target.float())
    return loss, loss.detach()


class FoodAdaptation(L.LightningModule):
    """Two independent tasks sharing a cache contract; no manual optimization."""
    def __init__(self, network, config, metadata, cache_hash):
        super().__init__()
        self.network = network
        self.config = config
        self.metadata = metadata
        self.cache_hash = cache_hash
        self.phase = "size" if config["mode"] == "size" else "shape"
        # The large pretrained network is not serialized as a hyperparameter.
        self.save_hyperparameters({"config": config, "metadata": metadata, "cache_hash": cache_hash})

    def training_step(self, batch, batch_idx):
        loss, _ = objective(self.network, batch, self.config, self.phase)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 batch_size=len(batch["features"]), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Independent fixed noise per validation batch; repeated every epoch.
        generator = torch.Generator(device=self.device).manual_seed(1729 + batch_idx)
        loss, diagnostic = objective(self.network, batch, self.config, self.phase,
                                     generator=generator, training=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 batch_size=len(batch["features"]), prog_bar=True)
        if self.phase == "size":
            self.log("val_radius_mse_mm2", diagnostic, on_epoch=True,
                     batch_size=len(batch["features"]))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                    lr=self.config["lr"], weight_decay=self.config["weight_decay"],
                    betas=(0.9, 0.999), eps=1e-8, foreach=False)
        # the trainer runs stop_epoch epochs; the schedule is laid out over all `epochs`
        total = max(1, int(self.trainer.estimated_stepping_batches * self.config["epochs"] / self.config["stop_epoch"]))
        warmup = max(1, round(total * self.config["warmup_fraction"]))
        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step-warmup)/max(1, total-warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def on_train_epoch_end(self):
        # --no-val: no score picks an epoch, so every epoch's adapter is exported (epoch numbers from 0)
        if getattr(self, "export_dir", None) is not None and not self.trainer.sanity_checking:
            self.export(self.export_dir / f"epoch{self.current_epoch}.pt", self.current_epoch, float("nan"))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["adaptation"] = dict(schema=SCHEMA, config=self.config,
                    metadata=self.metadata, cache_hash=self.cache_hash, phase=self.phase)

    def on_load_checkpoint(self, checkpoint):
        info = checkpoint["adaptation"]
        if info["config"] != self.config or info["cache_hash"] != self.cache_hash or info["phase"] != self.phase:
            raise ValueError("Resume config/cache/phase mismatch")

    def export(self, path, epoch, val_loss):
        state = self.network.state_dict() if self.phase == "size" else model_state(self.network, self.config["mode"])
        torch.save(dict(schema=SCHEMA, config=self.config, metadata=self.metadata,
                   cache_hash=self.cache_hash, phase=self.phase, epoch=epoch,
                   val_loss=val_loss, state=state), path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--memory-test", action="store_true", help="One effective batch, optimizer allocation, no checkpoints")
    p.add_argument("--no-val", action="store_true", help="Final model on a manifest with no holdout (prepare.py manifest --all-train): "
                   "runs exactly stop_epoch epochs, exports epoch<k>.pt after every epoch and last.pt")
    args = p.parse_args()
    config = read_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Use an authorized Colab/Kaggle CUDA machine for model training")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU lacks bf16; use the fp16 T4 configuration")
    L.seed_everything(config["seed"], workers=True)
    meta_path = args.cache / "metadata.json"
    meta, cache_hash = read_json(meta_path), digest(meta_path)
    if meta["schema"] != SCHEMA:
        raise ValueError("Unsupported cache schema")
    tr = [r for r in meta["rows"] if r["split"] == "train"]
    va = [r for r in meta["rows"] if r["split"] == "val"]
    if args.no_val:
        if not tr or va:
            raise ValueError("--no-val needs a cache whose rows are all training (prepare.py manifest --all-train)")
    elif not tr or not va or {r["group"] for r in tr} & {r["group"] for r in va}:
        raise ValueError("Empty or leaking train/validation split")
    train_data = CachedObjects(args.cache, tr)
    val_data = CachedObjects(args.cache, va) if va else None
    first = train_data[0]
    if first["features"].shape != (257, 1024) or first["mean"].shape != (meta["tokens"], 64):
        raise ValueError("Unexpected cache tensor shapes")
    if config["mode"] == "size":
        bias = float(np.mean([float(train_data[i]["radius_mm"].log()) for i in range(len(tr))]))
        network = SizeHead(hidden=config["hidden"], bias=bias)
    else:
        upstream(args.upstream)
        from triposg.models.transformers import TripoSGDiTModel
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
        network = TripoSGDiTModel.from_pretrained(meta["model"], subfolder="transformer",
                    revision=meta["revision"], torch_dtype=dtype if config["mode"] == "lora" else torch.float32)
        configure_transformer(network, config)
        for param in network.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    task = FoodAdaptation(network, config, meta, cache_hash)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Use --resume or an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run.json", dict(config=config, cache_hash=cache_hash,
               train_objects=len(tr), val_objects=len(va),
               trainable_parameters=sum(v.numel() for v in network.parameters() if v.requires_grad)))
    # Drop the final short training microbatch so Lightning's accumulation averages
    # equally sized microbatches. All validation objects are retained.
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=True)
    if not len(train_loader):
        raise ValueError("Training split smaller than batch_size")
    val_loader = None if args.no_val else DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                                                     num_workers=0, pin_memory=True)
    if args.no_val:  # fixed epochs: a resumable last.ckpt plus an exported adapter per epoch, no early stopping
        saver = ModelCheckpoint(dirpath=args.output, filename="latest", monitor=None, save_top_k=1, save_on_train_epoch_end=True)
        callbacks = [saver]
        task.export_dir = args.output
    else:
        saver = ModelCheckpoint(dirpath=args.output, filename="best", monitor="val_loss", mode="min",
                                save_top_k=1, save_last=True, save_on_train_epoch_end=False)
        callbacks = [saver, EarlyStopping(monitor="val_loss", mode="min", patience=config["patience"], check_finite=True)]
    precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[config["precision"]]
    trainer = L.Trainer(accelerator="gpu", devices=1, precision=precision,
                max_epochs=config["stop_epoch"], accumulate_grad_batches=config["accumulation"],
                gradient_clip_val=config["clip_grad"], gradient_clip_algorithm="norm",
                callbacks=[] if args.memory_test else callbacks,
                logger=False if args.memory_test else CSVLogger(args.output, name="logs"),
                enable_checkpointing=not args.memory_test, enable_model_summary=False,
                num_sanity_val_steps=0 if args.memory_test or args.no_val else 2,
                limit_train_batches=min(config["accumulation"], len(train_loader)) if args.memory_test else 1.0,
                limit_val_batches=0 if args.memory_test or args.no_val else 1.0,
                max_steps=1 if args.memory_test else -1, log_every_n_steps=1,
                deterministic="warn")
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(args.resume) if args.resume else None)
    print({"peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
           "peak_reserved_GiB": torch.cuda.max_memory_reserved()/2**30}, flush=True)
    if args.no_val and not args.memory_test:
        task.export(args.output / "last.pt", task.current_epoch - 1, float("nan"))
    elif not args.memory_test and saver.best_model_path:
        saved = torch.load(saver.best_model_path, map_location="cpu", weights_only=False)
        task.load_state_dict(saved["state_dict"])
        task.export(args.output / "best.pt", saved["epoch"], float(saver.best_model_score))


if __name__ == "__main__":
    main()
