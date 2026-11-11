"""Offline CPU contracts: tiny random upstream modules, no pretrained weights."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
# Import before Diffusers to avoid Torch 2.5's lazy-import cycle on some installs.
import torch._dynamo  # noqa: F401
import trimesh

from common import (configure_transformer, flow_batch, metric_points, model_state,
                    normalized_surface, prepared_image, restore_model, upstream)


class Contracts(unittest.TestCase):
    def tiny(self):
        from triposg.models.transformers import TripoSGDiTModel
        return TripoSGDiTModel(num_attention_heads=4, width=32, in_channels=8,
                              num_layers=3, cross_attention_dim=16)

    def config(self, mode):
        return dict(mode=mode, rank=2, alpha=2, lora_dropout=0.0, gradient_checkpointing=True)

    def test_scheduler_direction_and_time(self):
        from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler
        clean, noise = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
        for s in (0.0, 0.5, 1.0):
            sigma = torch.full((2,), s)
            noisy, target, time = flow_batch(clean, noise, sigma)
            self.assertTrue(torch.allclose(noisy, (1-s)*clean+s*noise))
            self.assertTrue(torch.allclose(time, sigma*1000))
        scheduler = RectifiedFlowScheduler()
        scheduler.set_timesteps(1)
        result = scheduler.step(clean-noise, scheduler.timesteps[0], noise).prev_sample
        self.assertTrue(torch.allclose(result, clean, atol=1e-6))

    def test_lora_gradients_and_export_roundtrip(self):
        torch.manual_seed(3)
        base = self.tiny()
        original = {n: p.detach().clone() for n, p in base.named_parameters()}
        clone = self.tiny()
        clone.load_state_dict(base.state_dict())
        net = configure_transformer(base, self.config("lora"))
        sample, features, time = torch.randn(2, 8, 8), torch.randn(2, 5, 16), torch.tensor([200., 800.])
        net.train()
        output = net(sample, time, encoder_hidden_states=features, return_dict=False)[0]
        self.assertEqual(output.shape, sample.shape)
        output.square().mean().backward()
        grads = [p.grad for n, p in net.named_parameters() if "lora_" in n]
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in grads))
        self.assertTrue(all(p.grad is None for n, p in net.named_parameters() if "lora_" not in n))
        torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.1).step()
        # Check original base tensors, including PEFT's renamed base_layer paths.
        for name, param in net.named_parameters():
            if "lora_" not in name:
                self.assertTrue(torch.equal(param, original[name.replace(".base_layer", "")]))
        state = model_state(net, "lora")
        self.assertTrue(state and all("lora_" in n for n in state))
        configure_transformer(clone, self.config("lora"))
        restore_model(clone, state, "lora")
        net.eval(); clone.eval()
        self.assertTrue(torch.allclose(net(sample, time, encoder_hidden_states=features, return_dict=False)[0],
                                      clone(sample, time, encoder_hidden_states=features, return_dict=False)[0]))

    def test_full_gradients(self):
        net = configure_transformer(self.tiny(), self.config("full"))
        net.train()
        out = net(torch.randn(1, 8, 8), torch.tensor([500.]),
                  encoder_hidden_states=torch.randn(1, 5, 16), return_dict=False)[0]
        out.square().mean().backward()
        self.assertTrue(all(p.requires_grad for p in net.parameters()))
        self.assertTrue(all(p.grad is not None for p in net.parameters()))
        clone = self.tiny()
        restore_model(clone, model_state(net, "full"), "full")

    def test_vae_xyz_normal_contract(self):
        from triposg.models.autoencoders import TripoSGVAEModel
        import triposg.models.autoencoders.autoencoder_kl_triposg as module
        from unittest.mock import patch
        # Stub ONLY FPS selection to avoid compiling torch-cluster for CPU tests.
        # Actual encoding/decoding is the real upstream implementation.
        with patch.object(module, "fps", lambda xyz, batch, **kw: torch.arange(0, len(xyz), 4), create=True):
            vae = TripoSGVAEModel(latent_channels=8, num_attention_heads=4,
                                  width_encoder=32, width_decoder=32,
                                  num_layers_encoder=1, num_layers_decoder=1).eval()
            posterior = vae.encode(torch.randn(1, 64, 6), num_tokens=8, seed=42).latent_dist
            self.assertEqual(posterior.mean.shape, (1, 8, 8))
            self.assertEqual(vae.decode(posterior.mode(), sampled_points=torch.randn(1, 16, 3)).sample.shape, (1, 16, 1))

    def test_geometry_scale_and_normals(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=20)
        surface, radius = normalized_surface(mesh, 512, 42, 1., "pca")
        surface2, radius2 = normalized_surface(trimesh.Trimesh(mesh.vertices*3, mesh.faces, process=False), 512, 42, 1., "pca")
        self.assertEqual(surface.shape, (512, 6))
        self.assertLessEqual(np.abs(surface[:, :3]).max(), .951)
        self.assertTrue(np.allclose(np.linalg.norm(surface[:, 3:], axis=1), 1))
        self.assertAlmostEqual(radius2/radius, 3., places=5)
        self.assertTrue(np.allclose(surface, surface2, atol=1e-5))
        cloud = metric_points(surface[:, :3], 20.)
        self.assertAlmostEqual(float(np.sqrt(np.square(cloud-cloud.mean(0)).sum(1).mean())), 20., places=4)

    def test_rigid_scorer_does_not_remove_scale(self):
        from evaluate import aligned_cd
        rng = np.random.default_rng(42)
        truth = rng.normal(size=(100, 3)) * [20, 10, 3]
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        self.assertLess(aligned_cd(truth@rotation+[50,20,10], truth), 1e-10)
        self.assertGreater(aligned_cd(truth*2, truth), 1.)

    def test_duplicate_meshes_never_cross_split(self):
        from prepare import manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, body in enumerate((b"same", b"same", b"other", b"third")):
                directory = root / "train" / "fruit" / "berry" / str(i)
                directory.mkdir(parents=True)
                (directory / "mesh.ply").write_bytes(body)
                (directory / "rgb_industrial.png").write_bytes(b"image")
            out = root / "manifest.json"
            manifest(SimpleNamespace(data=root, groups=None, seed=42, val_fraction=.5, output=out))
            rows = json.loads(out.read_text())["rows"]
            self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_all_configs_validate(self):
        from train import read_config
        for path in (Path(__file__).parent / "configs").glob("*.json"):
            read_config(path)

    def test_lightning_fit_checkpoint_and_resume(self):
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader
        from train import FoodAdaptation, read_config
        config = read_config(Path(__file__).parent / "configs/lora_24gb.json")
        config.update(rank=2, alpha=2, lora_dropout=0., precision="fp32", epochs=2)
        batches = [dict(features=torch.randn(5, 16), mean=torch.randn(8, 8),
                        logvar=torch.zeros(8, 8), radius_mm=torch.tensor(20.)) for _ in range(4)]
        loader = DataLoader(batches, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            saver = ModelCheckpoint(dirpath=path, monitor="val_loss", save_last=True)
            task = FoodAdaptation(configure_transformer(self.tiny(), config), config, {}, "test-cache")
            trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                        accumulate_grad_batches=2, callbacks=[saver], logger=False,
                        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
            self.assertEqual(trainer.global_step, 2)
            task.export(path / "adapter.pt", 0, 1.)
            artifact = torch.load(path / "adapter.pt", weights_only=False)
            self.assertTrue(all("lora_" in k for k in artifact["state"]))
            restored = FoodAdaptation(configure_transformer(self.tiny(), config), config, {}, "test-cache")
            resumed = L.Trainer(accelerator="cpu", devices=1, max_epochs=2,
                        accumulate_grad_batches=2, callbacks=[ModelCheckpoint(dirpath=path, monitor="val_loss", save_last=True)],
                        logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
            resumed.fit(restored, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=saver.last_model_path)
            self.assertEqual(resumed.global_step, 4)
            bad = dict(config, lr=config["lr"]*2)
            incompatible = FoodAdaptation(self.tiny(), bad, {}, "test-cache")
            with self.assertRaises(ValueError):
                incompatible.on_load_checkpoint(torch.load(saver.last_model_path, weights_only=False))

    def test_size_loss_ablation_preserves_model_and_units(self):
        from common import SizeHead
        from train import objective
        network = SizeHead(width=16, hidden=8, bias=float(np.log(20.))).eval()
        batch = dict(features=torch.randn(2, 5, 16), radius_mm=torch.tensor([10., 30.]))
        log_loss, radius_error = objective(network, batch, {"size_loss": "log_mse"}, "size")
        linear_loss, same_error = objective(network, batch, {"size_loss": "radius_mse"}, "size")
        self.assertAlmostEqual(radius_error.item(), 100., places=3)
        self.assertTrue(torch.equal(radius_error, same_error))
        self.assertAlmostEqual(linear_loss.item(), 100./1600, places=5)
        self.assertGreater(log_loss.item(), 0)
        linear_loss.backward()
        self.assertIsNotNone(network.net[-1].weight.grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    args, remaining = parser.parse_known_args()
    upstream(args.upstream)
    unittest.main(argv=[__file__, *remaining])
