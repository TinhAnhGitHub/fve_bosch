"""Shared helpers for point-mix ensembles and strict scoring (ens.py, oof.py).

cd: squared Chamfer between two clouds. align_to: best of the 8 axis sign flips (mirrors allowed)
about the centre, then rigid ICP onto a reference. strict: the grader's view, centroid then rigid ICP
onto the truth, no flip search.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import kabsch, nearest  # noqa: E402

SIGNS = torch.tensor([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=torch.float32)


@torch.no_grad()
def cd(q: torch.Tensor, t: torch.Tensor) -> float:
    return float(nearest(q, t)[0].mean() + nearest(t, q)[0].mean())


@torch.no_grad()
def align_to(p: torch.Tensor, ref: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Best sign flip (mirrors allowed) of p about its centre, then rigid ICP onto ref."""
    pc = p - p.mean(0)
    best, best_d = None, float("inf")
    for s in SIGNS.to(p.device):
        q = pc * s + ref.mean(0)
        d = cd(q[::8], ref[::8])
        if d < best_d:
            best, best_d = q, d
    q = best
    for _ in range(iters):
        _, j = nearest(q, ref)
        R, tr = kabsch(q, ref[j])
        q = q @ R.T + tr
    return q


@torch.no_grad()
def strict(pred: torch.Tensor, t: torch.Tensor, iters: int = 10) -> float:
    q = pred - pred.mean(0) + t.mean(0)
    for _ in range(iters):
        _, j = nearest(q, t)
        R, tr = kabsch(q, t[j])
        q = q @ R.T + tr
    return cd(q, t)
