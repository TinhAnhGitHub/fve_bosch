"""Ordinal (CORAL) freshness head on RF-DETR decoder queries. Monkeypatch for rfdetr 1.12.

class_embed becomes FreshLinear: same Linear (same state_dict keys), plus `fresh`, an MLP on the same
decoder query features hs -> 2 logits, P(grade>0) and P(grade>1). It runs for every decoder layer and every
Group-DETR query group because class_embed does. The loss uses the queries rfdetr already matched to GT
(captured from SetCriterion._get_matched_targets) and grades decoded from the 46-class labels, so the
dataset, matcher, 46-class head and checkpoints stay unchanged. One network, one ONNX graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

GRADE: torch.Tensor | None = None  # label index -> grade 0..2, -1 = no grade
WEIGHT = 1.0
INST: list = []  # patched LWDETR instances
LAST: dict = {}  # last forward: pred_boxes [B,Q,4] cxcywh 0..1, fresh logits [B,Q,2]
_log = {"n": 0, "sum": 0.0}


class FreshLinear(nn.Linear):
    def forward(self, x):
        self.last_fresh = self.fresh(x)
        return super().forward(x)


def _swap(model, keep=None):
    ce = model.class_embed
    if isinstance(ce, FreshLinear):
        return
    new = FreshLinear(ce.in_features, ce.out_features, device=ce.weight.device, dtype=ce.weight.dtype)
    new.load_state_dict(ce.state_dict())
    d = ce.in_features
    new.fresh = keep if keep is not None else nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2)).to(ce.weight.device)
    model.class_embed = new


def ordinal_probs(logits):
    """2 CORAL logits -> 3 grade probs (rows sum to 1)."""
    s = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32))
    s2 = torch.minimum(s[..., 1], s[..., 0])
    p = torch.stack([1 - s[..., 0], s[..., 0] - s2, s2], -1).clamp_min(1e-6)
    return p / p.sum(-1, keepdim=True)


def fresh_loss(logits, grades):
    """CORAL BCE over boxes with a grade (grades -1 are ignored)."""
    ok = grades >= 0
    z, y = logits[ok].float(), grades[ok]
    if z.numel() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(z, torch.stack([y > 0, y > 1], -1).float())


def patch():
    from rfdetr.models import criterion as C
    from rfdetr.models import lwdetr as L

    if getattr(L.LWDETR, "_fresh_patched", False):
        return
    L.LWDETR._fresh_patched = True
    init, reinit, fwd, fexp = L.LWDETR.__init__, L.LWDETR.reinitialize_detection_head, L.LWDETR.forward, L.LWDETR.forward_export

    def __init__(self, *a, **k):
        init(self, *a, **k)
        _swap(self)
        INST.append(self)

    def reinitialize_detection_head(self, num_classes):
        keep = getattr(self.class_embed, "fresh", None)
        reinit(self, num_classes)
        _swap(self, keep)

    def forward(self, *a, **k):
        out = fwd(self, *a, **k)
        f = getattr(self.class_embed, "last_fresh", None)
        if isinstance(out, dict) and f is not None and f.dim() == 4:  # [layers, B, Q, 2]
            out["pred_fresh"] = f[-1]
            for i, aux in enumerate(out.get("aux_outputs") or []):
                aux["pred_fresh"] = f[i]
            LAST["boxes"], LAST["fresh"] = out["pred_boxes"].detach(), f[-1].detach()
        return out

    def forward_export(self, *a, **k):
        out = fexp(self, *a, **k)
        f = self.class_embed.last_fresh
        LAST["boxes"], LAST["fresh"] = out[0].detach(), (f[-1] if f.dim() == 4 else f).detach()
        return out

    L.LWDETR.__init__, L.LWDETR.reinitialize_detection_head = __init__, reinitialize_detection_head
    L.LWDETR.forward, L.LWDETR.forward_export = forward, forward_export

    gmt, cfwd = C.SetCriterion._get_matched_targets, C.SetCriterion.forward

    def _get_matched_targets(self, targets, indices):
        self.__dict__.setdefault("_fresh_idx", []).append(indices)
        return gmt(self, targets, indices)

    def cforward(self, outputs, targets, *a, **k):
        self._fresh_idx = []
        losses = cfwd(self, outputs, targets, *a, **k)
        layers = [outputs] + list(outputs.get("aux_outputs") or [])
        if GRADE is None or "pred_fresh" not in outputs or len(self._fresh_idx) < len(layers):
            return losses
        g = GRADE.to(outputs["pred_fresh"].device)
        for i, lay in enumerate(layers):
            idx = self._fresh_idx[i]
            src = self._get_src_permutation_idx(idx)
            lab = torch.cat([t["labels"][j.to(t["labels"].device)] for t, (_, j) in zip(targets, idx)])
            key = "loss_fresh" + ("" if i == 0 else f"_{i - 1}")
            losses[key] = fresh_loss(lay["pred_fresh"][src], g[lab])
            self.weight_dict.setdefault(key, WEIGHT)
        if self.training:
            _log["n"] += 1
            _log["sum"] += float(losses["loss_fresh"].detach())
            if _log["n"] == 1:
                print("fresh: first batch grades", torch.bincount(g[lab] + 1, minlength=4).tolist(), "(none, g0, g1, g2)", flush=True)
            if _log["n"] % 100 == 0:
                print(f"fresh: step {_log['n']} mean loss_fresh {_log["sum"] / 100:.4f}", flush=True)
                _log["sum"] = 0.0
        return losses

    C.SetCriterion._get_matched_targets, C.SetCriterion.forward = _get_matched_targets, cforward


def grade_table(dataset_dir, names):
    """label index -> grade, using rfdetr's own category-id -> label mapping."""
    import json
    from pathlib import Path

    from rfdetr.datasets.coco import _train_split_cat2label

    cats = json.loads((Path(dataset_dir) / "train" / "_annotations.coco.json").read_text())["categories"]
    cat2label = _train_split_cat2label(Path(dataset_dir)) or {c["id"]: c["id"] for c in cats}
    t = torch.full((max(cat2label.values()) + 2,), -1, dtype=torch.long)
    for c in cats:
        n = c["name"]
        if n in names and n[-3:-1] == "_g":
            t[cat2label[c["id"]]] = int(n[-1])
    return t


def _selftest():
    p = ordinal_probs(torch.tensor([[9.0, 9.0], [-9.0, -9.0], [9.0, -9.0], [0.0, 2.0]]))
    assert torch.allclose(p.sum(-1), torch.ones(4)) and p[0].argmax() == 2 and p[1].argmax() == 0 and p[2].argmax() == 1
    z = torch.zeros(3, 2, requires_grad=True)
    loss = fresh_loss(z, torch.tensor([-1, 0, 2]))
    loss.backward()
    assert abs(float(loss) - 0.6931) < 1e-3 and z.grad[0].abs().sum() == 0
    assert float(fresh_loss(z, torch.tensor([-1, -1, -1]))) == 0.0
    lin = nn.Module()
    lin.class_embed = nn.Linear(8, 5)
    w = lin.class_embed.weight.clone()
    _swap(lin)
    out = lin.class_embed(torch.randn(6, 3, 8))
    assert out.shape == (6, 3, 5) and lin.class_embed.last_fresh.shape == (6, 3, 2) and torch.equal(lin.class_embed.weight, w)
    assert set(lin.state_dict()) >= {"class_embed.weight", "class_embed.bias", "class_embed.fresh.0.weight"}
    print("fresh_head ok")


if __name__ == "__main__":
    _selftest()
