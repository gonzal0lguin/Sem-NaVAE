"""Legacy behaviour that the port fixed, re-implemented so the parity tests can still pin
everything else to the legacy numbers. Only used by ``tests/test_golden.py``."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from sem_navae.models.heatmap import metric_to_pixel


def legacy_constrained_sample(self, proposals, pred_tm1, heatmap, mu, preds):
    """Legacy rejection rule (T7): the 'heading' check compares consecutive velocity
    *differences* and is skipped on the first step; acceleration ignores dt."""
    pred_tm2 = preds[-2] if len(preds) > 1 else None
    speed = proposals.norm(dim=-1)
    valid = (speed < self.max_speed) & (speed > self.min_speed)
    if pred_tm2 is not None:
        prev_dv = (pred_tm1 - pred_tm2).unsqueeze(0)
        cos = F.cosine_similarity(prev_dv, proposals - pred_tm1.unsqueeze(0), dim=-1)
        valid &= torch.acos(cos.clamp(-1, 1)) <= self.max_heading_change
    valid &= torch.norm(proposals - pred_tm1.unsqueeze(0), dim=-1) < self.max_accel
    if len(preds) > 1:
        position = torch.stack(preds, 0)[1:].cumsum(0)[-1].unsqueeze(0)
    else:
        position = torch.zeros_like(proposals)
    px = metric_to_pixel(position + proposals, self.heatmap_spec).clip(0, heatmap.shape[-1] - 1)
    K, B, _ = px.shape
    valid &= heatmap[torch.arange(B).unsqueeze(0).repeat(K, 1), px[..., 1], px[..., 0]] > 0
    best = proposals[valid.float().argmax(0), torch.arange(B)]
    return torch.where(valid.any(0).unsqueeze(1), best, mu)


def legacy_current_velocity(batch):
    """Legacy decoder initial state (T8b): the *normalised* velocity, used as if in m/s."""
    return batch["previous_path"][:, -1, 2:4]


def apply_legacy_rules(model) -> None:
    model.decoder.constrained_sample = types.MethodType(legacy_constrained_sample, model.decoder)
    model._current_velocity = legacy_current_velocity
