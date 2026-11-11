"""Displacement metrics over K sampled trajectories."""

from __future__ import annotations

from typing import Dict

import torch


def min_ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Best-of-K average / final displacement error.

    Args:
        pred: ``(B, K, N, 2)`` predicted positions.
        gt: ``(B, N, 2)`` ground-truth positions.

    Returns:
        ``min_ade`` and ``min_fde``, averaged over the batch.
    """
    err = torch.norm(pred - gt.unsqueeze(1), dim=-1)  # (B, K, N)
    return {
        "min_ade": err.mean(dim=-1).min(dim=1)[0].mean(),
        "min_fde": err[..., -1].min(dim=1)[0].mean(),
    }
