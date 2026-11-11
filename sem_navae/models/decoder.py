"""Stochastic autoregressive velocity decoder p(y | x, z) (paper Sec. III-A3)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal

from sem_navae.config import HeatmapSpec
from sem_navae.models.heatmap import metric_to_pixel


class StochasticDecoder(nn.Module):
    """GRU decoder that outputs a diagonal Gaussian over the N_w future velocities.

    When the next input is not teacher-forced, it is drawn by rejection sampling: a batch of
    proposals from the current step's Gaussian is filtered by speed, acceleration, heading and
    (optionally) heatmap constraints, and the first valid proposal is kept (the mean if none
    is valid). Velocities are in m/s; waypoints are ``dt`` seconds apart.

    Returns ``Independent(Normal(mu, std), 2)`` with ``mu``/``std`` of shape ``(B, N_w, 2)``.
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        steps: int,
        heatmap: HeatmapSpec,
        use_heatmap: bool = True,
        use_map_features: bool = True,
        out_dim: int = 2,
        num_proposals: int = 20000,
        min_speed: float = 0.1,
        max_speed: float = 2.0,
        max_accel: float = 0.9,
        max_heading_change: float = torch.pi / 4,
        dt: float = 1.0,
    ):
        super().__init__()
        self.steps = steps
        self.out_dim = out_dim
        self.use_heatmap = use_heatmap
        self.use_map_features = use_map_features
        self.num_proposals = num_proposals
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_heading_change = max_heading_change
        self.dt = dt
        self.heatmap_spec = heatmap

        hidden_dim = context_dim
        # z -> initial GRU state; latent_dim and context_dim are independent (paper: 512 / 256).
        self.in_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        in_dim = context_dim + latent_dim + out_dim + out_dim * int(use_map_features)
        self.gru = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_logvar = nn.Linear(hidden_dim, out_dim)

        if use_map_features:
            half = heatmap.size_px // 2
            self.map_enc = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(16 * half * half, 2),
            )

    def forward(
        self,
        h_x: torch.Tensor,
        z: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        gt: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            h_x: ``(B, context_dim)`` context.
            z: ``(B, latent_dim)`` latent sample.
            heatmap: ``(B, H, W)`` traversability heatmap (needed for heatmap / map features).
            gt: ``(N_w, B, 4)`` ground truth [x, y, vx, vy] for teacher forcing, or None.
            current_state: ``(B, 2)`` current velocity used as the first input (zeros if None).
        """
        B = h_x.size(0)
        mu_0 = (
            current_state
            if current_state is not None
            else torch.zeros(B, self.out_dim, device=z.device)
        )
        preds: List[torch.Tensor] = [mu_0.clone()]
        h = self.in_fc(z)

        if self.use_map_features:
            map_feat = self.map_enc(heatmap.unsqueeze(1))
            x = torch.cat((h_x, z, mu_0, map_feat), dim=-1)
        else:
            x = torch.cat((h_x, z, mu_0), dim=-1)

        pred_tm1 = mu_0.clone()
        mus, stds = [], []
        for i in range(self.steps):
            h = self.gru(x, h)
            mu = self.fc_mu(h)
            std = torch.exp(0.5 * self.fc_logvar(h))
            mus.append(mu)
            stds.append(std)

            if i == self.steps - 1:
                break

            if gt is not None:
                pred = gt[:, :, 2:4][i + 1]
            else:
                proposals = Independent(Normal(mu, std), 2).rsample((self.num_proposals,))
                pred = self.constrained_sample(proposals, pred_tm1, heatmap, mu, preds)
            pred_tm1 = pred.clone()
            preds.append(pred)

            if self.use_map_features:
                x = torch.cat([h_x, z, pred, map_feat], dim=-1)
            else:
                x = torch.cat([h_x, z, pred], dim=-1)

        mus = torch.stack(mus, dim=0).permute(1, 0, 2)
        stds = torch.stack(stds, dim=0).permute(1, 0, 2)
        return Independent(Normal(mus, stds), 2)

    def constrained_sample(self, proposals, pred_tm1, heatmap, mu, preds):
        """Pick the first proposal ``(K, B, 2)`` that satisfies every constraint (Sec. III-A3).

        A proposal v_t is valid if ``min_speed < |v_t| < max_speed``, the acceleration
        ``|v_t - v_{t-1}| / dt`` is below ``max_accel``, the heading change (angle between
        v_t and v_{t-1}) is at most ``max_heading_change`` whenever v_{t-1} is not ~zero,
        and (with ``use_heatmap``) the resulting waypoint lands on a traversable heatmap
        cell. Falls back to the mean when no proposal is valid.
        """
        speed = proposals.norm(dim=-1)
        valid = (speed < self.max_speed) & (speed > self.min_speed)

        accel = torch.norm(proposals - pred_tm1.unsqueeze(0), dim=-1) / self.dt
        valid &= accel < self.max_accel

        moving = pred_tm1.norm(dim=-1) > self.min_speed  # heading undefined when ~stopped
        cos = F.cosine_similarity(pred_tm1.unsqueeze(0).expand_as(proposals), proposals, dim=-1)
        heading_ok = torch.acos(cos.clamp(-1, 1)) <= self.max_heading_change
        valid &= heading_ok | ~moving.unsqueeze(0)

        if self.use_heatmap and heatmap is not None:
            position = torch.stack(preds[1:], dim=0).sum(dim=0) * self.dt if len(preds) > 1 else 0
            xy = position + proposals * self.dt
            px = metric_to_pixel(xy, self.heatmap_spec).clip(0, heatmap.shape[-1] - 1)
            K, B, _ = px.shape
            batch_idx = torch.arange(B, device=heatmap.device).unsqueeze(0).expand(K, B)
            valid &= heatmap[batch_idx, px[..., 1], px[..., 0]] > 0.0

        valid_any = valid.any(dim=0)
        first_valid = valid.float().argmax(dim=0)
        best = proposals[first_valid, torch.arange(proposals.size(1), device=proposals.device)]
        return torch.where(valid_any.unsqueeze(1), best, mu)
