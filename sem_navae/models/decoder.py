"""Stochastic autoregressive velocity decoder p(y | x, z) (paper Sec. III-A3)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal


class StochasticDecoder(nn.Module):
    """GRU decoder that outputs a diagonal Gaussian over the N_w future velocities.

    When the next input is not teacher-forced, it is drawn by rejection sampling: a batch of
    proposals from the current step's Gaussian is filtered by speed, acceleration, heading and
    heatmap constraints, and the first valid proposal is kept (the mean if none is valid).

    Returns ``Independent(Normal(mu, std), 2)`` with ``mu``/``std`` of shape ``(B, N_w, 2)``.
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        steps: int,
        heatmap_size_px: int,
        use_heatmap: bool = True,
        use_map_features: bool = True,
        out_dim: int = 2,
        num_proposals: int = 20000,
        min_speed: float = 0.1,
        max_speed: float = 2.0,
        max_accel: float = 0.9,
        max_heading_change: float = torch.pi / 4,
        map_resolution: float = 0.2,
        map_origin_px: float = 150.0,
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
        self.map_resolution = map_resolution
        self.map_origin_px = map_origin_px

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
            half = heatmap_size_px // 2
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

        pred_tm2 = None
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
                pred_tm2, pred_tm1 = pred_tm1.clone(), pred.clone()
                preds.append(pred)
            elif self.use_heatmap:
                proposals = Independent(Normal(mu, std), 2).rsample((self.num_proposals,))
                pred = self.constrained_sample(proposals, pred_tm1, heatmap, mu, preds, pred_tm2)
                pred_tm2, pred_tm1 = pred_tm1.clone(), pred.clone()
                preds.append(pred)
            else:
                # TODO(T7): without the heatmap, pred_tm1/preds are not updated.
                pred = Independent(Normal(mu, std), 2).rsample()

            if self.use_map_features:
                x = torch.cat([h_x, z, pred, map_feat], dim=-1)
            else:
                x = torch.cat([h_x, z, pred], dim=-1)

        mus = torch.stack(mus, dim=0).permute(1, 0, 2)
        stds = torch.stack(stds, dim=0).permute(1, 0, 2)
        return Independent(Normal(mus, stds), 2)

    def constrained_sample(self, proposals, pred_tm1, heatmap, mu, preds, pred_tm2=None):
        """Pick the first proposal ``(K, B, 2)`` that satisfies every constraint.

        Assumes a 1 s step between waypoints.
        """
        speed = proposals.norm(dim=-1)
        speed_mask = (speed < self.max_speed) & (speed > self.min_speed)

        if pred_tm1 is not None and pred_tm2 is not None:
            # TODO(T7): compares velocity *differences*, not headings.
            prev_dv = pred_tm1 - pred_tm2
            curr_dv = proposals - pred_tm1.unsqueeze(0)
            cos = F.cosine_similarity(prev_dv.unsqueeze(0), curr_dv, dim=-1).clamp(-1, 1)
            angle_mask = torch.acos(cos) <= self.max_heading_change
        else:
            angle_mask = torch.ones_like(speed_mask, dtype=torch.bool)

        accel = torch.norm(proposals - pred_tm1.unsqueeze(0), dim=-1)
        accel_mask = accel < self.max_accel

        if len(preds) > 1:
            position = torch.stack(preds, dim=0)[1:].cumsum(dim=0)[-1].unsqueeze(0)
        else:
            position = torch.zeros_like(proposals)
        xy = position + proposals
        # TODO(T6): pixel mapping is not derived from the heatmap spec.
        px = (xy / self.map_resolution + self.map_origin_px).long().clip(0, heatmap.shape[-1] - 1)
        K, B, _ = px.shape
        batch_idx = torch.arange(B, device=heatmap.device).unsqueeze(0).repeat(K, 1)
        heatmap_mask = heatmap[batch_idx, px[..., 1], px[..., 0]] > 0.0

        valid = heatmap_mask & speed_mask & accel_mask & angle_mask
        valid_any = valid.any(dim=0)
        first_valid = valid.float().argmax(dim=0)
        best = proposals[first_valid, torch.arange(B, device=proposals.device)]
        return torch.where(valid_any.unsqueeze(1), best, mu)
