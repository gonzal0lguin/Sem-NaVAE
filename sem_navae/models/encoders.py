"""Prior p(z|x) and posterior q(z|x, y) encoders (paper Sec. III-A1/2)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class FeatureAttention(nn.Module):
    """Self-attention over the per-modality feature streams, mean-pooled.

    Input ``(B, S, feats_dim)``, output ``(B, hidden_dim)``.
    """

    def __init__(self, feats_dim: int, hidden_dim: int):
        super().__init__()
        self.key = nn.Linear(feats_dim, hidden_dim, bias=False)
        self.query = nn.Linear(feats_dim, hidden_dim, bias=False)
        self.value = nn.Linear(feats_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5
        self.ln_in = nn.LayerNorm(feats_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(feats)
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)
        fused = (attn @ v).mean(dim=1)
        return self.ln_out(self.out(fused))


class PriorEncoder(nn.Module):
    """Fuses LiDAR, past-trajectory and goal features into the context h_x and p(z|x).

    Returns ``h_x`` ``(B, context_dim)`` and ``(mu, logvar)`` ``(B, latent_dim)``.
    """

    NUM_STREAMS = 3  # lidar, past trajectory, goal

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        feats_dim: int,
        hidden_dim: int,
        use_attention: bool = True,
        dropout: float = 0.1,
        logvar_clamp: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ):
        super().__init__()
        self.use_attention = use_attention
        self.logvar_clamp = logvar_clamp

        self.ln_traj = nn.LayerNorm(feats_dim)
        self.ln_lidar = nn.LayerNorm(feats_dim)
        self.ln_goal = nn.LayerNorm(feats_dim)

        if use_attention:
            self.fuse = FeatureAttention(feats_dim, hidden_dim)
        else:
            n = self.NUM_STREAMS
            self.post_concat_ln = nn.LayerNorm(feats_dim * n)
            self.fuse = nn.Sequential(
                nn.Linear(feats_dim * n, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.fuse_ln = nn.LayerNorm(hidden_dim)

        self.pre_latent_ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, traj_feats, lidar_feats, goal_feats):
        o = self.ln_traj(traj_feats)
        p = self.ln_lidar(lidar_feats)
        g = self.ln_goal(goal_feats)

        if self.use_attention:
            h = self.fuse(torch.stack([p, o, g], dim=1))
        else:
            h = self.fuse_ln(self.fuse(self.post_concat_ln(torch.cat([p, o, g], dim=1))))

        h_lat = self.pre_latent_ln(h)
        mu = self.fc_mu(h_lat)
        logvar = self.fc_logvar(h_lat)
        if self.logvar_clamp is not None:
            logvar = torch.clamp(logvar, *self.logvar_clamp)

        return self.context_proj(h), (mu, logvar)


class PosteriorEncoder(nn.Module):
    """q(z | x, y): a bidirectional LSTM over the GT velocities, initialised from h_x."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        context_dim: int,
        state_dim: int = 2,
        dropout_mlp: float = 0.3,
        dropout_rnn: float = 0.25,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_rnn = dropout_rnn
        self.dropout_mlp = dropout_mlp

        self.init_h = nn.Linear(context_dim, hidden_dim)
        self.init_c = nn.Linear(context_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(4 * hidden_dim, mlp_dim)  # (fwd, bwd) x (h, c)
        self.fc2 = nn.Linear(mlp_dim, latent_dim * 2)

    def forward(self, h_x: torch.Tensor, gt_traj: torch.Tensor, train: bool = False):
        """
        Args:
            h_x: ``(B, context_dim)`` context.
            gt_traj: ``(B, N_w, 2)`` ground-truth velocities.
        """
        h0 = self.init_h(h_x)
        c0 = self.init_c(h_x)
        state = (
            torch.stack([h0, torch.zeros_like(h0)], dim=0),
            torch.stack([c0, torch.zeros_like(c0)], dim=0),
        )
        _, state = self.lstm(gt_traj.permute(1, 0, 2), state)

        h = torch.cat(state, dim=0).permute(1, 0, 2)
        h = F.dropout(h, p=self.dropout_rnn, training=train)
        stats = self.fc1(h.reshape(-1, 4 * self.hidden_dim))
        stats = F.dropout(torch.tanh(stats), p=self.dropout_mlp, training=train)
        stats = self.fc2(stats)
        return stats[:, : self.latent_dim], stats[:, self.latent_dim :]
