"""Observation encoders: LiDAR (PointNet), past trajectory, goal (paper Sec. III-A1)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from sem_navae.config import ModelConfig
from sem_navae.models.encoders import PriorEncoder
from sem_navae.models.pointnet import PointNetSemSeg

log = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class TrajectoryEncoder(nn.Module):
    """LSTM over the past trajectory ``(N_v, B, 4)`` -> ``(B, out_dim)``."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        state_dim: int = 4,
        dropout_rnn: float = 0.25,
        dropout_mlp: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rnn = dropout_rnn
        self.dropout_mlp = dropout_mlp
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h = F.dropout(h, p=self.dropout_rnn, training=train)
        out = self.fc(h.permute(1, 0, 2).reshape(-1, self.hidden_dim))
        return F.dropout(F.leaky_relu(out), p=self.dropout_mlp, training=train)


class LidarEncoder(nn.Module):
    """PointNet traversability segmentation + an MLP over its 64x64 feature transform."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.freeze = cfg.freeze_pointnet
        self.encoder = PointNetSemSeg(num_classes=2, channel=4)
        self.mlp_head = nn.Sequential(
            nn.Linear(64 * 64, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, cfg.feature_dim),
            nn.LeakyReLU(),
        )
        if cfg.pointnet_weights:
            self.load_pointnet(cfg.pointnet_weights)
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_pointnet(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.encoder.load_state_dict(state, strict=True)
        log.info("Loaded PointNet weights from %s", path)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:  # a frozen PointNet always runs with its BatchNorm statistics
            self.encoder.eval()
        return self

    def forward(self, lidar: torch.Tensor):
        """
        Args:
            lidar: ``(B, N_l, N_p, 4)`` or ``(B, N, 4)`` normalised points.

        Returns:
            Per-point log-probabilities ``(B, N, 2)`` and features ``(B, feature_dim)``.
        """
        if lidar.dim() == 4:
            lidar = lidar.view(lidar.size(0), -1, lidar.size(-1))
        B = lidar.size(0)
        lidar = lidar.permute(0, 2, 1)
        if self.freeze:
            with torch.no_grad():
                seg_logprobs, feats = self.encoder(lidar)
        else:
            seg_logprobs, feats = self.encoder(lidar)
        return seg_logprobs, self.mlp_head(feats.view(B, -1))


class Perception(nn.Module):
    """Encodes the observation x = (LiDAR, past trajectory, goal) into h_x and p(z|x)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.latent_dim != cfg.context_dim:
            # TODO(T5): the decoder initialises its hidden state from z with a
            # (context_dim x context_dim) layer, so both sizes must match until a
            # z -> hidden projection is added.
            raise ValueError(
                f"latent_dim ({cfg.latent_dim}) must equal context_dim ({cfg.context_dim})"
            )
        self.lidar_enc = LidarEncoder(cfg)
        self.past_traj_enc = TrajectoryEncoder(hidden_dim=64, out_dim=cfg.feature_dim, state_dim=4)
        self.goal_enc = MLP(in_dim=2, hidden_dim=64, out_dim=cfg.feature_dim)
        self.prior_encoder = PriorEncoder(
            latent_dim=cfg.latent_dim,
            context_dim=cfg.context_dim,
            feats_dim=cfg.feature_dim,
            hidden_dim=cfg.prior_hidden_dim,
            use_attention=cfg.use_feature_attention,
        )

    def forward(self, past_traj, lidar, goal, train: bool = False):
        """
        Returns:
            ``h_x``, ``(mu_x, logvar_x)`` and ``(seg_logprobs, seg_labels)``.
        """
        seg_logprobs, lidar_feats = self.lidar_enc(lidar)
        goal_feats = self.goal_enc(goal)
        traj_feats = self.past_traj_enc(past_traj.permute(1, 0, 2), train=train)
        h_x, stats = self.prior_encoder(traj_feats, lidar_feats, goal_feats)
        return h_x, stats, (seg_logprobs, torch.argmax(seg_logprobs, dim=-1))
