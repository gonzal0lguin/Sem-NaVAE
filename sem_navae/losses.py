"""Training objective (paper Eq. 3-5): reconstruction log-likelihood, KL and collision terms."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import kl_divergence

from sem_navae.config import DataConfig, LossConfig
from sem_navae.keys import Keys


class NaVAELoss(nn.Module):
    def __init__(self, cfg: LossConfig, data_cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.map_resolution = data_cfg.semantic_map_resolution
        self.map_origin_px = data_cfg.semantic_map_half_size_px

    def kl_weight(self, epoch: int) -> float:
        return min(self.cfg.beta_max, epoch / self.cfg.kl_anneal_epochs)

    # ------------------------------------------------------------------ collision
    def _path_collision(self, path_px: torch.Tensor, obstacle_px: torch.Tensor) -> torch.Tensor:
        """Mean ``arctanh(1 - d)`` over waypoints, d = distance [m] to the nearest obstacle."""
        path = path_px.view(1, -1, path_px.size(-1))
        obstacles = obstacle_px.view(-1, 1, obstacle_px.size(-1)).to(torch.float)
        d = torch.min(torch.norm(obstacles - path, dim=-1), dim=0)[0] * self.map_resolution
        return torch.arctanh(1 - torch.clamp(d, 0.0001, 0.999)).mean()

    def collision_loss(self, paths: torch.Tensor, semantic_map: torch.Tensor) -> torch.Tensor:
        """Per-path collision penalty.

        TODO(T2): replace with Eq. 5 (log sum GK(M_sem(p))), vectorised.

        Args:
            paths: ``(B, N, 2)`` or ``(B, S, N, 2)`` positions [m] in the robot frame.
            semantic_map: ``(B, H, W)`` map, 1 = non-traversable.
        """
        if paths.dim() == 3:
            paths = paths.unsqueeze(1)
        B = paths.size(0)
        assert semantic_map.size(0) == B and semantic_map.size(1) == semantic_map.size(2)
        paths_px = (paths / self.map_resolution + self.map_origin_px).to(torch.int)
        losses = []
        for b in range(B):
            obstacles = torch.stack(torch.where(semantic_map[b] == 1), dim=1)
            for path in paths_px[b]:
                losses.append(self._path_collision(path, obstacles))
        return torch.stack(losses)

    # ------------------------------------------------------------------ objectives
    def train_loss(self, out: dict, epoch: int) -> dict:
        """Multi-GT objective on the output of ``NaVAE.train_multimodal_forward``."""
        # TODO(T1): prior_mean_paths are already positions (and detached); integrated again here.
        paths = out[Keys.prior_mean_paths].cumsum(dim=1)
        col = self.collision_loss(paths, out[Keys.semantic_map]).mean() * self.cfg.collision_weight
        beta = self.kl_weight(epoch)
        loss = out[Keys.nll] + beta * out[Keys.kl] + col
        return {
            Keys.loss: loss,
            Keys.kl_loss: beta * out[Keys.kl],
            Keys.ll_prior: out[Keys.nll],
            Keys.collision_loss: col,
            Keys.kl_weight: beta,
        }

    def eval_loss(self, out: dict, epoch: int) -> dict:
        """Single-GT objective on the output of ``NaVAE.forward``."""
        y_gt = out[Keys.path][:, :, 2:4]
        # TODO(T4): this is KL(p || q) with free bits, the training path uses KL(q || p).
        kl = kl_divergence(out[Keys.p_dist], out[Keys.q_dist])
        if self.cfg.free_bits > 0:
            kl = torch.clamp(kl, min=self.cfg.free_bits)
        kl = kl.mean()
        ll_prior = out[Keys.y_hat].log_prob(y_gt).mean()
        ll_posterior = out[Keys.y_hat_q].log_prob(y_gt).mean()
        col = self.collision_loss(out[Keys.y_hat].mean.cumsum(dim=1), out[Keys.semantic_map])
        col = col.mean() * self.cfg.collision_weight
        beta = self.kl_weight(epoch)
        loss = -(ll_posterior + ll_prior - beta * kl) + col
        return {
            Keys.loss: loss,
            Keys.kl_loss: beta * kl,
            Keys.loglikelihood: ll_posterior + ll_prior,
            Keys.ll_prior: ll_prior,
            Keys.ll_posterior: ll_posterior,
            Keys.collision_loss: col,
            Keys.kl_weight: beta,
        }
