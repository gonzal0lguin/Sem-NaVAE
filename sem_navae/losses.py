"""Training objective (paper Eq. 3-5): reconstruction log-likelihood, KL and collision terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sem_navae.config import DataConfig, LossConfig
from sem_navae.keys import Keys
from sem_navae.models.heatmap import gaussian_filter


class NaVAELoss(nn.Module):
    def __init__(self, cfg: LossConfig, data_cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.map_resolution = data_cfg.semantic_map_resolution
        self.map_half_size_px = data_cfg.semantic_map_half_size_px

    def kl_weight(self, epoch: int) -> float:
        return min(self.cfg.beta_max, epoch / self.cfg.kl_anneal_epochs)

    # ------------------------------------------------------------------ collision
    def collision_loss(self, paths: torch.Tensor, semantic_map: torch.Tensor) -> torch.Tensor:
        """Eq. 5: ``log(1 + sum_p GK(M_sem)(p))`` per path, differentiable in the positions.

        ``M_sem`` (1 = non-traversable) is smoothed with a Gaussian kernel of std
        ``collision_sigma_px`` and read at every waypoint with bilinear interpolation, so the
        gradient pushes waypoints away from obstacles. The ``1 +`` keeps the loss finite and
        zero for collision-free paths (Eq. 5 writes ``log sum``; see docs/paper_to_code.md).

        Map convention (``docs/dataset_format.md``): ``semantic_map[b, i, j]`` with
        ``i = x / res + half`` and ``j = y / res + half``, robot at the centre.

        Args:
            paths: ``(B, N, 2)`` or ``(B, S, N, 2)`` positions [m] in the robot frame.
            semantic_map: ``(B, H, W)`` with ``H = W = 2 * semantic_map_half_size_px``.

        Returns:
            ``(B, S)`` per-path losses.
        """
        if paths.dim() == 3:
            paths = paths.unsqueeze(1)
        B, S, N, _ = paths.shape
        assert semantic_map.shape[1:] == (2 * self.map_half_size_px,) * 2, semantic_map.shape
        smoothed = gaussian_filter((semantic_map == 1).float(), self.cfg.collision_sigma_px)
        # grid_sample (align_corners=False): u in [-1, 1] spans the map edges; u_w indexes
        # columns (j <-> y) and u_h rows (i <-> x).
        u = paths / (self.map_resolution * self.map_half_size_px)
        grid = torch.stack([u[..., 1], u[..., 0]], dim=-1)  # (B, S, N, 2) as (u_w, u_h)
        values = F.grid_sample(
            smoothed.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return torch.log1p(values.squeeze(1).sum(dim=-1))

    # ------------------------------------------------------------------ objective
    def forward(self, out: dict, epoch: int) -> dict:
        """``L = beta * KL(q || p) + L_recon + lambda * L_col`` on the output of ``NaVAE``.

        The same objective is used for training and validation. The KL is KL(q || p) (the
        standard ELBO term; Eq. 3 prints D_KL(p || q), see docs/errata.md). With
        ``free_bits > 0`` each observation's KL is floored at that value (default off).
        """
        kl = out[Keys.kl]
        if self.cfg.free_bits > 0:
            kl = torch.clamp(kl, min=self.cfg.free_bits)
        kl = kl.mean()
        beta = self.kl_weight(epoch)
        col = self.collision_loss(out[Keys.prior_mean_paths], out[Keys.semantic_map])
        col = col.mean() * self.cfg.collision_weight
        loss = out[Keys.nll] + beta * kl + col
        return {
            Keys.loss: loss,
            Keys.nll: out[Keys.nll],
            Keys.ll_prior: out[Keys.ll_prior],
            Keys.ll_posterior: out[Keys.ll_posterior],
            Keys.kl: kl,
            Keys.kl_loss: beta * kl,
            Keys.collision_loss: col,
            Keys.kl_weight: beta,
        }
