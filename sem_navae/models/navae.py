"""NaVAE: conditional VAE that generates N_w-waypoint trajectories (paper Sec. III-A)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.distributions import Independent, Normal

from sem_navae.config import ModelConfig
from sem_navae.keys import Keys
from sem_navae.models.decoder import StochasticDecoder
from sem_navae.models.encoders import PosteriorEncoder
from sem_navae.models.heatmap import traversability_heatmap
from sem_navae.models.perception import Perception


def independent_normal(mu: torch.Tensor, std: torch.Tensor, event_dims: int = 1):
    return Independent(Normal(mu, std), event_dims)


def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q || p) between diagonal Gaussians, summed over the last dimension."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        (logvar_p - logvar_q).sum(-1)
        + ((var_q + (mu_q - mu_p) ** 2) / var_p).sum(-1)
        - mu_q.size(-1)
    )


def logmeanexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """``log(mean(exp(x)))`` along ``dim``, computed stably."""
    m = x.max(dim=dim, keepdim=True).values
    lme = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True)) - math.log(x.size(dim))
    return lme if keepdim else lme.squeeze(dim)


class NaVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_waypoints = cfg.num_waypoints
        self.use_heatmap = cfg.use_heatmap
        self.perception = Perception(cfg)
        self.posterior_encoder = PosteriorEncoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.posterior_hidden_dim,
            mlp_dim=cfg.posterior_mlp_dim,
            context_dim=cfg.context_dim,
        )
        self.decoder = StochasticDecoder(
            context_dim=cfg.context_dim,
            latent_dim=cfg.latent_dim,
            steps=cfg.num_waypoints,
            heatmap=cfg.heatmap,
            use_heatmap=cfg.use_heatmap,
            use_map_features=cfg.use_map_features,
            num_proposals=cfg.num_proposals,
            min_speed=cfg.min_speed,
            max_speed=cfg.max_speed,
            max_accel=cfg.max_accel,
            max_heading_change=math.radians(cfg.max_heading_change_deg),
            dt=cfg.dt,
        )

    @property
    def needs_heatmap(self) -> bool:
        return self.cfg.use_heatmap or self.cfg.use_map_features

    def encode(self, batch: dict, train: bool = False):
        return self.perception(
            batch[Keys.past_traj], batch[Keys.lidar], batch[Keys.goal], train=train
        )

    def decode(self, h_x, z, heatmap=None, gt=None, current_state=None):
        return self.decoder(h_x, z, heatmap=heatmap, gt=gt, current_state=current_state)

    def heatmap(
        self, lidar: torch.Tensor, logprobs, labels, sigma: Optional[float] = None
    ) -> torch.Tensor:
        """I_seg (Eq. 2) with ``cfg.heatmap``; ``sigma`` overrides its smoothing (tests)."""
        points = lidar.reshape(lidar.size(0), -1, lidar.size(-1))
        return traversability_heatmap(
            points, logprobs, labels, self.cfg.heatmap, self.cfg.lidar_norm_dist, sigma
        )

    @staticmethod
    def _current_velocity(batch: dict) -> torch.Tensor:
        return batch[Keys.past_traj][:, -1, 2:4]

    def forward(self, batch: dict, train: bool = False) -> dict:
        """Single-GT pass: decode one posterior and one prior sample (used for validation)."""
        h_x, (mu_x, logvar_x), (logprobs, labels) = self.encode(batch, train=train)
        p_dist = independent_normal(mu_x, torch.exp(logvar_x * 0.5))
        ghm = None
        if self.needs_heatmap:
            ghm = self.heatmap(batch[Keys.lidar], logprobs, labels)

        gt_traj = batch[Keys.path][:, :, 2:4]
        mu_y, logvar_y = self.posterior_encoder(h_x, gt_traj, train=train)
        q_dist = independent_normal(mu_y, torch.exp(logvar_y * 0.5))

        current = self._current_velocity(batch)
        y_hat_q = self.decode(
            h_x, q_dist.rsample(), ghm, gt=batch[Keys.path].permute(1, 0, 2), current_state=current
        )
        y_hat = self.decode(h_x, p_dist.rsample(), ghm, current_state=current)

        batch.update(
            {
                Keys.y_hat: y_hat,
                Keys.y_hat_q: y_hat_q,
                Keys.p_dist: p_dist,
                Keys.q_dist: q_dist,
                Keys.heatmap: ghm,
            }
        )
        return batch

    def train_multimodal_forward(
        self,
        batch: dict,
        num_prior_samples: int = 6,
        heatmap_sigma: Optional[float] = None,
    ) -> dict:
        """Training pass with every valid GT per observation (log-mean-exp likelihood, Eq. 4)."""
        h_x, (mu_x, logvar_x), (logprobs, labels) = self.encode(batch, train=True)
        B = h_x.size(0)
        p_latent = independent_normal(mu_x, torch.exp(0.5 * logvar_x))
        ghm = None
        if self.needs_heatmap:
            ghm = self.heatmap(batch[Keys.lidar], logprobs, labels, heatmap_sigma)

        current = self._current_velocity(batch)
        N = num_prior_samples
        prior_lme, post_lme, kls = [], [], []
        prior_means, post_means = [], []

        for b in range(B):
            Yb = torch.stack(batch[Keys.all_paths][b], dim=0)[:, :, 2:4].to(h_x.device)
            M_b = Yb.size(0)

            # ----- prior: N latent samples, each scored against all M_b GTs -----
            z_p = p_latent.rsample((N,))[:, b, :]
            hm_N = ghm[b].unsqueeze(0).expand(N, *ghm[b].shape) if ghm is not None else None
            p_hat = self.decode(
                h_x[b].unsqueeze(0).expand(N, -1),
                z_p,
                hm_N,
                current_state=current[b].unsqueeze(0).expand(N, -1),
            )
            loc, scale = p_hat.base_dist.loc, p_hat.base_dist.scale
            p_hat_NM = Independent(
                Normal(
                    loc.unsqueeze(1).expand(N, M_b, -1, -1),
                    scale.unsqueeze(1).expand(N, M_b, -1, -1),
                ),
                2,
            )
            ll_NM = p_hat_NM.log_prob(Yb.unsqueeze(0).expand(N, M_b, -1, -1))
            # TODO(T3): logmeanexp already subtracts log(M_b).
            lme_over_M = logmeanexp(ll_NM, dim=1) - math.log(M_b)
            prior_lme.append(logmeanexp(lme_over_M, dim=0))
            best = torch.argmax(lme_over_M)
            # TODO(T1): detached and later integrated a second time by the collision loss.
            prior_means.append(torch.cumsum(loc[best], dim=0).detach())

            # ----- posterior: one latent per GT -----
            hx_M = h_x[b].unsqueeze(0).expand(M_b, -1)
            mu_q, logvar_q = self.posterior_encoder(hx_M, Yb, train=True)
            q_dist = independent_normal(mu_q, torch.exp(0.5 * logvar_q))
            kl_b = kl_diag_gaussians(
                mu_q,
                logvar_q,
                mu_x[b].unsqueeze(0).expand_as(mu_q),
                logvar_x[b].unsqueeze(0).expand_as(logvar_q),
            )
            kls.append(kl_b.mean())

            z_q = q_dist.rsample()
            hm_M = ghm[b : b + 1].expand(M_b, *ghm.shape[1:]) if ghm is not None else None
            q_hat = self.decode(
                hx_M, z_q, hm_M, current_state=current[b].unsqueeze(0).expand(M_b, -1)
            )
            # TODO(T3): Eq. 4 uses a plain expectation over the posterior; also double -log(M_b).
            post_lme.append(logmeanexp(q_hat.log_prob(Yb), dim=0) - math.log(M_b))
            post_means.append(q_hat.base_dist.loc.detach())

        ll_prior = torch.stack(prior_lme).mean()
        ll_posterior = torch.stack(post_lme).mean()
        batch.update(
            {
                Keys.nll: -(ll_prior + ll_posterior),
                Keys.ll_prior: ll_prior,
                Keys.ll_posterior: ll_posterior,
                Keys.kl: torch.stack(kls).mean(),
                Keys.p_dist: p_latent,
                Keys.prior_mean_paths: torch.stack(prior_means),
                Keys.post_mean_paths: post_means,
                Keys.heatmap: ghm,
            }
        )
        return batch

    @torch.no_grad()
    def sample_forward(
        self, batch: dict, num_samples: int, heatmap_sigma: Optional[float] = None
    ) -> dict:
        """Draw ``num_samples`` trajectories per observation from the prior.

        Writes ``batch[Keys.y_hat]``: sampled velocities ``(B, K, N_w, 2)``.
        """
        K = num_samples
        h_x, (mu, logvar), (logprobs, labels) = self.encode(batch)
        B = h_x.size(0)
        ghm = ghm_r = None
        if self.needs_heatmap:
            ghm = self.heatmap(batch[Keys.lidar], logprobs, labels, heatmap_sigma)
            ghm_r = ghm.unsqueeze(1).repeat(1, K, 1, 1).reshape(B * K, *ghm.shape[-2:])

        std = torch.exp(logvar * 0.5)
        z = torch.stack([independent_normal(mu, std).rsample() for _ in range(K)], dim=1)
        z = z.reshape(B * K, -1)
        h_x_r = h_x.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
        current = self._current_velocity(batch).unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)

        y_hat = self.decode(h_x_r, z, ghm_r, current_state=current)
        batch.update(
            {Keys.y_hat: y_hat.rsample().view(B, K, self.num_waypoints, 2), Keys.heatmap: ghm}
        )
        return batch
