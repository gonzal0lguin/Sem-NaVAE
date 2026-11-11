"""Offline trajectory generation from a trained checkpoint."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from sem_navae.config import Config, config_from_dict
from sem_navae.keys import Keys
from sem_navae.models import NaVAE
from sem_navae.utils.io import get_device, to_device


class Generator:
    """Loads a checkpoint (with its embedded config) and samples K trajectories."""

    def __init__(
        self, checkpoint: Optional[str] = None, cfg: Optional[Config] = None, device: str = "cpu"
    ):
        self.device = get_device(device)
        state = None
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu")
            cfg = config_from_dict(ckpt["config"])
            state = ckpt["model"]
        if cfg is None:
            raise ValueError("Pass a checkpoint or an explicit config (untrained model).")
        self.cfg = cfg
        self.model = NaVAE(cfg.model)
        if state is not None:
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def sample(
        self, observation: dict, num_samples: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample trajectories for a single observation (batch of 1).

        Returns:
            ``(K, N_w, 2)`` positions in the robot frame and the ``(H, W)`` heatmap (or None).
        """
        out = self.model.sample_forward(
            to_device(observation, self.device), num_samples=num_samples
        )
        trajs = torch.cumsum(out[Keys.y_hat].squeeze(0), dim=1) * self.cfg.model.dt
        trajs = trajs.cpu().numpy()
        ghm = out[Keys.heatmap]
        return trajs, (ghm.squeeze(0).cpu().numpy() if ghm is not None else None)
