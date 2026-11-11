"""Record legacy NaVAE outputs for the parity tests (run once; not part of the test suite).

Usage:
    python tests/golden/generate_from_legacy.py /path/to/original_code \
        /path/to/Pointnet_Pointnet2_Pytorch

Writes ``tests/golden/legacy_model.npz``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from legacy_init import (  # noqa: E402
    GOAL,
    PATHS_FOR_GOAL,
    deterministic_init,
    synthetic_batch,
    to_torch_batch,
)


def main(legacy_root: str, pointnet_root: str) -> None:
    sys.path[:0] = [legacy_root, pointnet_root]
    from navae.config.training_config import BaseConfig, CollisionLossType
    from navae.models.navae_model import NaVAE
    from navae.training.loss import LossEvaluation
    from navae.utils.funcs import get_path_dist_to_goal

    cfg = BaseConfig
    cfg.model.lidar_pth = None
    cfg.model.use_heatmap = True
    cfg.model.use_map_feats = True
    cfg.loss.collision_type = CollisionLossType.local_dis

    model = NaVAE(cfg.model)
    names = deterministic_init(model, legacy=True)
    model.eval()

    arrays = synthetic_batch()
    out = {}
    out["param_names"] = np.array(names)

    def batch():
        b = to_torch_batch(arrays, map_key="png")
        b["vel"] = torch.zeros(2, 20, 2)  # read by the legacy forward but unused (joint velocity)
        return b

    with torch.no_grad():
        b = batch()
        hx, (mu, logvar), (logp, labels) = model.encode(
            b["previous_path"], b["lidar"], b["vel"], b["goal"]
        )
        out.update(hx=hx.numpy(), mu=mu.numpy(), logvar=logvar.numpy(), logprobs=logp.numpy())
        ghm = model.create_heatmap(b["lidar"].reshape(-1, 7680, 4), logp, labels, sigma=5)
        out["heatmap_sigma5"] = ghm.numpy()[:, ::3, ::3]  # subsampled to keep the file small

        torch.manual_seed(1)
        f = model(batch())
        out.update(
            fwd_y_hat_loc=f["y_hat"].base_dist.loc.numpy(),
            fwd_y_hat_scale=f["y_hat"].base_dist.scale.numpy(),
            fwd_y_hat_q_loc=f["y_hat_q"].base_dist.loc.numpy(),
        )

    torch.manual_seed(2)
    t = model.train_multimodal_forward(batch())
    losses = LossEvaluation(cfg.loss)(t, epoch=30)
    out.update(
        train_nll=t["nll"].item(),
        train_ll_prior=t["ll_prior"].item(),
        train_ll_posterior=t["ll_posterior"].item(),
        train_kl=t["kl"].item(),
        train_prior_mean_paths=t["prior_mean_paths"].numpy(),
        train_loss=losses["loss"].item(),
        train_collision=losses["collision_loss_mean"].item(),
    )

    torch.manual_seed(3)
    s = model.sample_forward(batch(), N=3)
    out["sample_y_hat"] = s["y_hat"].numpy()
    out["goal_cost"] = get_path_dist_to_goal(PATHS_FOR_GOAL, GOAL)

    np.savez_compressed(os.path.join(HERE, "legacy_model.npz"), **out)
    print("wrote", os.path.join(HERE, "legacy_model.npz"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
