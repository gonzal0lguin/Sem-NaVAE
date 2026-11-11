"""Parity of the port with the legacy NaVAE code (random weights, fixed seeds).

Goldens come from ``tests/golden/generate_from_legacy.py``. Every behaviour-changing fix
regenerates the affected values deliberately in its own commit.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
from legacy_init import (  # noqa: E402
    GOAL,
    PATHS_FOR_GOAL,
    deterministic_init,
    synthetic_batch,
    to_torch_batch,
)

from sem_navae.config import Config, HeatmapSpec  # noqa: E402
from sem_navae.keys import Keys  # noqa: E402
from sem_navae.losses import NaVAELoss  # noqa: E402
from sem_navae.models import NaVAE  # noqa: E402
from sem_navae.utils.geometry import get_path_dist_to_goal  # noqa: E402

GOLDEN = np.load(os.path.join(os.path.dirname(__file__), "golden", "legacy_model.npz"))
TOL = dict(rtol=1e-4, atol=1e-5)


def legacy_config() -> Config:
    cfg = Config()
    m = cfg.model
    m.latent_dim = 256
    m.heatmap = HeatmapSpec(size_m=30.0, resolution=0.1, sigma_px=5.0)
    m.max_accel = 0.9
    cfg.loss.collision_weight = 1.0
    return cfg


@pytest.fixture(scope="module")
def setup():
    cfg = legacy_config()
    model = NaVAE(cfg.model)
    names = deterministic_init(model)
    model.eval()
    arrays = synthetic_batch()
    return cfg, model, names, (lambda: to_torch_batch(arrays, map_key=Keys.semantic_map))


def test_parameters_match(setup):
    _, _, names, _ = setup
    assert names == sorted(GOLDEN["param_names"].tolist())


def test_perception_and_heatmap(setup):
    _, model, _, batch = setup
    with torch.no_grad():
        b = batch()
        hx, (mu, logvar), (logp, labels) = model.encode(b)
        ghm = model.heatmap(b[Keys.lidar], logp, labels, sigma=5.0)
    np.testing.assert_allclose(hx.numpy(), GOLDEN["hx"], **TOL)
    np.testing.assert_allclose(mu.numpy(), GOLDEN["mu"], **TOL)
    np.testing.assert_allclose(logvar.numpy(), GOLDEN["logvar"], **TOL)
    np.testing.assert_allclose(logp.numpy(), GOLDEN["logprobs"], **TOL)
    np.testing.assert_allclose(ghm.numpy(), GOLDEN["heatmap_sigma5"], **TOL)


def test_forward(setup):
    _, model, _, batch = setup
    torch.manual_seed(1)
    with torch.no_grad():
        out = model(batch())
    np.testing.assert_allclose(
        out[Keys.y_hat].base_dist.loc.numpy(), GOLDEN["fwd_y_hat_loc"], **TOL
    )
    np.testing.assert_allclose(
        out[Keys.y_hat].base_dist.scale.numpy(), GOLDEN["fwd_y_hat_scale"], **TOL
    )
    np.testing.assert_allclose(
        out[Keys.y_hat_q].base_dist.loc.numpy(), GOLDEN["fwd_y_hat_q_loc"], **TOL
    )


def test_train_forward_and_loss(setup):
    cfg, model, _, batch = setup
    torch.manual_seed(2)
    out = model.train_multimodal_forward(batch(), num_prior_samples=6)
    losses = NaVAELoss(cfg.loss, cfg.data).train_loss(out, epoch=30)
    for key, golden in [
        (Keys.nll, "train_nll"),
        (Keys.ll_prior, "train_ll_prior"),
        (Keys.ll_posterior, "train_ll_posterior"),
        (Keys.kl, "train_kl"),
    ]:
        np.testing.assert_allclose(out[key].item(), GOLDEN[golden], **TOL)
    np.testing.assert_allclose(
        out[Keys.prior_mean_paths].numpy(), GOLDEN["train_prior_mean_paths"], **TOL
    )
    np.testing.assert_allclose(losses[Keys.loss].item(), GOLDEN["train_loss"], **TOL)
    np.testing.assert_allclose(losses[Keys.collision_loss].item(), GOLDEN["train_collision"], **TOL)


def test_sample_forward(setup):
    _, model, _, batch = setup
    torch.manual_seed(3)
    out = model.sample_forward(batch(), num_samples=3)
    np.testing.assert_allclose(out[Keys.y_hat].numpy(), GOLDEN["sample_y_hat"], **TOL)


def test_goal_cost():
    np.testing.assert_allclose(get_path_dist_to_goal(PATHS_FOR_GOAL, GOAL), GOLDEN["goal_cost"])
