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
from legacy_rules import apply_legacy_rules  # noqa: E402

from sem_navae.config import Config, HeatmapSpec  # noqa: E402
from sem_navae.keys import Keys  # noqa: E402
from sem_navae.losses import NaVAELoss  # noqa: E402
from sem_navae.models import NaVAE  # noqa: E402
from sem_navae.utils.geometry import get_path_dist_to_goal  # noqa: E402

GOLDEN = np.load(os.path.join(os.path.dirname(__file__), "golden", "legacy_model.npz"))
TOL = dict(rtol=1e-4, atol=1e-5)
# The training-pass goldens depend on the dropout random stream, which changes between torch
# releases; they are only comparable with the torch version that recorded them.
GOLDEN_TORCH = "2.15"
same_dropout_rng = pytest.mark.skipif(
    not torch.__version__.startswith(GOLDEN_TORCH + "."),
    reason=f"dropout RNG goldens recorded with torch {GOLDEN_TORCH}",
)


def legacy_config() -> Config:
    cfg = Config()
    m = cfg.model
    m.latent_dim = 256
    m.heatmap = HeatmapSpec(size_m=30.0, resolution=0.1, sigma_px=5.0)
    m.max_accel = 0.9
    cfg.loss.collision_weight = 1.0
    return cfg


# The legacy code smoothed the heatmap with sigma 5 (forward), 6 (training) and 8 (sampling)
# and its decoder read the 0.1 m/px map as if it were 0.2 m/px around pixel 150. The
# port has one HeatmapSpec; these overrides reproduce the legacy numbers.
LEGACY_TRAIN_SIGMA, LEGACY_SAMPLE_SIGMA = 6.0, 8.0
LEGACY_DECODER_SPEC = HeatmapSpec(size_m=60.0, resolution=0.2)  # origin pixel 150


@pytest.fixture(scope="module")
def setup():
    cfg = legacy_config()
    model = NaVAE(cfg.model)
    names = deterministic_init(model)
    model.decoder.heatmap_spec = LEGACY_DECODER_SPEC
    apply_legacy_rules(model)
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
    np.testing.assert_allclose(ghm.numpy()[:, ::3, ::3], GOLDEN["heatmap_sigma5"], **TOL)


@same_dropout_rng
def test_train_forward_and_loss(setup):
    cfg, model, _, batch = setup
    torch.manual_seed(2)
    # The goldens were recorded with the legacy model in eval() but its training pass forcing
    # the functional (RNN) dropout on; reproduce that mix: train mode, nn.Dropout layers off.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    out = model(batch(), num_prior_samples=6, heatmap_sigma=LEGACY_TRAIN_SIGMA)
    model.eval()
    losses = NaVAELoss(cfg.loss, cfg.data)(out, epoch=30)
    np.testing.assert_allclose(out[Keys.kl].mean().item(), GOLDEN["train_kl"], **TOL)
    # The legacy prior term subtracted log(M_b) twice; the port's is larger by the
    # batch mean of log(M_b) (M = 3 and 2 ground truths in the synthetic batch).
    log_m = np.mean(np.log([3, 2]))
    np.testing.assert_allclose(out[Keys.ll_prior].item(), GOLDEN["train_ll_prior"] + log_m, **TOL)
    # The posterior term is a plain expectation (Eq. 4), not a log-mean-exp: no golden.
    np.testing.assert_allclose(
        out[Keys.nll].item(), -(out[Keys.ll_prior] + out[Keys.ll_posterior]).item(), **TOL
    )
    # The legacy collision term saw only the best prior mean, detached; the port keeps
    # every prior mean (B, N, N_w, 2) in the graph. The legacy path must be one of them.
    paths = out[Keys.prior_mean_paths].detach().numpy()
    for b, legacy_path in enumerate(GOLDEN["train_prior_mean_paths"]):
        assert np.abs(paths[b] - legacy_path).max(axis=(1, 2)).min() < 1e-4
    # Eq. 5 replaces the legacy arctanh distance penalty (no golden value).
    expected = out[Keys.nll] + losses[Keys.kl_loss] + losses[Keys.collision_loss]
    np.testing.assert_allclose(losses[Keys.loss].item(), expected.item(), **TOL)


def test_sample_forward(setup):
    _, model, _, batch = setup
    torch.manual_seed(3)
    out = model.sample_forward(batch(), num_samples=3, heatmap_sigma=LEGACY_SAMPLE_SIGMA)
    np.testing.assert_allclose(out[Keys.y_hat].numpy(), GOLDEN["sample_y_hat"], **TOL)


def test_goal_cost():
    # The heading now uses the direction towards the goal, so |theta| = pi - |legacy|.
    new = get_path_dist_to_goal(PATHS_FOR_GOAL, GOAL)
    legacy = GOLDEN["goal_cost"]
    np.testing.assert_allclose(new[:, 0], legacy[:, 0])
    np.testing.assert_allclose(np.abs(new[:, 1]), np.pi - np.abs(legacy[:, 1]))
