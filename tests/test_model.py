"""Model-level checks with the paper's default configuration."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
from legacy_init import synthetic_batch, to_torch_batch  # noqa: E402

from sem_navae.config import Config  # noqa: E402
from sem_navae.keys import Keys  # noqa: E402
from sem_navae.models import NaVAE  # noqa: E402


def small_default_config() -> Config:
    cfg = Config()
    cfg.model.num_proposals = 256
    return cfg


def test_default_config_builds_with_paper_dims():
    cfg = small_default_config()
    assert cfg.model.latent_dim == 512 and cfg.model.context_dim == 256
    model = NaVAE(cfg.model).eval()
    torch.manual_seed(0)
    batch = to_torch_batch(synthetic_batch(), map_key=Keys.semantic_map)
    out = model.sample_forward(batch, num_samples=4)
    assert out[Keys.y_hat].shape == (2, 4, cfg.model.num_waypoints, 2)
    assert out[Keys.heatmap].shape == (2, cfg.model.heatmap.size_px, cfg.model.heatmap.size_px)
