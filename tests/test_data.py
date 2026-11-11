"""Deterministic, persisted train/val split (fix T8)."""

import json

import pytest

pytest.importorskip("torch")

from sem_navae.config import Config  # noqa: E402
from sem_navae.data.dataset import load_or_create_split, split_ids  # noqa: E402

IDS = [f"sample_{i:04d}.pkl" for i in range(100)]


def test_split_is_disjoint_complete_and_deterministic():
    a = split_ids(IDS, 0.2, seed=0)
    assert not set(a["train"]) & set(a["val"])
    assert sorted(a["train"] + a["val"]) == IDS and len(a["val"]) == 20
    assert split_ids(list(reversed(IDS)), 0.2, seed=0) == a  # input order does not matter
    assert split_ids(IDS, 0.2, seed=1)["val"] != a["val"]


def test_split_file_is_written_once_and_reused(tmp_path):
    cfg = Config()
    cfg.training.output_dir = str(tmp_path)
    first = load_or_create_split(cfg, IDS)
    path = tmp_path / cfg.name / "split.json"
    assert json.loads(path.read_text()) == first
    cfg.seed = 123  # an existing file wins over the seed
    assert load_or_create_split(cfg, IDS) == first
    with pytest.raises(ValueError):
        load_or_create_split(cfg, IDS[:50])


def test_past_trajectory_is_in_the_robot_frame():
    import numpy as np

    from sem_navae.data.transforms import past_trajectory

    # robot driving along world +y at 1 m/s, heading +y (yaw = 90 deg), sampled at 0.5 s
    stamps = np.arange(5) * 0.5
    positions = np.stack([np.full(5, 3.0), 10.0 + stamps, np.zeros(5)], -1)
    past = past_trajectory(stamps, positions, yaw=np.pi / 2, max_vel=2.0)
    assert past.shape == (4, 4)
    np.testing.assert_allclose(past[:, 0], [-1.5, -1.0, -0.5, 0.0], atol=1e-9)  # behind
    np.testing.assert_allclose(past[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(past[:, 2:], [[0.5, 0.0]] * 4, atol=1e-9)  # forward, /max_vel
