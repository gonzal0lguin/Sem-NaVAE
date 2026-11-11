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
