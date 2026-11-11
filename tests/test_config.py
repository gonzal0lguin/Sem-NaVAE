import pytest

from sem_navae.config import Config, HeatmapSpec, load_config, save_config


def test_defaults_follow_paper():
    cfg = Config()
    assert cfg.model.latent_dim == 512
    assert cfg.model.feature_dim == 256
    assert cfg.model.lidar_num_scans == 3 and cfg.model.lidar_max_points == 2560
    assert cfg.model.num_waypoints == 12
    assert cfg.model.max_accel == 0.5
    assert cfg.loss.collision_weight == 10.0
    assert cfg.training.lr == 1e-3


def test_heatmap_spec_size():
    assert HeatmapSpec(size_m=18.0, resolution=0.2).size_px == 90


def test_yaml_roundtrip_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    save_config(Config(), path)
    cfg = load_config(path, ["model.latent_dim=64", "model.heatmap.resolution=0.1", "name=x"])
    assert cfg.model.latent_dim == 64
    assert cfg.model.heatmap.resolution == 0.1
    assert isinstance(cfg.model.heatmap, HeatmapSpec)
    assert cfg.name == "x"


def test_unknown_keys_rejected(tmp_path):
    with pytest.raises(KeyError, match="model"):
        load_config(None, ["model.latnt_dim=3"])


def test_shipped_config_loads():
    cfg = load_config("configs/train/navae.yaml")
    assert cfg == Config()
