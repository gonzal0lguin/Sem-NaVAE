"""Traversability labels and a short PointNet pre-training run on synthetic data."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sem_navae.config import Config  # noqa: E402
from sem_navae.data.synthetic import make_synthetic_dataset  # noqa: E402
from sem_navae.models.perception import LidarEncoder  # noqa: E402
from sem_navae.training.pointnet_pretrain import (  # noqa: E402
    IGNORE_INDEX,
    PointNetDataset,
    PointNetPretrainConfig,
    label_points,
    pretrain_pointnet,
)


def test_label_points_axis_convention_and_filters():
    local_map = np.zeros((20, 20), dtype=np.uint8)  # 0.5 m/px, robot at pixel (10, 10)
    local_map[14, 10] = 1  # x in [2, 2.5), y in [0, 0.5)
    points = np.array(
        [
            [2.2, 0.1, 0.0],  # on the obstacle cell
            [0.1, 2.2, 0.0],  # transposed position: free
            [2.2, 0.1, 3.0],  # above max_z: ignored
            [0.0, 0.0, 0.0],  # zero padding: ignored
            [4.9, -4.9, 0.0],  # inside the map (+-5 m), free
            [5.1, 0.0, 0.0],  # outside the map: ignored
        ]
    )
    labels = label_points(points, local_map, resolution=0.5, max_z=1.0)
    assert labels.tolist() == [1, 0, IGNORE_INDEX, IGNORE_INDEX, 0, IGNORE_INDEX]


def test_max_z_is_relative_to_the_ground():
    pcfg = PointNetPretrainConfig(robot_height=0.6, sensor_height=0.7)
    assert pcfg.max_z == pytest.approx(1.5 * 0.6 - 0.7)


def _cfg(tmp_path) -> Config:
    cfg = Config(name="pn", device="cpu")
    cfg.data.root = str(tmp_path / "data")
    cfg.data.num_workers = 0
    cfg.data.val_fraction = 0.25
    cfg.training.output_dir = str(tmp_path / "out")
    cfg.model.lidar_max_points = 256
    return cfg


def test_synthetic_samples_have_both_classes(tmp_path):
    cfg = _cfg(tmp_path)
    make_synthetic_dataset(cfg.data.root, num_samples=4, cfg=cfg)
    ds = PointNetDataset(cfg, PointNetPretrainConfig(robot_height=1.0, sensor_height=0.4), True)
    points, labels = ds.load(0)
    assert points.shape == (cfg.model.lidar_num_scans * cfg.model.lidar_max_points, 4)
    assert {0, 1} <= set(labels.tolist())


def test_pretrained_weights_load_into_navae(tmp_path):
    cfg = _cfg(tmp_path)
    make_synthetic_dataset(cfg.data.root, num_samples=8, cfg=cfg)
    pcfg = PointNetPretrainConfig(robot_height=1.0, sensor_height=0.4, epochs=1, batch_size=2)
    path = pretrain_pointnet(cfg, pcfg)
    cfg.model.pointnet_weights = path
    encoder = LidarEncoder(cfg.model)
    saved = torch.load(path, map_location="cpu")["model_state_dict"]
    for name, value in encoder.encoder.state_dict().items():
        assert torch.equal(value, saved[name]), name
