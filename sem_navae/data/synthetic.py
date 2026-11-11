"""Tiny synthetic dataset in the training format (``docs/dataset_format.md``).

Meant for smoke tests and CI only: the scenes are random and carry no navigation signal.
Each sample is a robot at the origin of its own frame with a few circular obstacles, a
LiDAR-like cloud (ground points plus obstacle points), a straight-ish past trajectory and
1-3 ground-truth futures that curve towards a goal.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from sem_navae.config import Config


def _future(rng: np.random.Generator, n: int, dt: float) -> np.ndarray:
    speed = rng.uniform(0.6, 1.4)
    yaw_rate = rng.uniform(-0.15, 0.15)
    heading = np.cumsum(np.full(n, yaw_rate * dt)) + rng.uniform(-0.3, 0.3)
    vel = speed * np.stack([np.cos(heading), np.sin(heading)], -1)
    return np.concatenate([np.cumsum(vel * dt, axis=0), vel], -1).astype(np.float32)


def _past(rng: np.random.Generator, n: int, period: float) -> np.ndarray:
    speed = rng.uniform(0.5, 1.5)
    t = -period * np.arange(n)[::-1]
    pos = np.stack([speed * t, np.zeros(n)], -1)
    vel = np.tile([speed, 0.0], (n, 1))
    return (np.concatenate([pos, vel], -1) + rng.normal(0, 0.02, (n, 4))).astype(np.float32)


def make_sample(rng: np.random.Generator, cfg: Config) -> dict:
    m, d = cfg.model, cfg.data
    half_m = d.semantic_map_half_size_px * d.semantic_map_resolution
    obstacles = rng.uniform([2.0, -8.0], [15.0, 8.0], size=(int(rng.integers(2, 6)), 2))
    radii = rng.uniform(0.3, 1.0, size=len(obstacles))

    n_side = 2 * d.semantic_map_half_size_px
    coords = (np.arange(n_side) - d.semantic_map_half_size_px + 0.5) * d.semantic_map_resolution
    xx, yy = np.meshgrid(coords, coords, indexing="ij")  # map[i <-> x, j <-> y]
    local_map = np.zeros((n_side, n_side), dtype=np.uint8)
    for (ox, oy), r in zip(obstacles, radii):
        local_map[(xx - ox) ** 2 + (yy - oy) ** 2 <= r**2] = 1

    scans = []
    for _ in range(m.lidar_num_scans):
        ground = np.c_[
            rng.uniform(-half_m / 2, half_m / 2, (m.lidar_max_points, 2)),
            rng.normal(-0.4, 0.02, m.lidar_max_points),
        ]
        k = m.lidar_max_points // 4
        idx = rng.integers(0, len(obstacles), k)
        angle = rng.uniform(0, 2 * np.pi, k)
        wall = np.c_[
            obstacles[idx] + radii[idx, None] * np.c_[np.cos(angle), np.sin(angle)],
            rng.uniform(-0.4, 1.5, k),
        ]
        pts = np.concatenate([ground[: m.lidar_max_points - k], wall])
        scans.append(np.c_[pts, rng.uniform(0, 100, len(pts))].astype(np.float32))

    futures = [_future(rng, m.num_waypoints, m.dt) for _ in range(int(rng.integers(1, 4)))]
    goal = futures[0][-1, :2] * rng.uniform(2.0, 10.0)
    return {
        "path": futures[0],
        "all_paths": futures,
        "pose": np.zeros(3, dtype=np.float32),
        "goal": goal.astype(np.float32),
        "previous_path": _past(rng, m.past_traj_len, 0.5),
        "lidar_dn": np.stack(scans),
        "local_map": local_map,
    }


def make_synthetic_dataset(
    root: str, num_samples: int, cfg: Optional[Config] = None, seed: int = 0
) -> str:
    """Write ``num_samples`` samples and ``index.pkl`` under ``root``; returns the index path."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    samples_dir, maps_dir = "samples", "maps"
    os.makedirs(os.path.join(root, samples_dir), exist_ok=True)
    os.makedirs(os.path.join(root, maps_dir), exist_ok=True)
    ids = []
    for i in range(num_samples):
        name = f"{i:06d}.pkl"
        with open(os.path.join(root, samples_dir, name), "wb") as f:
            pickle.dump(make_sample(rng, cfg), f)
        ids.append((name, name))
    index_path = os.path.join(root, cfg.data.index_file)
    with open(index_path, "wb") as f:
        pickle.dump({"root": [maps_dir, samples_dir], "ids": ids}, f)
    return index_path
