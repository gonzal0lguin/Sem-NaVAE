"""Point-cloud preprocessing shared by training and deployment."""

from __future__ import annotations

import numpy as np


def voxelize_with_intensity(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    """Average ``(N, 4)`` [x, y, z, i] points per voxel; subsample or zero-pad to ``max_points``."""
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((counts.shape[0], 4), dtype=np.float32)
    np.add.at(sums, inv, points[:, :4])
    means = sums / counts[:, None]

    n = means.shape[0]
    if n > max_points:
        means = means[np.random.choice(n, max_points, replace=False)]  # noqa: NPY002
    elif n < max_points:
        means = np.concatenate((means, np.zeros((max_points - n, 4), dtype=np.float32)), axis=0)
    return means


def normalize_lidar(scans: np.ndarray, norm_dist: float) -> np.ndarray:
    """Scale xyz by ``1 / norm_dist`` and standardise intensity over all scans (in place)."""
    scans[..., :3] = scans[..., :3] / norm_dist
    scans[..., 3] = (scans[..., 3] - scans[..., 3].mean()) / (scans[..., 3].std() + 1e-6)
    return scans


def past_trajectory(
    stamps: np.ndarray, positions: np.ndarray, yaw: float, max_vel: float
) -> np.ndarray:
    """``[x, y, vx, vy]`` of the past poses in the current robot frame.

    The oldest pose is dropped (it only serves the finite differences). Velocities are
    differences over the actual stamps, divided by ``max_vel``; positions are relative to
    the newest one; both are rotated by ``-yaw`` (odometry yaw of the newest pose).
    """
    vel = np.stack([np.gradient(positions[:, i], stamps) for i in range(2)], -1)
    rel = positions[:, :2] - positions[-1, :2]
    c, s = np.cos(yaw), np.sin(yaw)
    world_to_robot = np.array([[c, s], [-s, c]])
    rel, vel = rel @ world_to_robot.T, vel @ world_to_robot.T / max_vel
    return np.concatenate([rel[1:], vel[1:]], axis=-1)
