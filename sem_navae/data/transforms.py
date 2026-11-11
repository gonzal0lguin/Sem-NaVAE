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
