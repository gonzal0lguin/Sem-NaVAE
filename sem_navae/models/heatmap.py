"""Bird's-eye-view traversability heatmap I_seg (paper Eq. 2) and Gaussian smoothing helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sem_navae.config import HeatmapSpec


def gaussian_kernel2d(kernel_size: int, sigma: float, device=None) -> torch.Tensor:
    ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_filter(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a batch of 2D maps ``(B, H, W)`` with an isotropic Gaussian (std in pixels)."""
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = gaussian_kernel2d(kernel_size, sigma, device=maps.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return F.conv2d(maps.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)


def metric_to_pixel(xy: torch.Tensor, spec: HeatmapSpec) -> torch.Tensor:
    """Robot-frame ``[..., (x, y)]`` [m] -> integer ``[..., (col, row)]`` of the heatmap.

    Column = x, row = y, the robot at the map centre. Shared by the heatmap builder and the
    decoder's constrained sampling so they can never disagree on the metric scale.
    """
    return torch.floor(xy / spec.resolution + spec.origin_px).long()


def gaussian_heatmap(
    points: torch.Tensor,
    probs: torch.Tensor,
    xlim: Sequence[float],
    ylim: Sequence[float],
    res: float = 0.1,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Average ``probs`` per grid cell, then smooth with a Gaussian.

    Args:
        points: ``(B, N, 2)`` xy coordinates [m].
        probs: ``(B, N)`` value of each point.
        xlim, ylim: metric extent of the grid [m].
        res: cell size [m/px].
        sigma: Gaussian std [px].

    Returns:
        ``(B, H, W)`` map, row index = y, column index = x.
    """
    B = points.shape[0]
    device = points.device

    x_bins = int((xlim[1] - xlim[0]) / res)
    y_bins = int((ylim[1] - ylim[0]) / res)
    flat_size = x_bins * y_bins

    xi = torch.floor((points[..., 0] - xlim[0]) / res).long()
    yi = torch.floor((points[..., 1] - ylim[0]) / res).long()
    valid = (xi >= 0) & (xi < x_bins) & (yi >= 0) & (yi < y_bins)

    flat_idx = yi * x_bins + xi
    flat_idx[~valid] = 0
    masked_probs = probs * valid
    scatter_idx = (flat_idx + torch.arange(B, device=device).unsqueeze(1) * flat_size).view(-1)

    voxel_map = torch.zeros(B * flat_size, device=device)
    count_map = torch.zeros_like(voxel_map)
    voxel_map.scatter_add_(0, scatter_idx, masked_probs.reshape(-1))
    count_map.scatter_add_(0, scatter_idx, valid.float().view(-1))

    voxel_map = voxel_map.view(B, y_bins, x_bins)
    count_map = count_map.view(B, y_bins, x_bins)
    nonzero = count_map > 0
    voxel_map[nonzero] = voxel_map[nonzero] / count_map[nonzero]

    return gaussian_filter(voxel_map, sigma)


def filter_and_pad_pcl(
    points: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the masked points of each cloud and right-pad with zeros to a common length.

    Args:
        points: ``(B, N, C)``.
        values: ``(B, N)`` per-point value carried along.
        mask: ``(B, N)`` boolean selection.

    Returns:
        ``(B, M, C)`` points, ``(B, M)`` values and ``(B, M)`` validity mask, M = max count.
    """
    B, _, C = points.shape
    device = points.device
    M_max = mask.sum(dim=1).max()

    out_points = torch.zeros((B, M_max, C), device=device)
    out_values = torch.zeros((B, M_max), device=device)
    out_mask = torch.zeros((B, M_max), dtype=torch.bool, device=device)
    for b in range(B):
        idx = mask[b].bool()
        M = idx.sum()
        out_points[b, :M] = points[b][idx]
        out_values[b, :M] = values[b][idx]
        out_mask[b, :M] = True
    return out_points, out_values, out_mask


def traversability_heatmap(
    points: torch.Tensor,
    logprobs: torch.Tensor,
    labels: torch.Tensor,
    spec: HeatmapSpec,
    lidar_norm_dist: float,
    sigma: Optional[float] = None,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Build I_seg from the PointNet segmentation (Eq. 2).

    Traversable points add their probability, non-traversable points below
    ``spec.max_obstacle_height`` subtract theirs; the difference is clipped to [-1, 1].
    Eq. 2 adds +1 so that I_seg >= 0 with "unknown" at 1; the decoder's ``> 0`` test on
    this map is the same as ``> 1`` on Eq. 2's.

    Args:
        points: ``(B, N, 4)`` normalised LiDAR points (xyz divided by ``lidar_norm_dist``).
        logprobs: ``(B, N, 2)`` per-point log-probabilities (class 0 = traversable).
        labels: ``(B, N)`` argmax labels.
        spec: map extent / resolution / smoothing.
        sigma: override of ``spec.sigma_px`` (tests only).
        lidar_norm_dist: factor that converts normalised xyz back to metres.
    """
    sigma = spec.sigma_px if sigma is None else sigma
    half = spec.size_m / 2.0
    max_z = spec.max_obstacle_height / lidar_norm_dist
    pos_pts, pos_probs, _ = filter_and_pad_pcl(points, torch.exp(logprobs[:, :, 0]), labels == 0)
    neg_pts, neg_probs, _ = filter_and_pad_pcl(
        points, torch.exp(logprobs[:, :, 1]), (labels == 1) & (points[:, :, 2] < max_z)
    )
    lim = [-half, half]
    a = gaussian_heatmap(
        pos_pts[:, :, :2] * lidar_norm_dist, pos_probs, lim, lim, spec.resolution, sigma
    )
    b = gaussian_heatmap(
        neg_pts[:, :, :2] * lidar_norm_dist, neg_probs, lim, lim, spec.resolution, sigma
    )
    return torch.clip(a - b * penalty, -1, 1)
