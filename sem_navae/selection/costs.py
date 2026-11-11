"""Semantic and goal costs of candidate trajectories (Eqs. 8-10).

Per-waypoint costs are kept **unweighted**; the discount gamma^j is applied only when a
trajectory is scored, with an explicit waypoint index j. This keeps re-scoring of a
partially executed trajectory consistent with the cost it was selected with.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from sem_navae.utils.geometry import get_path_dist_to_goal


def sample_waypoint_costs(
    cost_image: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray,
    unknown_cost: float,
    occlusion_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-waypoint terms of Eq. 8, without the discount.

    A waypoint costs ``I_c(p)`` if it projects inside the image onto a pixel whose cost
    does not exceed ``T_occ``; otherwise it is unknown and costs ``C_u``: either the pixel
    exceeds ``T_occ`` (occlusion, ``u_j = 1``) or the waypoint is behind the camera or
    outside the image (no evidence).

    Args:
        cost_image: ``(H, W)`` semantic cost map I_c.
        pixels: ``(..., N, 2)`` float pixel coordinates ``(u, v)``.
        valid: ``(..., N)`` from :func:`~sem_navae.selection.projection.project_to_image`.

    Returns:
        ``costs`` ``(..., N)`` and ``observed`` ``(..., N)``: the waypoint projected inside
        the image (its cost reflects the current image, occluded or not).
    """
    h, w = cost_image.shape
    u = np.where(valid, pixels[..., 0], 0).astype(int).clip(0, w - 1)
    v = np.where(valid, pixels[..., 1], 0).astype(int).clip(0, h - 1)
    sampled = cost_image[v, u].astype(float)
    occluded = sampled > occlusion_threshold
    costs = np.where(valid & ~occluded, sampled, float(unknown_cost))
    return costs, valid.copy()


def discount(num_waypoints: int, gamma: float, first_index: int = 1) -> np.ndarray:
    """``gamma ** j`` for ``j = first_index, ..., first_index + N - 1`` (Eq. 8 uses j = 1..N)."""
    return gamma ** np.arange(first_index, first_index + num_waypoints, dtype=float)


def semantic_cost(waypoint_costs: np.ndarray, gamma: float, first_index: int = 1) -> np.ndarray:
    """Eq. 8: discounted sum of per-waypoint costs over the last axis."""
    n = waypoint_costs.shape[-1]
    return np.sum(waypoint_costs * discount(n, gamma, first_index), axis=-1)


def goal_cost(paths: np.ndarray, goal: np.ndarray, alpha1: float, alpha2: float) -> np.ndarray:
    """Eq. 9: ``alpha1 log(1 + d) + alpha2 |theta_end| / pi`` for ``(K, N, >=2)`` paths.

    ``d`` is the distance from the last waypoint to the goal and ``theta_end`` the angle
    between the last segment's heading and the direction from the last waypoint to the
    goal. ``goal`` must be in the frame of ``paths``.
    """
    rho_theta = get_path_dist_to_goal(np.asarray(paths), np.asarray(goal, dtype=float))
    return alpha1 * np.log1p(rho_theta[:, 0]) + alpha2 * np.abs(rho_theta[:, 1]) / np.pi
