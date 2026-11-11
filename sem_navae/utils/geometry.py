"""Planar geometry helpers."""

from __future__ import annotations

import numpy as np


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angle_batch(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def rotate_pointcloud(points: np.ndarray, theta: float) -> np.ndarray:
    """Rotate ``(N, 3)`` points by ``theta`` around z."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ R.T


def world_to_local(points: np.ndarray, pose) -> np.ndarray:
    """Express ``(N, 2)`` world points in the frame of ``pose = (x, y, yaw)``."""
    x, y, yaw = pose
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    return (points - np.array([x, y])) @ R.T


def get_path_dist_to_goal(waypoints: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Distance from each path end to the goal and heading error (Eq. 9 inputs).

    Args:
        waypoints: ``(K, N, >=2)`` paths.
        goal: ``(2,)`` goal in the same frame.

    Returns:
        ``(K, 2)`` array of ``[rho, theta]``.
    """
    distances = np.linalg.norm(waypoints[:, -1, :2] - goal, axis=1)
    final_heading = np.arctan2(
        waypoints[:, -1, 1] - waypoints[:, -2, 1], waypoints[:, -1, 0] - waypoints[:, -2, 0]
    )
    # Direction from the last waypoint *towards* the goal (the original code used
    # wp - goal, which scored paths heading straight at the goal as the worst).
    last_to_goal = np.arctan2(goal[1] - waypoints[:, -1, 1], goal[0] - waypoints[:, -1, 0])
    angles = normalize_angle_batch(final_heading - last_to_goal)
    return np.stack((distances, angles), axis=-1)
