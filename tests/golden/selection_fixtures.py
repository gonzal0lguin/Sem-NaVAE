"""Synthetic inputs for the selection parity tests (numpy only, no ROS).

The same arrays are fed to the legacy ``TrajectorySelector._calculate_path_cost_full``
(``generate_selection_from_legacy.py``) and to ``sem_navae.selection``.
"""

from __future__ import annotations

import numpy as np

IMG_H, IMG_W = 96, 128
INTRINSICS = np.array([[80.0, 0.0, 64.0], [0.0, 80.0, 48.0], [0.0, 0.0, 1.0]])

GAMMA = 0.8
UNKNOWN_COST = 2.0
ALPHA1 = 2.0
ALPHA2 = 0.2


def _quat_y(angle: float) -> np.ndarray:
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])


def _quat_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])


# tf2 "lookup_transform(target, source)" results as (translation, quaternion xyzw).
TF = {
    # camera_link <- base_link: camera 0.2 m ahead, 0.6 m up, pitched 10 deg down.
    ("camera_link", "base_link"): (np.array([-0.2, 0.0, -0.6]), _quat_y(np.deg2rad(10.0))),
    # base_link <- map: robot at some pose in the map.
    ("base_link", "map"): (np.array([1.0, -2.0, 0.0]), _quat_z(np.deg2rad(30.0))),
}


def costmap() -> np.ndarray:
    """Per-pixel class cost image (uint8), road at the bottom, sky at the top."""
    cm = np.zeros((IMG_H, IMG_W), dtype=np.uint8)  # road = 0
    cm[: IMG_H // 3] = 4  # sky
    cm[IMG_H // 3 : 2 * IMG_H // 3, : IMG_W // 2] = 2  # grass, left
    cm[IMG_H // 3 : 2 * IMG_H // 3, IMG_W // 2 :] = 3  # tree, right
    cm[2 * IMG_H // 3 :, : IMG_W // 4] = 1  # some soft class, bottom-left
    return cm


def paths() -> np.ndarray:
    """``(6, 12, 3)`` candidate paths in base_link (z = 0), avoiding exact zero depth."""
    t = np.arange(1, 13, dtype=float)
    out = [
        np.stack([0.8 * t, 0.01 * t, 0 * t], -1),  # straight ahead
        np.stack([0.7 * t, 0.05 * t**1.5, 0 * t], -1),  # curving left
        np.stack([0.7 * t, -0.05 * t**1.5, 0 * t], -1),  # curving right
        np.stack([0.3 * t, 0.9 * t, 0 * t], -1),  # sharp left, leaves the image
        np.stack([-0.5 * t - 0.3, 0.1 * t, 0 * t], -1),  # backwards, behind the camera
        np.stack([0.25 * t, -0.02 * t, 0 * t], -1),  # short, stays near the robot
    ]
    return np.stack(out)


GOAL_MAP = np.array([12.0, 5.0])
