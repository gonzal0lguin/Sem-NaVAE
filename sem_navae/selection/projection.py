"""Rigid transforms and pinhole projection of trajectories onto the camera image (Sec. III-B3).

``y' = K [I | 0] T_{r->c} y``: waypoints are moved into the camera frame and projected
with the intrinsics. Points behind the camera or outside the image are flagged invalid
instead of being clipped or mirrored onto the image border.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion given as ``(x, y, z, w)`` (ROS order)."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class RigidTransform:
    """``p_target = R p_source + t`` (what tf2 ``lookup_transform(target, source)`` returns)."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> RigidTransform:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_quaternion(cls, translation: Sequence[float], quaternion: Sequence[float]):
        return cls(quaternion_to_matrix(quaternion), np.asarray(translation, dtype=float))

    @classmethod
    def from_xy_yaw(cls, x: float, y: float, yaw: float) -> RigidTransform:
        c, s = np.cos(yaw), np.sin(yaw)
        return cls(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform ``(..., 3)`` points."""
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def inverse(self) -> RigidTransform:
        r_inv = self.rotation.T
        return RigidTransform(r_inv, -r_inv @ self.translation)

    def __matmul__(self, other: RigidTransform) -> RigidTransform:
        """``(a @ b).apply(p) == a.apply(b.apply(p))``."""
        return RigidTransform(
            self.rotation @ other.rotation, self.rotation @ other.translation + self.translation
        )


def project_to_image(
    points_cam: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    min_depth: float = 0.1,
    optical_frame: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``(..., 3)`` camera-frame points to pixels.

    Args:
        points_cam: points in the camera frame.
        intrinsics: ``(3, 3)`` matrix K.
        image_size: ``(width, height)`` in pixels.
        min_depth: points closer than this along the optical axis are invalid [m].
        optical_frame: ``False`` for a ROS body frame such as ``camera_link`` (x forward,
            y left, z up; what the original code used); ``True`` for an optical frame
            (x right, y down, z forward).

    Returns:
        ``pixels`` ``(..., 2)`` as float ``(u, v)`` (NaN where the point is behind the
        camera) and ``valid`` ``(...)``: in front of the camera and inside the image.
    """
    p = np.asarray(points_cam, dtype=float)
    if optical_frame:
        x_r, y_d, depth = p[..., 0], p[..., 1], p[..., 2]
    else:
        x_r, y_d, depth = -p[..., 1], -p[..., 2], p[..., 0]
    in_front = depth > min_depth
    safe_depth = np.where(in_front, depth, np.nan)
    u = intrinsics[0, 0] * x_r / safe_depth + intrinsics[0, 1] * y_d / safe_depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * y_d / safe_depth + intrinsics[1, 2]
    width, height = image_size
    with np.errstate(invalid="ignore"):
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=-1), in_front & inside
