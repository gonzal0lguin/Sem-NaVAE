"""ROS-free selection core: cost map + candidates + transforms -> switching decisions.

``selector_node`` only converts messages and tf lookups into the arguments below, so the
whole selection logic can be tested and replayed without ROS.

Frames: candidates arrive in the robot frame (``base_link``) at the time they were
generated. The adopted trajectory is stored in a fixed frame (e.g. ``odom``) using the
transform *at that time*, so it does not drift while the robot moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sem_navae.selection.config import SelectionConfig
from sem_navae.selection.costmap import SemanticCostmap
from sem_navae.selection.costs import goal_cost, sample_waypoint_costs
from sem_navae.selection.projection import RigidTransform, project_to_image
from sem_navae.selection.update_policy import Decision, UpdatePolicy


@dataclass
class ScoredPaths:
    waypoint_costs: np.ndarray  # (K, N) unweighted
    observed: np.ndarray  # (K, N)
    pixels: np.ndarray  # (K, N, 2)


class SelectionCore:
    def __init__(
        self,
        cfg: SelectionConfig,
        intrinsics: np.ndarray,
        image_size: Tuple[int, int],
    ):
        self.cfg = cfg
        self.intrinsics = np.asarray(intrinsics, dtype=float).reshape(3, 3)
        self.image_size = image_size  # (width, height)
        self.class_names = list(cfg.classes.keys())
        self.class_costs = np.array([cfg.classes[c] for c in self.class_names], dtype=float)
        self.policy = UpdatePolicy(cfg.gamma, cfg.epsilon, cfg.mode)
        self.costmap: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- helpers
    def score_paths(self, paths: np.ndarray, T_cam_from_paths: RigidTransform) -> ScoredPaths:
        """Project ``(K, N, 3)`` paths and sample per-waypoint costs from the cost map."""
        if self.costmap is None:
            raise RuntimeError("No cost map yet")
        pixels, valid = project_to_image(
            T_cam_from_paths.apply(paths),
            self.intrinsics,
            self.image_size,
            self.cfg.min_depth,
            self.cfg.camera_optical_frame,
        )
        costs, observed = sample_waypoint_costs(
            self.costmap, pixels, valid, self.cfg.unknown_cost, self.cfg.occlusion_threshold
        )
        return ScoredPaths(costs, observed, pixels)

    # ----------------------------------------------------------------- events
    def on_costmap(
        self, costmap, T_cam_from_fixed: Optional[RigidTransform] = None
    ) -> Optional[float]:
        """Store a new cost map (``SemanticCostmap`` or ``(H, W)`` array) and re-score y_curr.

        Returns the updated J(y_curr), or ``None`` if nothing is being executed.
        """
        self.costmap = costmap.costs if isinstance(costmap, SemanticCostmap) else costmap
        cur = self.policy.current
        if cur is None or cur.waypoints is None or T_cam_from_fixed is None:
            return None
        scored = self.score_paths(cur.waypoints[None], T_cam_from_fixed)
        return self.policy.on_costmap(scored.waypoint_costs[0], scored.observed[0])

    def on_candidates(
        self,
        paths: np.ndarray,
        goal: np.ndarray,
        T_cam_from_robot: RigidTransform,
        T_fixed_from_robot: RigidTransform,
        feasible: Optional[np.ndarray] = None,
    ) -> Tuple[Decision, Optional[np.ndarray]]:
        """Score a new candidate set and decide whether to switch.

        Args:
            paths: ``(K, N, 3)`` candidates in the robot frame at generation time.
            goal: ``(2,)`` goal in the same robot frame.
            T_cam_from_robot: camera extrinsics ``T_{r->c}``.
            T_fixed_from_robot: robot pose in the fixed frame at generation time.
            feasible: optional ``(K,)`` collision-free mask; infeasible paths are skipped.

        Returns:
            The decision (``index`` refers to ``paths``) and, on a switch, the chosen path
            in the fixed frame.
        """
        paths = np.asarray(paths, dtype=float)
        keep = np.arange(len(paths)) if feasible is None else np.flatnonzero(feasible)
        scored = self.score_paths(paths[keep], T_cam_from_robot)
        g_cost = goal_cost(paths[keep], goal, self.cfg.alpha1, self.cfg.alpha2)
        fixed = T_fixed_from_robot.apply(paths[keep])
        decision = self.policy.on_candidates(scored.waypoint_costs, g_cost, fixed)
        if decision.index is not None:
            decision.index = int(keep[decision.index])
            return decision, self.policy.current.waypoints
        return decision, None

    def on_progress(self, waypoint_index: int) -> None:
        self.policy.on_progress(waypoint_index)

    def reset(self) -> None:
        self.policy.reset()

    @property
    def current_waypoints(self) -> Optional[np.ndarray]:
        cur = self.policy.current
        return None if cur is None else cur.waypoints
