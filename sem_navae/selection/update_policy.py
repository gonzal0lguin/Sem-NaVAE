"""Asynchronous trajectory update (Sec. III-B4): when to switch to a new candidate.

The executing trajectory ``y_curr`` is re-scored on every new cost map (``f_clip``) and
compared with the best candidate of every new generated set (``f_gen``). A switch happens
only if ``J(y_new) < J(y_curr) - epsilon`` (hysteresis).

Modes (all use the same objective ``J = J_sem + J_goal``, Eqs. 8-10):

* ``cost_to_go`` (default): ``y_curr`` is scored on its remaining waypoints only,
  discounted from the robot's current position (``gamma^1`` for the next waypoint), so
  both sides of the comparison are costs from "here". The remaining part is shorter than a
  fresh candidate, which slightly favours keeping ``y_curr``; the goal term compensates.
* ``paper``: Sec. III-B4 as written. Traversed waypoints keep their frozen costs, future
  ones are re-projected, and the whole trajectory is discounted with its original indices.
* ``periodic``: always switch to the best candidate of every new set (NaVAE-FixF
  ablation, same objective, ``epsilon = -inf``).

Waypoints that leave the image while ``y_curr`` is executed (e.g. below the camera's field
of view as the robot gets close) keep their last observed cost instead of becoming unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sem_navae.selection.costs import semantic_cost

MODES = ("cost_to_go", "paper", "periodic")


@dataclass
class CurrentTrajectory:
    waypoints: Optional[np.ndarray]  # (N, 3) in the fixed frame (opaque to the policy)
    waypoint_costs: np.ndarray  # (N,) unweighted, see costs.sample_waypoint_costs
    goal_cost: float  # J_goal; frame-invariant while the goal is fixed
    progress: int = 0  # index of the next waypoint to reach; [0, progress) are frozen

    @property
    def finished(self) -> bool:
        return self.progress >= len(self.waypoint_costs)


@dataclass
class Decision:
    switch: bool
    index: Optional[int]  # chosen candidate (index into the candidate set) if switch
    best_cost: float  # J of the best candidate
    current_cost: float  # J of y_curr before the decision (inf if none)
    candidate_costs: np.ndarray = field(default_factory=lambda: np.empty(0))


class UpdatePolicy:
    def __init__(self, gamma: float, epsilon: float, mode: str = "cost_to_go"):
        if mode not in MODES:
            raise ValueError(f"Unknown update mode '{mode}', expected one of {MODES}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.mode = mode
        self.current: Optional[CurrentTrajectory] = None

    def reset(self) -> None:
        self.current = None

    # ------------------------------------------------------------------ scoring
    def candidate_costs(self, waypoint_costs: np.ndarray, goal_costs: np.ndarray) -> np.ndarray:
        """Eq. 10 objective ``J_sem + J_goal`` for ``(K, N)`` waypoint costs."""
        return semantic_cost(waypoint_costs, self.gamma) + np.asarray(goal_costs, dtype=float)

    def current_cost(self) -> float:
        cur = self.current
        if cur is None or cur.finished:
            return float("inf")
        if self.mode == "paper":
            sem = semantic_cost(cur.waypoint_costs, self.gamma)
        else:  # cost_to_go (periodic never compares, report the same quantity)
            sem = semantic_cost(cur.waypoint_costs[cur.progress :], self.gamma)
        return float(sem + cur.goal_cost)

    # ------------------------------------------------------------------- events
    def on_candidates(
        self,
        waypoint_costs: np.ndarray,
        goal_costs: np.ndarray,
        waypoints: Optional[np.ndarray] = None,
    ) -> Decision:
        """Score a new candidate set and adopt its best member if it wins (Eq. 10 + hysteresis).

        Args:
            waypoint_costs: ``(K, N)`` unweighted per-waypoint costs.
            goal_costs: ``(K,)`` J_goal of each candidate.
            waypoints: optional ``(K, N, 3)`` geometry stored with the adopted candidate.
        """
        current = self.current_cost()
        if len(waypoint_costs) == 0:
            return Decision(False, None, float("inf"), current)
        costs = self.candidate_costs(waypoint_costs, goal_costs)
        best = int(np.argmin(costs))
        switch = (
            self.current is None
            or self.current.finished
            or self.mode == "periodic"
            or costs[best] < current - self.epsilon
        )
        if switch:
            self.current = CurrentTrajectory(
                waypoints=None if waypoints is None else np.array(waypoints[best]),
                waypoint_costs=np.array(waypoint_costs[best], dtype=float),
                goal_cost=float(goal_costs[best]),
            )
        return Decision(switch, best if switch else None, float(costs[best]), current, costs)

    def on_costmap(self, waypoint_costs: np.ndarray, observed: np.ndarray) -> float:
        """Re-score ``y_curr`` from a new cost map; returns the updated J(y_curr).

        Frozen (already traversed) waypoints never change; future waypoints take the new
        cost where they were observed in the new image and keep their previous cost
        otherwise.
        """
        cur = self.current
        if cur is None:
            return float("inf")
        update = np.asarray(observed, dtype=bool).copy()
        update[: cur.progress] = False
        cur.waypoint_costs = np.where(update, waypoint_costs, cur.waypoint_costs)
        return self.current_cost()

    def on_progress(self, waypoint_index: int) -> None:
        """Mark waypoints ``[0, waypoint_index)`` as traversed (monotone)."""
        if self.current is not None:
            n = len(self.current.waypoint_costs)
            self.current.progress = max(self.current.progress, min(int(waypoint_index), n))
