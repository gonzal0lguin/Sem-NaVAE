"""Interface between the trajectory selector and the robot's local planner.

The selector hands one waypoint at a time to a ``LocalPlanner`` and polls its status. Any
planner (move_base, a learned controller, a custom action server) can be used by writing a
small adapter; see ``docs/local_planner.md``. This module must stay importable without ROS.
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Dict, Optional, Tuple


class PlannerStatus(enum.Enum):
    IDLE = "idle"  # no waypoint sent, or the last one was cancelled
    ACTIVE = "active"  # moving towards the last waypoint
    SUCCEEDED = "succeeded"  # the last waypoint was reached
    FAILED = "failed"  # the planner gave up on the last waypoint


class LocalPlanner(abc.ABC):
    """Base class of the local planner plugins.

    Args:
        params: the ``local_planner/params`` dictionary from the ROS parameter server.

    Attributes:
        waypoint_stride: the selector sends every ``waypoint_stride``-th waypoint (plus the last
            one). The paper's RL planner used 3; planners that handle dense waypoints use 1.
    """

    waypoint_stride: int = 1

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self.waypoint_stride = int(self.params.get("waypoint_stride", type(self).waypoint_stride))
        if self.waypoint_stride < 1:
            raise ValueError(f"waypoint_stride must be >= 1, got {self.waypoint_stride}")

    @abc.abstractmethod
    def send_waypoint(
        self, xy: Tuple[float, float], frame_id: str, yaw: Optional[float] = None
    ) -> None:
        """Start driving to ``xy`` (expressed in ``frame_id``), replacing any previous waypoint."""

    @abc.abstractmethod
    def cancel(self) -> None:
        """Stop pursuing the current waypoint."""

    @abc.abstractmethod
    def status(self) -> PlannerStatus:
        """State of the last waypoint sent."""

    def wait_until_ready(self, timeout: float) -> bool:
        """Block until the planner can accept waypoints. Returns False on timeout."""
        return True
