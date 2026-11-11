"""A planner that only logs the waypoints: for dry runs and bag replay."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from sem_navae_ros.local_planners.base import LocalPlanner, PlannerStatus

log = logging.getLogger(__name__)


def _loginfo(msg: str) -> None:
    try:
        import rospy

        rospy.loginfo(msg)
    except ImportError:
        log.info(msg)


class NullPlanner(LocalPlanner):
    """Accepts every waypoint and reports it as reached; never moves the robot."""

    def __init__(self, params=None):
        super().__init__(params)
        self._status = PlannerStatus.IDLE
        self.last_waypoint: Optional[Tuple[float, float, str, Optional[float]]] = None

    def send_waypoint(self, xy, frame_id: str, yaw: Optional[float] = None) -> None:
        self.last_waypoint = (float(xy[0]), float(xy[1]), frame_id, yaw)
        self._status = PlannerStatus.SUCCEEDED
        _loginfo(f"[NullPlanner] waypoint ({xy[0]:.2f}, {xy[1]:.2f}) in '{frame_id}'")

    def cancel(self) -> None:
        self._status = PlannerStatus.IDLE

    def status(self) -> PlannerStatus:
        return self._status
