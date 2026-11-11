"""move_base adapter (actionlib ``MoveBaseAction``).

Params (``local_planner/params``): ``action_ns`` (default ``move_base``), ``waypoint_stride``.
"""

from __future__ import annotations

import math
from typing import Optional

from sem_navae_ros.local_planners.base import LocalPlanner, PlannerStatus


class MoveBasePlanner(LocalPlanner):
    def __init__(self, params=None):
        super().__init__(params)
        import actionlib
        import rospy
        from actionlib_msgs.msg import GoalStatus
        from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

        self._rospy = rospy
        self._goal_cls = MoveBaseGoal
        self._client = actionlib.SimpleActionClient(
            self.params.get("action_ns", "move_base"), MoveBaseAction
        )
        self._active = {GoalStatus.PENDING, GoalStatus.ACTIVE}
        self._succeeded = {GoalStatus.SUCCEEDED}
        self._failed = {GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST}
        self._sent = False

    def wait_until_ready(self, timeout: float) -> bool:
        return self._client.wait_for_server(self._rospy.Duration(timeout))

    def send_waypoint(self, xy, frame_id: str, yaw: Optional[float] = None) -> None:
        goal = self._goal_cls()
        goal.target_pose.header.frame_id = frame_id
        goal.target_pose.header.stamp = self._rospy.Time.now()
        goal.target_pose.pose.position.x = float(xy[0])
        goal.target_pose.pose.position.y = float(xy[1])
        yaw = 0.0 if yaw is None else yaw
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._client.send_goal(goal)
        self._sent = True

    def cancel(self) -> None:
        if self._sent:
            self._client.cancel_all_goals()
        self._sent = False

    def status(self) -> PlannerStatus:
        if not self._sent:
            return PlannerStatus.IDLE
        state = self._client.get_state()
        if state in self._active:
            return PlannerStatus.ACTIVE
        if state in self._succeeded:
            return PlannerStatus.SUCCEEDED
        if state in self._failed:
            return PlannerStatus.FAILED
        return PlannerStatus.IDLE  # PREEMPTED / RECALLED
