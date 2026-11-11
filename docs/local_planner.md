# Local planner plugins

The selector node chooses a trajectory and hands it to a **local planner** one waypoint at a time.
The planner turns each waypoint into velocity commands. Sem-NaVAE does not depend on a
particular planner: any planner can be used by writing a small adapter class.

> The paper's experiments used a proprietary RL local planner [37], which is **not** included.
> The paper's DWA ablation (Table III) reached the same success rate, so the built-in `move_base`
> adapter is a reasonable default.

## Choosing a planner

Set it in `config/sem_navae.yaml` (or with `rosparam`):

```yaml
local_planner:
  plugin: "move_base"          # or "null", or "my_pkg.my_module:MyPlanner"
  params:                      # passed unchanged to the planner's constructor
    action_ns: "move_base"
    waypoint_stride: 1
```

| Plugin | Description |
|--------|-------------|
| `move_base` | Sends each waypoint as an actionlib `MoveBaseAction` goal to `params.action_ns`. |
| `null` | Logs the waypoints and reports them as reached immediately. The robot does not move. Useful for dry runs and bag replay. |

`waypoint_stride: n` sends every n-th waypoint plus the last one. Planners that handle dense
waypoints poorly benefit from it. The paper's RL planner used `3`.

## Writing an adapter

Subclass `sem_navae_ros.local_planners.LocalPlanner` and implement three methods. The module only
needs to be importable by the selector node, e.g. inside your own catkin package:

```python
import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from my_planner.msg import GoToAction, GoToGoal
from sem_navae_ros.local_planners import LocalPlanner, PlannerStatus


class MyPlanner(LocalPlanner):
    waypoint_stride = 3  # default; overridable via params

    def __init__(self, params):
        super().__init__(params)
        self.client = actionlib.SimpleActionClient(params.get("action_ns", "goto"), GoToAction)

    def wait_until_ready(self, timeout):  # optional, defaults to True
        return self.client.wait_for_server(rospy.Duration(timeout))

    def send_waypoint(self, xy, frame_id, yaw=None):
        goal = GoToGoal()
        goal.target.header.frame_id = frame_id
        goal.target.header.stamp = rospy.Time.now()
        goal.target.pose.position.x, goal.target.pose.position.y = xy
        goal.target.pose.orientation.w = 1.0
        self.client.send_goal(goal)

    def cancel(self):
        self.client.cancel_all_goals()

    def status(self):  # IDLE | ACTIVE | SUCCEEDED | FAILED
        state = self.client.get_state()  # map your planner's states here
        if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            return PlannerStatus.ACTIVE
        if state == GoalStatus.SUCCEEDED:
            return PlannerStatus.SUCCEEDED
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED):
            return PlannerStatus.FAILED
        return PlannerStatus.IDLE
```

Then set `plugin: "my_pkg.my_planner:MyPlanner"`.

## Contract

- `send_waypoint(xy, frame_id, yaw=None)` replaces any waypoint that is still pending. `xy` is in
  `frame_id` (the selector's fixed frame) and `yaw` is in radians. A planner may ignore `yaw`.
- The selector advances to the next waypoint when the robot is within 1 m of the current one
  **or** `status()` returns `SUCCEEDED`. It skips the waypoint when `status()` returns `FAILED`
  or after a 10 s timeout.
- The selector calls `cancel()` when it switches trajectories, reaches the goal, starts a
  recovery or is overridden by the joystick.
- Calls happen on the selector's main loop, so they must not block.
- `base.py` imports nothing from ROS, so adapters can be unit-tested without a ROS master
  (see `tests/test_local_planner_plugin.py`).
