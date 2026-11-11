#!/usr/bin/env python3
"""Sem-NaVAE trajectory selector: ROS I/O around ``sem_navae.selection.SelectionCore``.

* Camera images are segmented with CLIPSeg at ``f_clip``; every new cost map re-scores the
  executing trajectory (Sec. III-B4).
* Every collision-filtered candidate set (``f_gen``) is scored with Eqs. 8-10 and may
  replace the executing trajectory (hysteresis ``epsilon``).
* The adopted trajectory is stored in ``fixed_frame`` with the transform at the candidates'
  stamp, then dispatched waypoint by waypoint to the local planner plugin.

All decision logic lives in ``sem_navae.selection`` and is unit-tested without ROS.
"""

import threading
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sem_navae_ros.local_planners import PlannerStatus, load_local_planner
from sem_navae_ros.message_helpers import (
    camera_params,
    collisions_from_msg,
    colorize_labels,
    cv2_to_imgmsg,
    dispatch_indices,
    draw_paths,
    imgmsg_to_rgb,
    lookup,
    numpy_to_path,
    param,
    patharray_to_numpy,
    rotate_in_place,
    stop_robot,
)
from sem_navae_ros.msg import PathArray
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Joy
from std_msgs.msg import Empty, Float32

from sem_navae.selection import (
    SelectionCore,
    SemanticCostmap,
    project_to_image,
    resize_nearest,
    selection_config_from_dict,
)
from sem_navae.selection.clipseg_costmap import ClipSegCostmapper


class SelectorNode:
    def __init__(self):
        self.cfg = selection_config_from_dict(param("selection", {}))
        self.robot_frame = param("frames/robot", "base_link")
        self.camera_frame = param("frames/camera", "camera_link")
        self.fixed_frame = param("frames/fixed", "odom")
        self.goal_th = float(param("model/goal_th", 5.0))
        self.reach_radius = float(param("execution/waypoint_reach_radius", 1.0))
        self.waypoint_timeout = float(param("execution/waypoint_timeout", 10.0))
        self.recovery_after = int(param("execution/recovery_after_empty_sets", 3))
        self.recovery_speed = float(param("execution/recovery_angular_vel", 0.4))
        self.recovery_timeout = float(param("execution/recovery_timeout", 10.0))
        self.joy_button = int(param("execution/joy_override_button", 4))

        self._lock = threading.Lock()
        self._image_msg: Optional[Image] = None
        self._image_new = False
        self._goal_fixed: Optional[np.ndarray] = None  # goal (x, y) in the fixed frame
        self._goal_key: Optional[np.ndarray] = None  # goal as received, to detect changes
        self._done = True
        self._speed = 0.0
        self._joy_override = False
        self._empty_sets = 0
        self._free_dirs = [-1.0]
        self._goal_bearing = 0.0  # goal direction in the robot frame at the last set
        self._new_trajectory = False
        self._dispatch = []  # waypoint indices still to send
        self._sent_at: Optional[rospy.Time] = None
        self._recovery_until: Optional[rospy.Time] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        camera_ns = param("topics/camera_ns", "camera")
        info = rospy.wait_for_message(camera_ns + "/camera_info", CameraInfo, timeout=30.0)
        K, image_size = camera_params(info)
        self.core = SelectionCore(self.cfg, K, image_size)
        # costmap/topic: optional external class-label image (mono8, class indices in the
        # order of selection/classes) used instead of CLIPSeg (other segmenters, tests).
        self.label_topic = param("costmap/topic", "")
        self.costmapper = None
        if not self.label_topic:
            self.costmapper = ClipSegCostmapper(
                self.cfg.classes, self.cfg.prompt_template, self.cfg.clipseg_model
            )

        spec = param("local_planner/plugin", "move_base")
        self.planner = load_local_planner(spec, param("local_planner/params", {}))
        if not self.planner.wait_until_ready(timeout=5.0):
            rospy.logwarn(f"[selector] local planner '{spec}' is not ready yet")

        self._costmap_pub = rospy.Publisher("~costmap", Image, queue_size=1)
        self._costmap_cmp_pub = rospy.Publisher(
            "~costmap/compressed", CompressedImage, queue_size=1
        )
        self._annot_pub = rospy.Publisher("~path_on_costmap", Image, queue_size=1)
        self._annot_cmp_pub = rospy.Publisher(
            "~path_on_costmap/compressed", CompressedImage, queue_size=1
        )
        self._selected_pub = rospy.Publisher(
            param("topics/selection", "/sem_navae/selected_traj"), Path, queue_size=1, latch=True
        )
        self._current_cost_pub = rospy.Publisher("~current_cost", Float32, queue_size=1)
        self._best_cost_pub = rospy.Publisher("~best_candidate_cost", Float32, queue_size=1)
        self._done_pub = rospy.Publisher(
            param("topics/done_signal", "/sem_navae/stop"), Empty, queue_size=1
        )
        self._cmd_vel_pub = rospy.Publisher(
            param("topics/cmd_vel", "/cmd_vel"), Twist, queue_size=1
        )

        if self.costmapper is None:
            rospy.Subscriber(self.label_topic, Image, self._labels_cb, queue_size=1)
        else:
            rospy.Subscriber(
                camera_ns + "/image_raw", Image, self._image_cb, queue_size=1, buff_size=2**24
            )
            rospy.Timer(rospy.Duration(1.0 / self.cfg.clip_rate_hz), self._costmap_timer)
        rospy.Subscriber(
            param("topics/trajs_filt", "/sem_navae/trajectories_filt"),
            PathArray,
            self._candidates_cb,
            queue_size=1,
        )
        rospy.Subscriber(param("topics/odom", "/odom"), Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(param("topics/joy", "/joy"), Joy, self._joy_cb, queue_size=1)
        rospy.loginfo(f"[selector] mode={self.cfg.mode}, planner={spec}, fixed={self.fixed_frame}")

    # ----------------------------------------------------------------- inputs
    def _image_cb(self, msg: Image) -> None:
        self._image_msg, self._image_new = msg, True

    def _odom_cb(self, msg: Odometry) -> None:
        self._speed = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    def _joy_cb(self, msg: Joy) -> None:
        self._joy_override = len(msg.buttons) > self.joy_button and bool(
            msg.buttons[self.joy_button]
        )

    def _costmap_timer(self, _event) -> None:
        if not self._image_new:
            return
        msg, self._image_new = self._image_msg, False
        self._process_costmap(self.costmapper(imgmsg_to_rgb(msg)), msg.header.stamp)

    def _labels_cb(self, msg: Image) -> None:
        labels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        labels = labels[:, : msg.width].astype(int)
        width, height = self.core.image_size
        if labels.shape != (height, width):
            labels = resize_nearest(labels, (height, width))
        costmap = SemanticCostmap(costs=self.core.class_costs[labels], labels=labels)
        self._process_costmap(costmap, msg.header.stamp)

    def _process_costmap(self, costmap: SemanticCostmap, stamp) -> None:
        cam_from_fixed = lookup(self.tf_buffer, self.camera_frame, self.fixed_frame, stamp)
        with self._lock:
            cost = self.core.on_costmap(costmap, cam_from_fixed)
            waypoints = self.core.current_waypoints
        if cost is not None:
            self._current_cost_pub.publish(Float32(data=cost))
        self._publish_costmap(costmap, waypoints, cam_from_fixed, stamp)

    def _candidates_cb(self, msg: PathArray) -> None:
        if not msg.paths:
            return
        stamp = msg.paths[0].header.stamp
        paths = patharray_to_numpy(msg)
        feasible = ~collisions_from_msg(msg, paths.shape[:2]).any(axis=1)
        self._free_dirs = list(msg.free_dir.data) or [-1.0]

        goal_frame = msg.goal.header.frame_id or self.fixed_frame
        goal_xyz = np.array([msg.goal.pose.position.x, msg.goal.pose.position.y, 0.0])
        robot_from_goal = lookup(self.tf_buffer, self.robot_frame, goal_frame, stamp)
        fixed_from_robot = lookup(self.tf_buffer, self.fixed_frame, self.robot_frame, stamp)
        cam_from_robot = lookup(self.tf_buffer, self.camera_frame, self.robot_frame, stamp)
        if robot_from_goal is None or fixed_from_robot is None or cam_from_robot is None:
            return
        goal_robot = robot_from_goal.apply(goal_xyz)[:2]
        self._goal_bearing = float(np.arctan2(goal_robot[1], goal_robot[0]))

        with self._lock:
            if self._goal_key is None or np.any(goal_xyz != self._goal_key):
                self._goal_key = goal_xyz
                self._goal_fixed = fixed_from_robot.apply(robot_from_goal.apply(goal_xyz))[:2]
                self._done = False
                self.core.reset()
            if self._done:
                return
            if not feasible.any():
                self._empty_sets += 1
                rospy.logwarn_throttle(2.0, "[selector] no collision-free candidates")
                return
            self._empty_sets = 0
            if self.core.costmap is None:
                rospy.logwarn_throttle(2.0, "[selector] waiting for the first cost map")
                return
            decision, fixed_path = self.core.on_candidates(
                paths, goal_robot, cam_from_robot, fixed_from_robot, feasible
            )
            if decision.switch:
                self._new_trajectory = True
                self._recovery_until = None
        self._best_cost_pub.publish(Float32(data=decision.best_cost))
        if decision.switch:
            rospy.loginfo(
                f"[selector] switch: J {decision.current_cost:.2f} -> {decision.best_cost:.2f}"
            )
            self._selected_pub.publish(numpy_to_path(fixed_path, self.fixed_frame, stamp))

    # -------------------------------------------------------------- execution
    def step(self) -> None:
        """One tick of the (non-blocking) execution loop."""
        if self._joy_override:
            self._abort_motion()
            return
        if self._done or self._goal_fixed is None:
            return
        robot = lookup(self.tf_buffer, self.fixed_frame, self.robot_frame)
        if robot is None:
            return
        position = robot.translation[:2]

        if np.linalg.norm(position - self._goal_fixed) < self.goal_th:
            rospy.loginfo("[selector] goal reached")
            self._abort_motion()
            with self._lock:
                self._done = True
                self.core.reset()
            self._done_pub.publish(Empty())
            return

        if self._recovery_until is not None:
            self._recovery_tick()
            return

        with self._lock:
            waypoints = self.core.current_waypoints
            if self._new_trajectory:
                self._new_trajectory = False
                self.planner.cancel()
                self._dispatch = dispatch_indices(len(waypoints), self.planner.waypoint_stride)
                self._sent_at = None

        if waypoints is None or not self._dispatch:
            if self._empty_sets >= self.recovery_after and self._speed < 0.25:
                self._start_recovery()
            return

        idx = self._dispatch[0]
        if self._sent_at is None:
            self.planner.send_waypoint(tuple(waypoints[idx, :2]), frame_id=self.fixed_frame)
            self._sent_at = rospy.Time.now()
            return

        status = self.planner.status()
        reached = np.linalg.norm(position - waypoints[idx, :2]) <= self.reach_radius
        timed_out = (rospy.Time.now() - self._sent_at).to_sec() > self.waypoint_timeout
        if reached or status in (PlannerStatus.SUCCEEDED, PlannerStatus.FAILED) or timed_out:
            if not reached and status != PlannerStatus.SUCCEEDED:
                rospy.logwarn(f"[selector] skipping waypoint {idx} ({status.name}, {timed_out=})")
                self.planner.cancel()
            with self._lock:
                if not self._new_trajectory:  # progress refers to the trajectory dispatched
                    self.core.on_progress(idx + 1)
            self._dispatch.pop(0)
            self._sent_at = None

    def _abort_motion(self) -> None:
        self.planner.cancel()
        self._dispatch, self._sent_at, self._recovery_until = [], None, None

    def _start_recovery(self) -> None:
        """Rotate towards the free direction closest to the goal (rate-limited, bounded)."""
        dirs = [d - np.pi / 2 for d in self._free_dirs if d >= 0]  # 0 = straight ahead
        target = (
            min(dirs, key=lambda d: abs(d - self._goal_bearing)) if dirs else self._goal_bearing
        )
        self._recovery_vel = float(np.sign(target) or 1.0) * self.recovery_speed
        self._recovery_until = rospy.Time.now() + rospy.Duration(self.recovery_timeout)
        self._empty_sets = 0
        rospy.logwarn("[selector] no feasible candidates, rotating in place")

    def _recovery_tick(self) -> None:
        if rospy.Time.now() > self._recovery_until:
            stop_robot(self._cmd_vel_pub)
            self._recovery_until = None
            return
        rotate_in_place(self._cmd_vel_pub, self._recovery_vel)

    # ---------------------------------------------------------- visualisation
    def _publish_costmap(self, costmap, waypoints, cam_from_fixed, stamp) -> None:
        if self._costmap_pub.get_num_connections() + self._costmap_cmp_pub.get_num_connections():
            raw, cmp = cv2_to_imgmsg(colorize_labels(costmap.labels), self.camera_frame, stamp)
            self._costmap_pub.publish(raw)
            self._costmap_cmp_pub.publish(cmp)
        if waypoints is None or cam_from_fixed is None:
            return
        if self._annot_pub.get_num_connections() + self._annot_cmp_pub.get_num_connections():
            pixels, valid = project_to_image(
                cam_from_fixed.apply(waypoints),
                self.core.intrinsics,
                self.core.image_size,
                self.cfg.min_depth,
                self.cfg.camera_optical_frame,
            )
            img = colorize_labels(costmap.labels)
            raw, cmp = cv2_to_imgmsg(draw_paths(img, pixels, valid), self.camera_frame, stamp)
            self._annot_pub.publish(raw)
            self._annot_cmp_pub.publish(cmp)


def main() -> None:
    rospy.init_node("sem_navae_selector")
    node = SelectorNode()
    rate = rospy.Rate(float(param("execution/rate_hz", 10.0)))
    while not rospy.is_shutdown():
        node.step()
        rate.sleep()


if __name__ == "__main__":
    main()
