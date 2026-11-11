#!/usr/bin/env python3
"""Sem-NaVAE generator: builds the observation (Sec. III-A1) and samples K trajectories.

Inputs are cached by subscribers and consumed by timers, so no callback blocks on
``wait_for_message``. Frame convention (``docs/dataset_format.md``): every input is in the
robot frame at the current time; past positions and velocities come from odometry and are
rotated by the current odometry yaw. The three stacked LiDAR scans are not
motion-compensated.
"""

from typing import List, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sem_navae_ros.message_helpers import cv2_to_imgmsg, numpy_to_path, param
from sem_navae_ros.msg import PathArray
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Empty, Header
from tf.transformations import euler_from_quaternion

from sem_navae.data.transforms import past_trajectory, voxelize_with_intensity
from sem_navae.inference import Generator
from sem_navae.keys import Keys
from sem_navae.utils.geometry import normalize_angle

GOAL_RANGE_NORM = 300.0  # [m] the goal distance input is divided by this (as in training)
POSE_SAMPLE_PERIOD = 0.5  # [s] past trajectory sampling (paper: last 5 s at 0.5 s)


class GeneratorNode:
    def __init__(self):
        self.robot_frame = param("frames/robot", "base_link")
        self.lidar_fov = param("model/lidar_fov", [-100, 100])  # [deg]
        self.n_paths = int(param("model/n_paths", 200))
        self.publish_heatmap = bool(param("model/publish_heatmap", True))
        self.publish_down_cloud = bool(param("annotation/publish_down_cloud", False))
        mode = param("model/localization_mode", "odom")  # odom | gps | amcl

        self.model = Generator(checkpoint=param("model/checkpoint"), device="cuda")
        cfg = self.model.cfg.model
        self.voxel, self.max_points = cfg.lidar_voxel_size, cfg.lidar_max_points
        self.lidar_norm, self.n_scans = cfg.lidar_norm_dist, cfg.lidar_num_scans
        self.n_poses, self.max_vel = cfg.past_traj_len + 1, cfg.max_vel

        self._cloud: Optional[PointCloud2] = None
        self._odom: Optional[Odometry] = None
        self._scans: List[np.ndarray] = []
        self._poses: List[Tuple[float, np.ndarray, float]] = []  # (stamp [s], xyz, yaw)
        self._goal_msg: Optional[PoseStamped] = None
        self._goal_obs: Optional[np.ndarray] = None
        self._yaw = 0.0

        goal_topic = param("topics/global_goal", "/sem_navae/global_goal_xy")
        rospy.Subscriber(goal_topic, PoseStamped, self._goal_cb)
        rospy.Subscriber(goal_topic + "_point", PointStamped, self._goal_cb)
        rospy.Subscriber(param("topics/done_signal", "/sem_navae/stop"), Empty, self._cancel_cb)
        rospy.Subscriber(
            param("topics/ouster", "/ouster/points"),
            PointCloud2,
            self._cloud_cb,
            queue_size=1,
            buff_size=2**24,
        )
        odom_topic = param("topics/odom", "/odom")
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=1)
        global_pose = {
            "odom": (odom_topic, Odometry),
            "gps": (param("topics/gps_odom", "/odometry/gps"), Odometry),
            "amcl": ("/amcl_pose", PoseWithCovarianceStamped),
        }[mode]
        rospy.Subscriber(*global_pose, self._global_pose_cb, queue_size=1)

        self._paths_pub = rospy.Publisher(
            param("topics/trajs_out", "/sem_navae/trajectories"), PathArray, queue_size=1
        )
        self._cloud_pub = rospy.Publisher(
            param("topics/down_cloud", "/sem_navae/downsampled_cloud"), PointCloud2, queue_size=1
        )
        self._heatmap_pub = rospy.Publisher("/sem_navae/heatmap", Image, queue_size=1)
        self._heatmap_cmp_pub = rospy.Publisher(
            "/sem_navae/heatmap/compressed", CompressedImage, queue_size=1
        )

        rospy.Timer(rospy.Duration(1.0 / 3.0), self._cloud_timer)
        rospy.Timer(rospy.Duration(POSE_SAMPLE_PERIOD), self._pose_timer)
        rospy.Timer(
            rospy.Duration(float(param("model/trajectory_gen_period", 2.0))), self._generate
        )
        rospy.loginfo(f"[generator] ready (goal input from {mode})")

    # ----------------------------------------------------------------- inputs
    def _cloud_cb(self, msg: PointCloud2) -> None:
        self._cloud = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    def _cloud_timer(self, _event) -> None:
        msg, self._cloud = self._cloud, None
        if msg is None:
            return
        pts = np.array(list(pc2.read_points(msg, ("x", "y", "z", "intensity"), skip_nans=True)))
        if pts.size == 0:
            return
        azimuth = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        keep = (azimuth >= self.lidar_fov[0]) & (azimuth <= self.lidar_fov[1])
        keep &= np.linalg.norm(pts[:, :3], axis=1) >= 1.0
        scan = voxelize_with_intensity(pts[keep], self.voxel, self.max_points)
        self._scans = (self._scans + [scan])[-self.n_scans :]
        if self.publish_down_cloud:
            self._publish_cloud(scan, msg.header)

    def _pose_timer(self, _event) -> None:
        msg = self._odom
        if msg is None:
            return
        stamp = msg.header.stamp.to_sec()
        if self._poses and stamp <= self._poses[-1][0]:
            return  # no new odometry since the last sample
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
        self._poses = (self._poses + [(stamp, np.array([p.x, p.y, 0.0]), yaw)])[-self.n_poses :]

    def _global_pose_cb(self, msg) -> None:
        pose = msg.pose.pose
        q = pose.orientation
        self._yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
        if self._goal_msg is None:
            return
        xy = np.array([pose.position.x, pose.position.y])
        g = np.array([self._goal_msg.pose.position.x, self._goal_msg.pose.position.y])
        rho = np.linalg.norm(g - xy) / GOAL_RANGE_NORM
        theta = normalize_angle(np.arctan2(g[1] - xy[1], g[0] - xy[0]) - self._yaw) / np.pi
        self._goal_obs = np.array([rho, theta], dtype=np.float32)

    def _goal_cb(self, msg) -> None:
        if isinstance(msg, PointStamped):
            goal = PoseStamped(header=msg.header)
            goal.pose.position = msg.point
            goal.pose.orientation.w = 1.0
            msg = goal
        self._goal_msg, self._goal_obs = msg, None
        p = msg.pose.position
        rospy.loginfo(f"[generator] new goal ({msg.header.frame_id}): [{p.x:.1f}, {p.y:.1f}]")

    def _cancel_cb(self, _msg: Empty) -> None:
        self._goal_msg, self._goal_obs = None, None

    # ------------------------------------------------------------- generation
    def _generate(self, _event) -> None:
        if (
            len(self._scans) < self.n_scans
            or len(self._poses) < self.n_poses
            or self._goal_obs is None
        ):
            return
        stamps = np.array([s for s, _, _ in self._poses])
        positions = np.stack([p for _, p, _ in self._poses])
        past = past_trajectory(stamps, positions, self._poses[-1][2], self.max_vel)

        lidar = np.stack(self._scans).astype(np.float32)
        lidar[..., :3] /= self.lidar_norm
        lidar[..., 3] = (lidar[..., 3] - lidar[..., 3].mean()) / (lidar[..., 3].std() + 1e-6)
        obs = {
            Keys.lidar: torch.from_numpy(lidar).unsqueeze(0),
            Keys.past_traj: torch.tensor(past, dtype=torch.float32).unsqueeze(0),
            Keys.goal: torch.from_numpy(self._goal_obs).unsqueeze(0),
        }
        trajs, heatmap = self.model.sample(obs, num_samples=self.n_paths)

        # Stamp with the observation time so the selector uses the matching tf.
        stamp = rospy.Time.from_sec(stamps[-1])
        msg = PathArray()
        msg.paths = [numpy_to_path(t, self.robot_frame, stamp) for t in trajs]
        msg.goal = self._goal_msg
        msg.goal_relative.data = self._goal_obs.tolist()
        msg.past_path = numpy_to_path(
            np.c_[past[:, :2], np.zeros(len(past))], self.robot_frame, stamp
        )
        self._paths_pub.publish(msg)

        if self.publish_heatmap and heatmap is not None:
            img = heatmap - min(heatmap.min(), 0.0)
            img = (255 * img / max(img.max(), 1e-6)).astype(np.uint8)
            raw, cmp = cv2_to_imgmsg(img, self.robot_frame, stamp)
            self._heatmap_pub.publish(raw)
            self._heatmap_cmp_pub.publish(cmp)

    def _publish_cloud(self, scan: np.ndarray, header: Header) -> None:
        fields = [
            PointField(name, 4 * i, PointField.FLOAT32, 1)
            for i, name in enumerate(("x", "y", "z", "intensity"))
        ]
        self._cloud_pub.publish(pc2.create_cloud(header, fields, scan[:, :4]))


if __name__ == "__main__":
    rospy.init_node("sem_navae_generator")
    GeneratorNode()
    rospy.spin()
