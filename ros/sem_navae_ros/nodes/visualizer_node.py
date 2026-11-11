#!/usr/bin/env python3
"""Visualise candidate trajectories: RViz markers, goal marker and the paths on the image."""

from typing import Optional

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Path
from sem_navae_ros.message_helpers import (
    camera_params,
    collisions_from_msg,
    cv2_to_imgmsg,
    imgmsg_to_rgb,
    lookup,
    param,
    patharray_to_numpy,
)
from sem_navae_ros.msg import PathArray
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

from sem_navae.selection import project_to_image

FREE, BLOCKED = (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)  # RGBA


class VisualizerNode:
    def __init__(self):
        self.robot_frame = param("frames/robot", "base_link")
        self.camera_frame = param("frames/camera", "camera_link")
        self.optical = bool(param("selection/camera_optical_frame", False))
        self.goal_radius = float(param("model/goal_th", 5.0))
        self.color = tuple(int(c) for c in param("annotation/color", [255, 0, 0]))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        camera_ns = param("topics/camera_ns", "camera")
        info = rospy.wait_for_message(camera_ns + "/camera_info", CameraInfo, timeout=30.0)
        self.K, self.image_size = camera_params(info)
        self._image: Optional[Image] = None

        annot = param("topics/annotation", "/sem_navae/annotated_trajectories")
        self._annot_pub = rospy.Publisher(annot, Image, queue_size=1)
        self._annot_cmp_pub = rospy.Publisher(annot + "/compressed", CompressedImage, queue_size=1)
        self._markers_pub = rospy.Publisher(
            param("topics/visualization", "/sem_navae/visualization"), MarkerArray, queue_size=1
        )
        self._goal_pub = rospy.Publisher("/sem_navae/goal_viz", Marker, queue_size=1)
        self._past_pub = rospy.Publisher("/sem_navae/past_path", Path, queue_size=1)

        rospy.Subscriber(
            camera_ns + "/image_raw", Image, self._image_cb, queue_size=1, buff_size=2**24
        )
        rospy.Subscriber(
            param("topics/trajs_filt", "/sem_navae/trajectories_filt"),
            PathArray,
            self._paths_cb,
            queue_size=1,
        )

    def _image_cb(self, msg: Image) -> None:
        self._image = msg

    def _paths_cb(self, msg: PathArray) -> None:
        if not msg.paths:
            return
        paths = patharray_to_numpy(msg)
        blocked = collisions_from_msg(msg, paths.shape[:2]).any(axis=1)
        self._publish_markers(msg, blocked)
        self._publish_goal(msg)
        if msg.past_path.poses:
            self._past_pub.publish(msg.past_path)
        if self._image is not None:
            self._publish_annotation(paths, blocked, msg.paths[0].header.stamp)

    def _publish_annotation(self, paths, blocked, stamp) -> None:
        if not self._annot_pub.get_num_connections() + self._annot_cmp_pub.get_num_connections():
            return
        cam_from_robot = lookup(self.tf_buffer, self.camera_frame, self.robot_frame, stamp)
        if cam_from_robot is None:
            return
        pixels, valid = project_to_image(
            cam_from_robot.apply(paths), self.K, self.image_size, optical_frame=self.optical
        )
        img = imgmsg_to_rgb(self._image).copy()
        for k in np.flatnonzero(~blocked):
            pts = pixels[k][valid[k]].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, self.color, 2)
        raw, cmp = cv2_to_imgmsg(img, self.camera_frame, stamp)
        self._annot_pub.publish(raw)
        self._annot_cmp_pub.publish(cmp)

    def _publish_markers(self, msg: PathArray, blocked: np.ndarray) -> None:
        markers = MarkerArray()
        for i, path in enumerate(msg.paths):
            m = Marker(
                header=path.header,
                ns="paths",
                id=i,
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                scale=Vector3(x=0.05, y=0.0, z=0.0),
                lifetime=rospy.Duration(5),
            )
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = BLOCKED if blocked[i] else FREE
            m.points = [p.pose.position for p in path.poses]
            markers.markers.append(m)
        self._markers_pub.publish(markers)

    def _publish_goal(self, msg: PathArray) -> None:
        m = Marker(header=msg.goal.header, ns="goal", id=0, type=Marker.LINE_STRIP)
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.2
        m.color.g, m.color.a = 1.0, 1.0
        cx, cy = msg.goal.pose.position.x, msg.goal.pose.position.y
        for a in np.linspace(0.0, 2.0 * np.pi, 51):
            m.points.append(
                Point(cx + self.goal_radius * np.cos(a), cy + self.goal_radius * np.sin(a), 0.0)
            )
        self._goal_pub.publish(m)


if __name__ == "__main__":
    rospy.init_node("sem_navae_visualizer")
    VisualizerNode()
    rospy.spin()
