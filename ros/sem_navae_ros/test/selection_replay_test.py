#!/usr/bin/env python3
"""rostest: switching decisions of selector_node on a synthetic sequence.

Two candidates, LEFT (y = +0.8 m) and RIGHT (y = -0.8 m). Class labels follow the order of
selection/classes in sem_navae.yaml: 0 = pavement (cost 0), 2 = grass (2), 5 = person (3).
"""

import unittest

import numpy as np
import rospy
import rostest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sem_navae_ros.msg import PathArray
from sensor_msgs.msg import CameraInfo, Image

W, H = 100, 80
PAVEMENT, GRASS, PERSON = 0, 2, 5
X = np.arange(2.0, 8.0)


def _path(y, stamp):
    path = Path()
    path.header.frame_id, path.header.stamp = "base_link", stamp
    for x in X:
        pose = PoseStamped(header=path.header)
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


class SelectionReplay(unittest.TestCase):
    def setUp(self):
        self.selected = []
        rospy.Subscriber("/sem_navae/selected_traj", Path, self.selected.append)
        info = CameraInfo(width=W, height=H)
        info.header.frame_id = "camera_link"
        info.K = [100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0]
        self.info_pub = rospy.Publisher(
            "/test/camera/camera_info", CameraInfo, queue_size=1, latch=True
        )
        self.info_pub.publish(info)
        self.labels_pub = rospy.Publisher("/test/labels", Image, queue_size=1)
        self.paths_pub = rospy.Publisher("/sem_navae/trajectories_filt", PathArray, queue_size=1)

    def publish(self, left_label, right_label):
        now = rospy.Time.now()
        labels = np.zeros((H, W), dtype=np.uint8)
        labels[:, : W // 2], labels[:, W // 2 :] = left_label, right_label
        img = Image(height=H, width=W, encoding="mono8", step=W, data=labels.tobytes())
        img.header.stamp, img.header.frame_id = now, "camera_link"
        self.labels_pub.publish(img)
        rospy.sleep(0.2)
        msg = PathArray()
        msg.paths = [_path(0.8, now), _path(-0.8, now)]
        msg.goal.header.frame_id = "odom"
        msg.goal.pose.position.x = 30.0
        msg.goal.pose.orientation.w = 1.0
        self.paths_pub.publish(msg)

    def wait_selection(self, count, left_label, right_label, timeout):
        end = rospy.Time.now() + rospy.Duration(timeout)
        while len(self.selected) < count and rospy.Time.now() < end and not rospy.is_shutdown():
            self.publish(left_label, right_label)
            rospy.sleep(0.5)
        return len(self.selected) >= count

    def test_switching_sequence(self):
        # 1. Grass left, pavement right -> RIGHT is adopted (stored in odom).
        self.assertTrue(self.wait_selection(1, GRASS, PAVEMENT, timeout=60.0))
        first = self.selected[0]
        self.assertEqual(first.header.frame_id, "odom")
        self.assertAlmostEqual(first.poses[-1].pose.position.y, -0.8, places=3)

        # 2. Same scene again: within the hysteresis -> no new selection.
        for _ in range(6):
            self.publish(GRASS, PAVEMENT)
            rospy.sleep(0.5)
        self.assertEqual(len(self.selected), 1)

        # 3. A person appears on the right, the left is now pavement -> switch to LEFT.
        self.assertTrue(self.wait_selection(2, PAVEMENT, PERSON, timeout=15.0))
        self.assertAlmostEqual(self.selected[-1].poses[-1].pose.position.y, 0.8, places=3)


if __name__ == "__main__":
    rospy.init_node("selection_replay_test")
    rostest.rosrun("sem_navae_ros", "selection_replay", SelectionReplay)
