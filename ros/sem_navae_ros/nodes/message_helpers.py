"""Message conversion and tf helpers shared by the Sem-NaVAE nodes."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from sem_navae.selection import RigidTransform


def param(name: str, default=None):
    """Read ``~name`` if set, else the global ``name`` (both launch styles work)."""
    if rospy.has_param("~" + name):
        return rospy.get_param("~" + name)
    return rospy.get_param(name, default)


# --------------------------------------------------------------------------- images
def camera_params(camera_info: CameraInfo) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Intrinsics ``K`` and ``(width, height)`` from a CameraInfo message."""
    return np.array(camera_info.K, dtype=float).reshape(3, 3), (
        camera_info.width,
        camera_info.height,
    )


def imgmsg_to_rgb(msg: Image) -> np.ndarray:
    """``sensor_msgs/Image`` (rgb8, bgr8, rgba8, bgra8 or mono8) to an RGB uint8 array."""
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding '{msg.encoding}'")
    img = img[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if msg.encoding.startswith("bgr"):
        img = img[..., [2, 1, 0]]
    return np.ascontiguousarray(img[..., :3])


def cv2_to_imgmsg(
    img: np.ndarray, frame_id: str = "camera_link", stamp: Optional[rospy.Time] = None
) -> Tuple[Image, CompressedImage]:
    """RGB (H, W, 3) or mono (H, W) uint8 array to raw and JPEG-compressed messages."""
    stamp = stamp or rospy.Time.now()
    rgb = img.ndim == 3
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "rgb8" if rgb else "mono8"
    msg.step = img.shape[1] * (3 if rgb else 1)
    msg.data = np.ascontiguousarray(img).tobytes()

    cmp = CompressedImage()
    cmp.header = msg.header
    cmp.format = "jpeg"
    ok, enc = cv2.imencode(".jpg", img[..., ::-1] if rgb else img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if ok:
        cmp.data = enc.tobytes()
    return msg, cmp


CLASS_COLORS = np.array(
    [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [127, 127, 127],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels % len(CLASS_COLORS)]


def draw_paths(
    image: np.ndarray, pixels: np.ndarray, valid: np.ndarray, color=(255, 0, 0), radius: int = 4
) -> np.ndarray:
    """Draw projected waypoints ``(K, N, 2)`` (only the valid ones)."""
    out = image.copy()
    for u, v in pixels[valid].astype(int):
        cv2.circle(out, (int(u), int(v)), radius, color, -1)
    return out


# ----------------------------------------------------------------------------- tf
def transform_from_msg(tf_msg) -> RigidTransform:
    t, q = tf_msg.transform.translation, tf_msg.transform.rotation
    return RigidTransform.from_quaternion([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def lookup(
    buffer: tf2_ros.Buffer, target: str, source: str, stamp=None, timeout: float = 0.2
) -> Optional[RigidTransform]:
    """``T_target_from_source`` at ``stamp`` (latest if None); None (with a warning) on failure."""
    try:
        msg = buffer.lookup_transform(
            target, source, stamp or rospy.Time(0), rospy.Duration(timeout)
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as exc:
        rospy.logwarn_throttle(2.0, f"tf {source} -> {target} unavailable: {exc}")
        return None
    return transform_from_msg(msg)


# -------------------------------------------------------------------------- paths
def patharray_to_numpy(msg) -> np.ndarray:
    """``PathArray.paths`` to a ``(K, N, 3)`` array."""
    return np.array(
        [
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in path.poses]
            for path in msg.paths
        ],
        dtype=float,
    ).reshape(len(msg.paths), -1, 3)


def collisions_from_msg(msg, shape: Tuple[int, int]) -> np.ndarray:
    """Per-waypoint collision flags ``(K, N)``; all free if the filter did not fill them."""
    data = np.frombuffer(bytes(msg.collisions.data), dtype=np.uint8)
    if data.size != shape[0] * shape[1]:
        return np.zeros(shape, dtype=bool)
    return data.reshape(shape).astype(bool)


def numpy_to_path(points: np.ndarray, frame_id: str, stamp: rospy.Time) -> Path:
    path = Path()
    path.header.frame_id, path.header.stamp = frame_id, stamp
    for x, y, *rest in points:
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.position.z = float(rest[0]) if rest else 0.0
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


def dispatch_indices(num_waypoints: int, stride: int) -> List[int]:
    """Waypoints sent to the local planner: every ``stride``-th one, always the last."""
    idx = list(range(stride - 1, num_waypoints, max(1, stride)))
    if not idx or idx[-1] != num_waypoints - 1:
        idx.append(num_waypoints - 1)
    return idx


# ------------------------------------------------------------------------ motion
def stop_robot(pub: rospy.Publisher) -> None:
    pub.publish(Twist())


def rotate_in_place(pub: rospy.Publisher, angular_vel: float) -> None:
    msg = Twist()
    msg.angular.z = angular_vel
    pub.publish(msg)
