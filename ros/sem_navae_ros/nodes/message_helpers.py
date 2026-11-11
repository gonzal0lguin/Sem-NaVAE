import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import MultiArrayDimension
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from sem_navae_ros.msg import PathArray
import tf2_ros


from typing import Union, List

import cv2


###############################################
# IMAGING STUFF
###############################################

def get_camera_params(camera_topic: str) -> Union[np.ndarray, int]:
    # try:
    camera_info = rospy.wait_for_message(camera_topic, CameraInfo, timeout=5)
    K = np.array(camera_info.K).reshape(3, 3)

    img_width  = camera_info.width
    img_height = camera_info.height

    return K, img_width, img_height

    # except rospy.ROSException as e:
    #     rospy.logerr("Error while getting camera info: {}".format(e))
    #     return False


def cv2_to_imgmsg(cv2_img: np.ndarray, frame_id: str = "camera_link", rgb: bool = True, compressed: bool = True) -> Image:
    img_msg = Image()
    img_msg.header.stamp = rospy.Time.now()
    img_msg.header.frame_id = frame_id
    img_msg.height = cv2_img.shape[0]
    img_msg.width  = cv2_img.shape[1]
    img_msg.step   = cv2_img.shape[1] * 3 if rgb else cv2_img.shape[1]
    img_msg.encoding = "rgb8" if rgb else "mono8"
    img_msg.is_bigendian = 0
    img_msg.data = cv2_img.tobytes()

    img_compressed = CompressedImage
    if compressed:
        img_compressed.header.stamp = rospy.Time.now()
        img_compressed.format = "jpeg"

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] 
        success, encoded_img = cv2.imencode('.jpg', cv2_img, encode_param)

        if success:
            img_compressed.data = encoded_img.tobytes()
            
    return img_msg, img_compressed    


def imgmsg_to_cv2(imgmsg: Image) -> np.ndarray:
    return np.frombuffer(imgmsg.data, dtype=np.uint8).reshape(imgmsg.height, imgmsg.width, -1)


def humanize_costmap(costmap: np.ndarray) -> np.ndarray:
    colors = [[0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [127, 127, 127],
            [0, 0, 0]]
    
    out = np.zeros((costmap.shape[0], costmap.shape[1], 3), dtype=np.uint8)
    for class_index, color in enumerate(colors):
        out[costmap == class_index] = color
    
    return out


def annotate_costmap(image_ori: np.ndarray, pixel_points: np.ndarray, distances=None) -> np.ndarray:
    image = image_ori.copy()

    if distances is None:
        for i, pixel_point in enumerate(pixel_points):
            pixel_point = pixel_point.astype(int)
            for j in range(len(pixel_point)):
                if pixel_point[j][0] >= image.shape[1] or pixel_point[j][1] >= image.shape[0] \
                    or pixel_point[j][0] < 0 or pixel_point[j][1] < 0:
                    continue
                cv2.circle(image, tuple(pixel_point[j]), 5, (255, 0, 0), -1)
    else:
        # order by distnave
        pass

    return image


################################################
# PATHS AND EFFICIENT TRANSFORMS
################################################
def transform_to_frame(tf_buffer: tf2_ros.Buffer, waypoints: np.ndarray, target_frame: str, source_frame: str) -> np.ndarray:
    """Transform waypoints from source frame to target frame.
    Args:
        waypoints (np.ndarray): input paths of shape (N_paths, N_waypoints, 3) in the source frame.
    Returns:
        np.ndarray: transformed waypoints of shape (N_paths, N_waypoints, 3) in the target frame.
    """
    try:
        trans = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), timeout=rospy.Duration(1.0))
        
        # Extract transform translation and rotation
        translation = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ])
        rotation = np.array([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ])

        # Convert quaternion to rotation matrix
        q = rotation
        R = np.array([
            [1 - 2*(q[1]**2 + q[2]**2),   2*(q[0]*q[1] - q[2]*q[3]),       2*(q[0]*q[2] + q[1]*q[3])],
            [2*(q[0]*q[1] + q[2]*q[3]),   1 - 2*(q[0]**2 + q[2]**2),       2*(q[1]*q[2] - q[0]*q[3])],
            [2*(q[0]*q[2] - q[1]*q[3]),   2*(q[1]*q[2] + q[0]*q[3]),       1 - 2*(q[0]**2 + q[1]**2)]
        ])

        # Perform the transformation in vectorized form
        points = np.array(waypoints).T 
        transformed_points = R.dot(points) + translation[:, np.newaxis]
        return transformed_points.T  # shape: (K, N, 3)

    except tf2_ros.LookupException:
        rospy.logerr("Transformation lookup failed.")
        return np.empty((0, 3))


def project_to_image_plane(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D points in the camera frame to the image plane.
    Vectorized implementation.
    """
    points = np.asarray(points)
    x_c, y_c, z_c = points[:, 2], -points[:, 1], points[:, 0]

    z_c = np.where(z_c == 0, 1e-6, z_c)

    u = (K[0, 0] * y_c / z_c) + K[0, 2]
    v = -(K[1, 1] * x_c / z_c) + K[1, 2]

    return np.vstack((u, v)).T


def transform_and_project_paths(tf_buffer: tf2_ros.Buffer, 
                                paths: Union[List[np.ndarray], np.ndarray], 
                                K: np.ndarray,
                                target_frame: str = "camera_link", 
                                source_frame: str = "base_link") -> np.ndarray:
    """
    Transform and project multiple paths efficiently.
    paths: List of N arrays, each containing waypoints (N_waypoints, 3)
    Returns:
        projected_paths: array of shape (N_paths, N_waypoints, 2)
    """
    all_waypoints = np.vstack(paths)
    
    # Transform and project all waypoints at once
    transformed_waypoints = transform_to_frame(tf_buffer, all_waypoints, target_frame, source_frame)
    projected_waypoints   = project_to_image_plane(transformed_waypoints, K)

    # Recover individual paths from the concatenated results
    lengths = [len(p) for p in paths]
    cumulative_indices = np.cumsum(lengths)[:-1]  # Exclude last index for proper splitting
    projected_paths = np.split(projected_waypoints, cumulative_indices)

    return projected_paths


################################################
# ROBOT MOVEMENT METHODS
################################################

def stop_robot(pub: rospy.Publisher):
    pub.publish(Twist())

def rotate_in_place(pub: rospy.Publisher, vel: float):
    msg = Twist()
    msg.angular.z = vel

    pub.publish(msg)
    