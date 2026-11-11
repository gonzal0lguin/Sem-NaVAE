#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import tf2_ros

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Vector3, Point
from std_msgs.msg import Header, Int16MultiArray, Float32MultiArray
from nav_msgs.msg import Path
from sem_navae_ros.msg import PathArray
from message_helpers import cv2_to_imgmsg, transform_and_project_paths, get_camera_params

from time import time

# THIS NODE DOES x THINGS:
# 1. Publishes the N generated trajectories, past path the goal and the selected trajectory
# 2. Projects the trajectories to the image plane and annotates the image with them


class WaypointProjector:
    def __init__(self):

        # TODO: make a better initialization
        camera_name           = rospy.get_param('topics/camera_ns', 'camera')
        self.trajs_topic      = rospy.get_param("topics/trajs_filt", "/sem_navae/trajectories")
        self.marker_topic     = rospy.get_param("topics/visualization", "/sem_navae/visualization")
        self.annotation_topic = rospy.get_param("topics/annotation", "/sem_navae/annotated_trajectories")
        self.imgarray_topic   = rospy.get_param("topics/img_array", "/sem_navae/image_array")
        self.selected_traj_topic   = rospy.get_param("topics/selected_traj", "/sem_navae/selected_traj")
        self._colors = rospy.get_param('annotation/color', [0, 0, 255])

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._camera_topic = camera_name + "/camera_info"
        self._image_topic  = camera_name + "/image_raw"  

        self._traj_sub    = rospy.Subscriber(self.trajs_topic, PathArray, self._waypoint_callback)
        # self._seltraj_sub = rospy.Subscriber(self.selected_traj_topic, SelectedPath, self._selected_callback)

        self._annotation_pub = rospy.Publisher(self.annotation_topic, Image, queue_size=1)
        self._annotationcmp_pub = rospy.Publisher(self.annotation_topic+'/compressed', CompressedImage, queue_size=1)

        self._vis_pub        = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self._path_pub       = rospy.Publisher(self.selected_traj_topic+'/path_viz', Path, queue_size=1)
        self._goal_pub       = rospy.Publisher("/sem_navae/goal_viz", Marker, queue_size=1)
        self.past_path_pub   = rospy.Publisher("/sem_navae/past_path", Path, queue_size=1)

        self._K, self._img_width, self.img_height = get_camera_params(self._camera_topic)  # Intrinsic matrix
       
    # for now were gonna use just a native Path message
    # def _selected_callback(self, msg: SelectedPath):
    #     self._path_pub.publish(msg.path)


    def _waypoint_callback(self, msg: PathArray):
        # to = time()
        pixel_points =  []
        distances = []
        ori_idx = []

        # NOTE: we ignore the order of the paths by distance (not implemented yet)
        # goal = np.array([msg.goal.pose.position.x, msg.goal.pose.position.y, 0])

        paths = [[[pose.pose.position.x, pose.pose.position.y, 0] for pose in path.poses] for path in msg.paths]

        pixel_points = transform_and_project_paths(self.tf_buffer, paths, self._K, "camera_link", "base_link")

        img = rospy.wait_for_message(self._image_topic, Image)
        img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)

        if len(distances) > 0:
            pixel_points, ori_idx = self.sort_paths(pixel_points, distances)
            annotated_img = self.annotate_image(img, pixel_points, [distances[i] for i in ori_idx])
            self._distance_pub.publish(Float32MultiArray(data=distances))
        else:
            annotated_img = self.annotate_image(img, pixel_points, balls=True)

        img_msg, imgcmp_msg = cv2_to_imgmsg(annotated_img, compressed=True)
        self._annotation_pub.publish(img_msg)
        self._annotationcmp_pub.publish(imgcmp_msg)

        self.visualize_paths_rviz(msg)
        self.visualize_goal(msg.goal)


    def annotate_image(self, image_ori, pixel_points, balls=False):      
        image =  image_ori.copy()

        for i, pixel_point in enumerate(pixel_points):
            pixel_point = pixel_point.astype(int)
            
            if balls:
                px = pixel_point[-1][0]
                if px < 20:
                    px += 20
                elif px > self._img_width - 20:
                    px -= 20

                cv2.circle(image, (px+5, pixel_point[-1][1]-15), 12, (255, 255, 255), -1)
                cv2.putText(image, str(i + 1), (px, pixel_point[-1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._colors, 2)
    
            for j in range(len(pixel_point) - 1):
                if pixel_point[j][0] >= image.shape[1] or pixel_point[j][1] >= image.shape[0] or \
                    pixel_point[j + 1][0] >= image.shape[1] or pixel_point[j + 1][1] >= image.shape[0]:
                    continue
                cv2.line(image, tuple(pixel_point[j]), tuple(pixel_point[j + 1]), self._colors, 2)
            
        return image


    def visualize_goal(self, goal):

        marker = Marker()
        marker.header.frame_id = "odom"  # Ensure it's in the correct reference frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # Creates a hollow outline
        marker.action = Marker.ADD
        marker.scale.x = 5  # Line width
        marker.color.r = 0.0  # Red
        marker.color.g = 1.0  # Yellow tint
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully visible
        marker.lifetime = rospy.Duration(0)  # Persistent

        # Circle parameters
        radius = 5.0  # Radius in meters
        num_points = 50  # Number of points for smoothness
        cx, cy, cz = goal.pose.position.x, goal.pose.position.y, 0  # Center of the circle

        # Generate circle points
        for i in range(num_points + 1):  # +1 to close the loop
            angle = 2.0 * np.pi * i / num_points
            p = Point()
            p.x = cx + radius * np.cos(angle)
            p.y = cy + radius * np.sin(angle)
            p.z = cz  # Keep it at the same height as the received pose
            marker.points.append(p)

        self._goal_pub.publish(marker)    


    def visualize_paths_rviz(self, paths_msg: PathArray, color=[1, 0, 1, 0]):
        """
        Visualize multiple paths as lines in RViz.

        Args:
            paths: List of Path messages, where each Path contains a list of PoseStamped points.
            color: List specifying the RGBA color of the lines.
        """
        marker_array = MarkerArray()
        header = Header()
        header.frame_id = "base_link"
        header.stamp = rospy.Time.now()

        ori_color = color
        try:
            height = paths_msg.collisions.layout.dim[0].size
            width  = paths_msg.collisions.layout.dim[1].size
            data_array = np.frombuffer(paths_msg.collisions.data, dtype=np.uint8)
            collisions = data_array.reshape(height, width).astype('bool')
        except:
            collisions = np.zeros((len(paths_msg.paths), 1), dtype=bool)

        for i, path in enumerate(paths_msg.paths):
            # Create a marker for each path
            marker = Marker(
                header=header,
                ns="paths",
                id=i,
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                scale=Vector3(x=0.1, y=0.05, z=0.0),  # Line width
                lifetime=rospy.Duration(5)
            )

            if collisions[i].any():
                color = [1, 1, 0, 0]
            else:
                color = ori_color
            # Set the color
            marker.color.a = color[0]
            marker.color.r = color[1]
            marker.color.g = color[2]
            marker.color.b = color[3]

            # Add points to the line strip
            for pose_stamped in path.poses:
                marker.points.append(pose_stamped.pose.position)

            marker_array.markers.append(marker)

        if len(paths_msg.past_path.poses) > 0:
            self.past_path_pub.publish(paths_msg.past_path)
            # # Create a marker for the past path
            # past_marker = Marker(
            #     header=header,
            #     ns="past_path",
            #     id=len(paths_msg.paths),
            #     type=Marker.LINE_STRIP,
            #     action=Marker.ADD,
            #     scale=Vector3(x=0.1, y=0.05, z=0.0),  # Line width
            #     lifetime=rospy.Duration(5)
            # )
            # past_marker.color.a = 1.0
            # past_marker.color.r = 0.0
            # past_marker.color.g = 0.0
            # past_marker.color.b = 1.0

            # for pose_stamped in paths_msg.past_path.poses:
            #     past_marker.points.append(pose_stamped.pose.position)
            # marker_array.markers.append(past_marker)
    
        # Publish the markers
        self._vis_pub.publish(marker_array)

    

# Main function
if __name__ == "__main__":
    rospy.init_node("waypoint_projector")

    projector = WaypointProjector()


    rospy.spin()