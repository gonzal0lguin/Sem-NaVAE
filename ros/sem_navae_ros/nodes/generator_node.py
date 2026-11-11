#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty, Header, MultiArrayDimension

from sem_navae_ros.msg import PathArray
from sem_navae.inference import Generator
from sem_navae.data.transforms import voxelize_with_intensity
from sem_navae.utils.geometry import normalize_angle, rotate_pointcloud
from message_helpers import cv2_to_imgmsg, transform_to_frame

import time
import torch
import numpy as np
from typing import Any, Union

# np.set_printoptions(precision=2)

class GeneratorNode:
    def __init__(self):
        self.name = "Sem-NaVAE generator"

        model_pth                 = rospy.get_param("model/checkpoint")
        self._lidar_fov           = rospy.get_param("model/lidar_fov", [-100, 100])  # [deg]
        self._max_wpts            = rospy.get_param("model/max_wpts", 12)
        self._n_paths             = rospy.get_param("model/n_paths", 20)
        self._traj_gen_period     = rospy.get_param("model/trajectory_gen_period", 1.0)
        self._goal_th             = rospy.get_param("model/goal_th", 3.)
        self._localization_mode   = rospy.get_param("model/localization_mode", "odom") # odom, gps --or amcl for tests only--        
        self._publish_heatmap     = rospy.get_param("model/publish_heatmap",    True)
        self._publish_down_cloud  = rospy.get_param("annotation/publish_down_cloud", True)

        self.odom_topic           = rospy.get_param("topics/odom",        "/odom")
        self.gps_topic            = rospy.get_param("topics/gps_odom",    "/odometry/gps")
        self.ouster_topic         = rospy.get_param("topics/ouster",      "/ouster/points")
        self.global_goal_topic    = rospy.get_param("topics/global_goal", "/sem_navae/global_goal_xy")
        self.out_trajs_topics     = rospy.get_param("topics/trajs_out",   "/sem_navae/trajectories_raw")
        self.downcloud_topic      = rospy.get_param("topics/down_cloud",  "/sem_navae/downsampled_cloud")
        
        self._accum_pcl:        list            = []
        self._accum_vel:        list            = []
        self._goal_observation: np.ndarray = None
        self._global_goal  = None
        self._odom_pose    = None
        self._pose_xy      = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.initialize_model(model_pth)

        self._goal_sub      = rospy.Subscriber(self.global_goal_topic, PoseStamped, self._goal_cb)
        self._goalpoint_sub = rospy.Subscriber(self.global_goal_topic+'_point', PointStamped, self._goal_cb)
        self._cancel_sub    = rospy.Subscriber('/sem_navae/stop', Empty, self._cancel_cb)

        self._trajectories_pub = rospy.Publisher(self.out_trajs_topics, PathArray, queue_size=1)

        if self._localization_mode == "gps":
            self._gps_sub = rospy.Subscriber(self.gps_topic, Odometry, self._global_pos_cb)
        elif self._localization_mode == "amcl":
            self._amcl_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self._global_pos_cb)
        elif self._localization_mode == "odom":
            self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._global_pos_cb)
        if self._publish_down_cloud:
            self._pcl_pub  = rospy.Publisher(self.downcloud_topic, PointCloud2, queue_size=1)
        if self._publish_heatmap:
            self._heatmap_pub = rospy.Publisher('/sem_navae/heatmap', Image, queue_size=1)
            self._heatmapcmp_pub = rospy.Publisher('/sem_navae/heatmap/compressed', CompressedImage, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / 3.0), self._process_pointcloud)
        rospy.Timer(rospy.Duration(1.0 / 2.0), self._process_velocities)
        rospy.Timer(rospy.Duration(self._traj_gen_period), self._timer_callback) # main generation cb

        rospy.loginfo(f"{[self.name]} Using reaching distance from: {self._localization_mode}.")        
        rospy.loginfo(f"{[self.name]} Model initialized.")



    def initialize_model(self, model_pth):
        self._model = Generator(checkpoint=model_pth, device='cuda')
        cfg = self._model.cfg.model

        self._lidar_downsample_vx = cfg.lidar_voxel_size
        self._lidar_max_points    = cfg.lidar_max_points
        self._lidar_norm_dst      = cfg.lidar_norm_dist
        self._lidar_num           = cfg.lidar_num_scans
        self._vel_num             = cfg.past_traj_len + 1

        self._max_wpts = min(self._max_wpts, cfg.num_waypoints)


    def _process_pointcloud(self, event) -> None:
        msg = rospy.wait_for_message(self.ouster_topic, PointCloud2)

        points  = self._pointcloud2_to_array(msg)
        downpcd = voxelize_with_intensity(points, self._lidar_downsample_vx, self._lidar_max_points)

        self._accum_pcl.append(downpcd)
        if len(self._accum_pcl) > self._lidar_num:
            self._accum_pcl.pop(0)

        if self._publish_down_cloud:
            self._publish_viz_cloud(downpcd)


    def _pointcloud2_to_array(self, pc2_msg: PointCloud2) -> np.ndarray:
        pcl = np.array(list(pc2.read_points(pc2_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))

        azimuth = np.arctan2(pcl[:, 1], pcl[:, 0])  # y, x
        ranges = np.linalg.norm(pcl, axis=1)

        mask = (azimuth >= np.radians(self._lidar_fov[0])) & \
               (azimuth <= np.radians(self._lidar_fov[1]))  & \
               (ranges >= 1) 
        
        pcl = pcl[mask]
        # Not needed for newer models trained with datasets > v2.0
        # pcl[:, :3] = rotate_pointcloud(pcl[:, :3], np.pi/2) # rotate only points

        return pcl  # (N, 4)


    def _publish_viz_cloud(self, pcd: np.ndarray):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'os_sensor'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, pcd[:, :4])

        self._pcl_pub.publish(cloud_msg)


    def _process_velocities(self, event):
        msg = rospy.wait_for_message(self.odom_topic, Odometry)

        positions  = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, 0.0]) # this positions are in a global frame

        self._accum_vel.append(positions)
        if len(self._accum_vel) > self._vel_num:
            self._accum_vel.pop(0)

    
    def _global_pos_cb(self, msg: Union[Odometry, PoseWithCovarianceStamped]) -> None:
        self._odom_pose = msg.pose.pose
        self._pose_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        if self._global_goal is not None:
            q = msg.pose.pose.orientation
            rho = np.linalg.norm(self._global_goal - self._pose_xy) / 300.0
            robot_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
            theta = np.arctan2(self._global_goal[1] - self._pose_xy[1], self._global_goal[0] - self._pose_xy[0]) - robot_theta
            theta = normalize_angle(theta) / np.pi

            self._goal_observation = np.array([rho, theta], dtype=np.float32)    


    def _goal_cb(self, goal: Union[PoseStamped, PointStamped]) -> None:
        if isinstance(goal, PoseStamped):
            self._global_goal_msg = goal
            self._global_goal     = np.array([goal.pose.position.x, goal.pose.position.y])
        else:
            self._global_goal = np.array([goal.point.x, goal.point.y])
            self._global_goal_msg = PoseStamped()
            self._global_goal_msg.pose.position = goal.point
            self._global_goal_msg.header = goal.header

        rospy.loginfo(f'New goal received at ({goal.header.frame_id}): [{self._global_goal[0]:.1f}, {self._global_goal[1]:.1f}]')


    def _cancel_cb(self, msg: Empty):
        self._global_goal = None
        self._goal_observation = None


    def _timer_callback(self, event):
        if len(self._accum_pcl) != self._lidar_num or \
           len(self._accum_vel) != self._vel_num or \
           self._goal_observation is None:
            return
        
        # TODO: what the hell man
        obs = self.build_observation()
        trajs, heatmap = self._model.sample(obs, num_samples=self._n_paths)
        # trajs = trajs[..., [1, 0]] # flip xy 
        # trajs[:, :, 1] = -trajs[:, :, 1]
        trajectories = trajs.copy()
        # trajectories = np.zeros((self._n_paths, self._max_wpts, 2))
        # trajectories[:] = trajs[:, :self._max_wpts]

        trajs_msg = self.build_patharray_message(trajectories, 
                                                 [], 
                                                 self._global_goal_msg, 
                                                 obs['goal'].squeeze().cpu().numpy().tolist(),
                                                 obs['previous_path'].squeeze().cpu().numpy())

        self._trajectories_pub.publish(trajs_msg)

        if self._publish_heatmap:
            if heatmap.min() < 0:
                heatmap -= heatmap.min()
            heatmap *= 255
            heatmap, heatmap_cmp = cv2_to_imgmsg(heatmap.astype(np.uint8), rgb=False, compressed=True)
            self._heatmap_pub.publish(heatmap)
            self._heatmapcmp_pub.publish(heatmap_cmp)


    def build_observation(self):
        # Normalize lidar data
        lidar_np = np.stack(self._accum_pcl)
        lidar_np[:, :, :3] = lidar_np[:, :, :3] / self._lidar_norm_dst
        lidar_np[:, :, 3]  = (lidar_np[:, :, 3] - lidar_np[:, :, 3].mean()) / (lidar_np[:, :, 3].std() + 1e-6)

        # Transform past poses to local frame and calculate velocities
        poses_np = np.stack(self._accum_vel) # convert to base link shouldnt i? instead of just subtracting last pos
        vx = np.gradient(poses_np[:, 0], .5)[1:] / 2.0
        vy = np.gradient(poses_np[:, 1], .5)[1:] / 2.0 # normalize by max speed

        poses_np[:, :2] -= poses_np[-1, :2] # is this actually correct? kinda, doest account for rotation but works bc it was trained like this

        posestot = np.concatenate((poses_np[1:, :2], vx[:, np.newaxis], vy[:, np.newaxis]), axis=-1)

        # print("PAST TRAJ: ")
        # print(posestot)
        # print("lidar: ", lidar_np.max(), lidar_np.min(), lidar_np.std(), lidar_np.mean())
        # print("goal: " )
        # print(self._goal_observation)
        # print("\n-----------------------------\n")

        lidar       = torch.tensor(lidar_np, dtype=torch.float32).unsqueeze(0)
        poses       = torch.tensor(posestot, dtype=torch.float32).unsqueeze(0)
        goal        = torch.from_numpy(self._goal_observation).unsqueeze(0)

        return {'lidar': lidar, 'previous_path': poses, 'goal': goal,
                'path': None, 'png': None, 'all_paths': None,
                'last_poses': None, 'local_map': None, 'vel': None}


    def build_patharray_message(self, trajectories, collisions=[], goal=PoseStamped(), goal_polar=None, past_path=None):
        path_array_msg = PathArray()     
        if len(collisions) > 0:   
            path_array_msg.collisions.data = collisions.astype(np.uint8).flatten().tolist()
            
            path_array_msg.collisions.layout.dim.append(MultiArrayDimension(label="height", size=collisions.shape[0], stride=collisions.shape[1]))
            path_array_msg.collisions.layout.dim.append(MultiArrayDimension(label="width", size=collisions.shape[1], stride=1))
        else:
            path_array_msg.collisions.data = collisions
        path_array_msg.goal = goal

        if goal_polar is not None:
            path_array_msg.goal_relative.data = goal_polar

        for traj in trajectories:
            path_msg = Path()
            path_msg.header.frame_id = "base_link"
            path_msg.header.stamp = rospy.Time.now()

            for sub_traj in traj:
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position.x = sub_traj[0]
                pose.pose.position.y = sub_traj[1]
                pose.pose.orientation.x = 0
                pose.pose.orientation.y = 0
                pose.pose.orientation.z = 0
                pose.pose.orientation.w = 1
                path_msg.poses.append(pose)

            path_array_msg.paths.append(path_msg)

        if past_path is not None:
            q = self._odom_pose.orientation
            robot_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
            past_path = rotate_pointcloud(past_path[:, :3], -robot_theta) # vx used as a dummy z 
            path_msg = Path()
            path_msg.header.frame_id = "base_link"
            path_msg.header.stamp = rospy.Time.now()

            for sub_traj in past_path[:, :2]:
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position.x = sub_traj[0]
                pose.pose.position.y = sub_traj[1]
                pose.pose.orientation.x = 0
                pose.pose.orientation.y = 0
                pose.pose.orientation.z = 0
                pose.pose.orientation.w = 1
                path_msg.poses.append(pose)
            path_array_msg.past_path = path_msg

        return path_array_msg



if __name__ == '__main__':
    rospy.init_node("sem_navae_generator", anonymous=True)

    navae_server = GeneratorNode()
    rospy.spin()
