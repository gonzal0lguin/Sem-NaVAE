#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import cv2
import torch

from sensor_msgs.msg import Image, Joy, CompressedImage # joy to override commands
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Path, Odometry
from sem_navae_ros.msg import PathArray # , Debug
from message_helpers import (get_camera_params,
                             annotate_costmap, 
                             imgmsg_to_cv2, 
                             cv2_to_imgmsg, 
                             humanize_costmap,
                             transform_to_frame,
                             transform_and_project_paths,
                             stop_robot, rotate_in_place)

from sem_navae.utils.geometry import get_path_dist_to_goal, normalize_angle
from sem_navae_ros.local_planners import PlannerStatus, load_local_planner

from typing import Union, List


class TrajectorySelector:
    def __init__(self):

        # TODO: make a better initialization
        camera_name           = rospy.get_param('topics/camera_ns', "camera")
        self.trajs_topic      = rospy.get_param("topics/trajs_filt",    "/sem_navae/trajectories_filt")
        self.marker_topic     = rospy.get_param("topics/visualization", "/sem_navae/visualization")
        self.annotation_topic = rospy.get_param("topics/annotation",    "/sem_navae/annotated_trajectories")
        self.imgarray_topic   = rospy.get_param("topics/img_array",     "/sem_navae/image_array")
        self.done_sig_topic   = rospy.get_param("topics/done_signal",   "/sem_navae/stop")
        self.costmap_topic    = rospy.get_param("topics/costmap",       "/sem_navae/costmap")
        self.selection_topic  = rospy.get_param("topics/selection",     "/sem_navae/selected_traj")
        self.odom_topic       = rospy.get_param("topics/odom",          "/odom")
        self.gps_topic        = rospy.get_param("topics/gps",          "/odometry/gps")
        
        self._localization_mode = rospy.get_param("model/localization_mode", "odom")
        self._max_wpts          = rospy.get_param("model/max_wpts", 12)
        self._n_paths           = rospy.get_param("model/n_paths", 3)
        self._goal_th           = rospy.get_param("model/goal_th", 4) 

        self._local_planner_spec   = rospy.get_param("local_planner/plugin", "move_base")
        self._local_planner_params = rospy.get_param("local_planner/params", {})

        self._camera_topic = camera_name + "/camera_info"
        self._image_topic  = camera_name + "/image_raw"  

        self._global_goal = None
        self._local_goal  = None

        self._update_strat = rospy.get_param("~selection/strategy", 'barebones')
        self._gamma  = rospy.get_param("~selection/discount_factor", 0.9)
        self._beta   = rospy.get_param("~selection/goal_dst_w", 0.02)
        self._beta2  = rospy.get_param("~selection/goal_angle_w", 0.5)
        self._delta  = rospy.get_param("~selection/collisions_w", 1)
        self.improvement_eps  = rospy.get_param("~selection/improvement_eps", 0.5)

        constraints = rospy.get_param("~selection/constraints") #["a paved road", "trees", "grass", "sky", "wall", "stair", "person", "hole"]
        self.C_unk  = rospy.get_param("~selection/unknown_cost", 1)

        self._constraints = list(constraints.keys())
        self._costs = np.array(list(constraints.values())) #np.array([0, 10, 3, 30, 20, 15, 40, 6])
        
        self._joy_override = False
        self._ignore_semantics = False
        self._current_trajectory = {}
        self._current_wpt = 0
        self._current_free_dir = -1
        self._done_signal = True
        self._recovery_signal = False
        self._last_cost   = np.infty
        self._alignment_cnt = 0
        self.start = True
        self.robot_frame  = "base_link"
        self.global_frame = "map" if self._localization_mode == "amcl" else "odom"

        self._current_vel = None
        self._last_vel_stamp = 0
        self._pose_xy = np.zeros(2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # NOTE: queue of 1 important to process the latest and not the queued! clip runs at less than 30Hz
        self._traj_sub        = rospy.Subscriber(self.trajs_topic, PathArray, self._waypoint_callback)
        self._joy_sub         = rospy.Subscriber('/joy', Joy, self._joy_cb) # just to avoid inminent crashes or disble motors

        self._costmap_pub        = rospy.Publisher(self.costmap_topic, Image, queue_size=1)
        self._pathon_costmap_pub = rospy.Publisher('/sem_navae/path_on_costmap', Image, queue_size=1)
        self._costmapcmp_pub        = rospy.Publisher(self.costmap_topic+'/compressed', CompressedImage, queue_size=1)
        self._pathon_costmapcmp_pub = rospy.Publisher('/sem_navae/path_on_costmap/compressed', CompressedImage, queue_size=1)

        self._selected_traj_pub  = rospy.Publisher(self.selection_topic, Path, queue_size=1)

        self._rcost_pub       = rospy.Publisher("tmp/running_cost", Float32, queue_size=1)
        self._acost_pub       = rospy.Publisher("tmp/actual_min_cost", Float32, queue_size=1)
        # self._ntrajs_debug    = rospy.Publisher("tmp/new_debug", Debug, queue_size=1)
        # self._ctrajs_debug    = rospy.Publisher("tmp/curr_debug", Debug, queue_size=1)
        self._done_signal_pub  = rospy.Publisher(self.done_sig_topic, Empty, queue_size=1)
        self._cmd_vel_pub      = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        if self._localization_mode == "gps":
            self._gps_sub = rospy.Subscriber(self.gps_topic, Odometry, self._global_pos_cb)
        elif self._localization_mode == "amcl":
            self._gps_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self._global_pos_cb)
        elif self._localization_mode == "odom":
            self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._global_pos_cb)

        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._vel_cb)

        self._K, self._img_width, self.img_height = get_camera_params(self._camera_topic)  # Intrinsic matrix
        self.init_selector() 

        self._local_planner = load_local_planner(self._local_planner_spec, self._local_planner_params)
        if not self._local_planner.wait_until_ready(timeout=5.0):
            rospy.logwarn(f"[Selector]: local planner '{self._local_planner_spec}' is not ready yet.")
        rospy.loginfo(f"[Selector]: Using local planner: {self._local_planner_spec}")
        rospy.loginfo("[Selector]: Initialized selector model.")

    
    def init_selector(self):
        if 'clip' in self._update_strat:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            self._clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self._clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            self._clip_model.to(torch.device('cuda'))
            self._clip_model.eval()
            rospy.sleep(5) # time to init the model

            rospy.Timer(rospy.Duration(0.4), self._fpvimg_cb) # 2.5Hz 
            if self._update_strat == 'clip_running_cost':
                self._perform_trajectory_update = self._perform_clip_update
            if self._update_strat == 'clip_periodic_cost':
                self._perform_trajectory_update = self._perform_periodic_update

        elif self._update_strat == 'barebones':
            # This method just deletes the paths that have collisions and chooses the closest to the goal.
            self._perform_trajectory_update = self._perform_barebones_update

        else:
            raise NotImplementedError(f"Update strategy {self._update_strat} not implemented.")


    def _joy_cb(self, msg: Joy):
        if msg.buttons[4]:
            self._joy_override = True
        else:
            self._joy_override = False


    def _vel_cb(self, msg: Odometry):
        self._current_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])


    def _global_pos_cb(self, msg: Union[Odometry, PoseWithCovarianceStamped]) -> None:
        self._odom_pose = msg.pose.pose
        # self._last_pose_xy = self._pose_xy.copy() 
        self._pose_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # self._current_vel = (self._pose_xy - self._last_pose_xy) / (msg.header.stamp.secs / 1e9 - self._last_vel_stamp)
        # print(np.linalg.norm(self._current_vel), msg.header.stamp.nsecs / 1e9 - self._last_vel_stamp)
        # self._last_vel_stamp = msg.header.stamp.nsecs / 1e9


        if self._global_goal is not None:
            q = msg.pose.pose.orientation
            rho = np.linalg.norm(self._global_goal - self._pose_xy)
            robot_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
            theta = np.arctan2(self._global_goal[1] - self._pose_xy[1], self._global_goal[0] - self._pose_xy[0]) - robot_theta
            theta = normalize_angle(theta)

            self._goal_observation = np.array([rho, theta], dtype=np.float32)   


    def _fpvimg_cb(self, event):
        imgmsg = rospy.wait_for_message(self._image_topic, Image, timeout=5.0)
        self._make_costmap(imgmsg_to_cv2(imgmsg))
        self.update_current_cost()

    def _waypoint_callback(self, msg: PathArray):

        paths = np.array([[[pose.pose.position.x, pose.pose.position.y, 0] for pose in path.poses] for path in msg.paths]) # KxNx3
        goal  = np.array([msg.goal.pose.position.x, msg.goal.pose.position.y])
        collisions = np.frombuffer(msg.collisions.data, dtype=np.uint8).reshape((self._n_paths, self._max_wpts)).astype('bool')

        if (goal != self._global_goal).all():
            self._done_signal = False
        self._global_goal = goal
        self._local_goal  = msg.goal_relative.data
        self._current_free_dir = msg.free_dir.data

        if not self._done_signal:
            # 1. remove collisions
            not_collision_id = (collisions == 0).all(axis=1) # all waypoints are not collisions. shape K
            paths = paths[not_collision_id]

            if len(paths) == 0:
                # trigger the counter for the recovery behaviour
                rospy.logwarn("[Selector]: No feasible paths found, skipping update.")
                self._alignment_cnt += 1
                return

            self._perform_trajectory_update(paths, msg, goal) # in whta frame is the goal?


    def _perform_barebones_update(self, paths: np.ndarray, paths_msg: PathArray, goal: np.ndarray):
        # Take in the filtered trajectories, calculate the goal distance costs
        # update if better than EPS w/r to current path
        local_goal = transform_to_frame(self.tf_buffer, np.concatenate((goal, np.zeros(1))).reshape(1, -1), self.robot_frame, self.global_frame).squeeze()[:2]
        new_costs = self._get_distance_cost(paths, local_goal) # must be in the same frame AND THEY ARE NOT IN HERE BC PATH IS STILL IN THE LOCAL FRAME FKN TONTO WEON
        # print("Last best cost: ", self._last_cost)
        # print("Cadidates: ", new_costs, "id ", np.argmin(new_costs))
        # print("update: ", self._last_cost - min(new_costs) > self.improvement_eps)
        self._current_trajectory['free_dir'] = paths_msg.free_dir.data # independent of update or not

        new_min_cost = min(new_costs)
        if self._current_trajectory.get('cost', np.infty) - new_min_cost > self.improvement_eps: 
            rospy.loginfo(f"Updating due to new best cost: {self._last_cost} -> {new_min_cost:.2f}")
            best_path_id = np.argmin(new_costs)

            T_optim = paths_msg.paths[best_path_id]
            self._selected_traj_pub.publish(T_optim)

            self._current_trajectory['traj_odom'] = transform_to_frame(self.tf_buffer, paths[best_path_id], "odom", "base_link")
            self._current_trajectory['traj']      = paths[best_path_id]
            self._current_trajectory['cost']      = new_min_cost
            self._update_signal = True
        
        else:
            pass # continue the loop until (1) get a better traj or (2) finish previous traj

    
    def _perform_periodic_update(self, paths: np.ndarray, paths_msg: PathArray, goal: np.ndarray):
        # Take in the filtered trajectories, calculate the goal distance costs
        # update EVERY TIME taking the minimum cost, at a fixed 'traj_gen_freq' frecuency.
        local_goal = transform_to_frame(self.tf_buffer, np.concatenate((goal, np.zeros(1))).reshape(1, -1), self.robot_frame, self.global_frame).squeeze()[:2]
        new_costs = self._get_distance_cost(paths, local_goal) # must be in the same frame AND THEY ARE NOT IN HERE BC PATH IS STILL IN THE LOCAL FRAME FKN TONTO WEON

        self._current_trajectory['free_dir'] = paths_msg.free_dir.data # independent of update or not

        new_min_cost = min(new_costs)
        rospy.loginfo(f"Updating due to new best cost: {self._last_cost} -> {new_min_cost:.2f}")
        best_path_id = np.argmin(new_costs)

        T_optim = paths_msg.paths[best_path_id]
        self._selected_traj_pub.publish(T_optim)

        self._current_trajectory['traj_odom'] = transform_to_frame(self.tf_buffer, paths[best_path_id], "odom", "base_link")
        self._current_trajectory['traj']      = paths[best_path_id]
        self._current_trajectory['cost']      = new_min_cost
        self._update_signal = True
    

    def _perform_clip_update(self, paths: np.ndarray, paths_msg: PathArray, goal: np.ndarray):

        img = self._last_cmap.copy()

        new_costs, costs_lst, _ = self._calculate_path_cost_full(img, paths, goal, source_frame=self.robot_frame)
        new_min_cost = min(new_costs)

        self._acost_pub.publish(Float32(data=new_min_cost))

        self._current_trajectory['free_dir'] = paths_msg.free_dir.data # independent of update or not

        last_cost = self._current_trajectory.get('cost', np.infty)
        if new_min_cost < last_cost: # or last_traj_done 
            rospy.loginfo(f"Updating due to new best cost: {last_cost:.3f} -> {new_min_cost:.3f}")
            best_path_id = np.argmin(new_costs)

            T_optim = paths_msg.paths[best_path_id]
            self._selected_traj_pub.publish(T_optim)

            paths = paths[best_path_id]
            cost_lst = costs_lst[best_path_id]
            # TODO(R3): subsample only when dispatching to the planner, not in the cost state.
            stride = self._local_planner.waypoint_stride
            if stride > 1:
                paths = np.vstack((paths[::stride], paths[-1]))
                cost_lst = np.concatenate((cost_lst[::stride], np.array([cost_lst[-1]])))

            self._current_trajectory['traj']      = paths
            self._current_trajectory['traj_odom'] = transform_to_frame(self.tf_buffer, paths, "odom", self.robot_frame)
            self._current_trajectory['cost']      = new_min_cost
            self._current_trajectory['frozen_costs'] = cost_lst
            self._update_signal = True



    def update_current_cost(self):
        # This func is to update the cost of the currently executed trajectory, freezing past waypoint costs
        costmap = self._last_cmap
        if self._current_trajectory.get('traj_odom') is not None:
            path = self._current_trajectory['traj_odom'][self._current_wpt:, :] # only the remaining path
            path = path[np.newaxis] #path.reshape(1, *path.shape)
            goal = self._global_goal

            # print("previo  ", costmap.shape, path.shape, goal.shape)
            cost, cost_lst, (pixel_points, out) = self._calculate_path_cost_full(costmap, path, goal, source_frame='odom')
            # print("after", cost, cost_lst.shape, pixel_points.shape, out.shape)
            # # print("updated cost:  ", cost, ' last cost ', self._current_trajectory['cost'], pixel_points.shape)

            # self._update_traj_cost(self._current_trajectory['frozen_costs'],
            #                        cost_lst.squeeze(), # (1, N_wpts) -> (N_wpts,)
            #                        cost[0], # only one value 
            #                        out.squeeze()) # (1, x) -> (x,) or it may have size 0

            img = cv2.cvtColor(costmap * 100, cv2.COLOR_GRAY2RGB)
            annotation = annotate_costmap(img, pixel_points[:, self._current_wpt:, :])
            
            annot_msg, annot_cmp_msg = cv2_to_imgmsg(annotation, compressed=True)
            self._pathon_costmap_pub.publish(annot_msg)
            self._pathon_costmapcmp_pub.publish(annot_cmp_msg)


    def _update_traj_cost(self, frozen_costs, new_wpt_costs, total_cost, outmask):
        print("frozen cost: ", frozen_costs, " new costs: ", new_wpt_costs)

        if outmask.size > 0: # if there are out of bounds waypoints, keep their previous costs
           new_wpt_costs[outmask] = frozen_costs[self._current_wpt:][outmask]

        frozen_cost = sum(frozen_costs[:self._current_wpt])

        self._current_trajectory['frozen_costs'][self._current_wpt:] = new_wpt_costs # keep prev costs and update the following
        self._current_trajectory['cost'] = frozen_cost + total_cost # save current trajectory cost
        
        self._rcost_pub.publish(Float32(data=frozen_cost + total_cost))


    def _update_traj_cost_old(self, x, y, costmap, umask, colmask, outmask=[]):
        # update only the current trajectory, ideally using the above function?
        prev_costs_lst = self._current_trajectory.get('frozen_costs')
        goal_dst = self._current_trajectory.get('goal_dst')

        if (prev_costs_lst and goal_dst) is not None:
            cost, cost_lst = self._calculate_path_cost(x, y, costmap, umask, colmask, goal_dst, 
                                                       start=self._current_wpt,
                                                       outmask=outmask,
                                                       prev_costs=prev_costs_lst) # TODO check exponent start idx on discount

            frozen_cost = sum(prev_costs_lst[:self._current_wpt])

            self._current_trajectory['frozen_costs'][self._current_wpt:] = cost_lst # keep prev costs and update the following
            self._current_trajectory['cost'] = frozen_cost + cost # save current trajectory cost

        else:
            rospy.logerr("No selected trajectory yet")


    def _make_costmap(self, img: np.ndarray):

        inputs = self._clip_processor(text=self._constraints, 
                                      images=[img] * len(self._constraints), 
                                      padding="max_length", 
                                      return_tensors="pt")
        inputs = {k: v.to(torch.device('cuda')) for k, v in inputs.items()}
         
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
        preds = outputs.logits.unsqueeze(1).cpu()
        mask1 = np.argmax(preds[:, 0].cpu().numpy(), axis=0) # take the max likelihood
        mask  = self._costs[mask1]
        mask  = cv2.resize(mask.astype(np.uint8), img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        self._last_cmap = mask

        costmap_viz = humanize_costmap(cv2.resize(mask1.astype(np.uint8), img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR))
        costmap, costmap_cmp = cv2_to_imgmsg(costmap_viz, rgb=True, compressed=True)
        self._costmap_pub.publish(costmap)
        self._costmapcmp_pub.publish(costmap_cmp)


    def _calculate_path_cost_full(self, map, paths, goal, source_frame):
        K, N, _ = paths.shape
        # shape: (K, N_wpts, 3), z is dummy 
        pixel_points = transform_and_project_paths(self.tf_buffer, paths, self._K, source_frame=source_frame)
        pixel_points = np.stack(pixel_points)


        x = pixel_points[..., 0].astype(np.int64)  # (K, N_wpts)
        outmask = np.where((x < 0) | (x > map.shape[1]))[0]
        x = x.clip(0, map.shape[1] - 1)
        y = pixel_points[..., 1].astype(np.int64).clip(0, map.shape[0] - 1)

        semantic_values = map[y, x] # (K, N_wpts)

        # Identify unknown space. All paths here are already collisions free!!!!! (in theory lol)
        occlusions = semantic_values > 1 # > or >= 

        # Compute semantic cost per waypoint vectorized
        gamma_factors = self._gamma ** np.arange(N)  # (N_wpts,)

        # Cost per waypoint: penalize unknown differently
        waypoint_costs = (
            (semantic_values * (~occlusions) + self.C_unk * occlusions) * gamma_factors
        )  # (N_paths, N_wpts)

        # Handle case when semantics are ignored
        if self._ignore_semantics:
            waypoint_costs.fill(0)

        # Sum semantic waypoint costs across paths
        semantic_cost = np.sum(waypoint_costs, axis=1)  # (N_paths,)

        # Vectorized distance
        local_goal = transform_to_frame(self.tf_buffer, 
                                        np.concatenate((goal, np.zeros(1))).reshape(1, -1), 
                                        self.robot_frame, self.global_frame).squeeze()[:2]

        distance_cost = self._get_distance_cost(paths, local_goal) # (N_paths,)
        
        # Total path costs
        path_costs = semantic_cost + distance_cost  # (N_paths,)

        return path_costs, waypoint_costs, (pixel_points, outmask)


    def _get_distance_cost(self, paths, goal):
        dists = get_path_dist_to_goal(paths, goal)
        distance_cost = np.log1p(dists[:, 0]) * self._beta # log will be less than 5 for a distance of 100
        angle_cost    = np.abs(dists[:, 1]) / np.pi * self._beta2 # angle cost is max 1 w/no beta

        return angle_cost + distance_cost
    

    def reset_trajectories(self):
        self._current_trajectory = {}
        self._current_wpt = 0
        self.start = True


    def execute_trajectory(self, path: np.ndarray) -> None:
        """ Process the received trajectory and send waypoints to the local planner. """
        rospy.loginfo("Processing new trajectory...")        
        # OJO ACA, EL GOAL DEBE ESTAR EN EL FRAME DE ODOMETRIA PQ EL LOCAL PLANNER FUNCIONA CON ODOMETRIA
        T_optim_global_frame = transform_to_frame(self.tf_buffer, path, target_frame=self.global_frame, source_frame=self.robot_frame)[:, :2]  # Kx2
        T_optim_odom_frame   = transform_to_frame(self.tf_buffer, path, target_frame=self.global_frame, source_frame=self.robot_frame)[:, :2]  # Kx2

        # T_optim_global_frame = np.vstack((T_optim_global_frame[::3], T_optim_global_frame[-1]))

        for m in range(self._current_wpt, min(self._max_wpts, len(T_optim_global_frame))):  # Ensure we don't exceed the number of waypoints
            current_waypoint = T_optim_global_frame[m]
            self._current_wpt = m
            print(f"Current waypoint {m}: {len(T_optim_global_frame)}")

            self._local_planner.send_waypoint(T_optim_odom_frame[m], frame_id=self.global_frame)

            start_time = rospy.Time.now()
            waypoint_timeout = 10.0
            while not rospy.is_shutdown():
                elapsed_time = (rospy.Time.now() - start_time).to_sec()

                goal_dst = np.linalg.norm(self._pose_xy - self._global_goal)
                if goal_dst < self._goal_th:
                    self._local_planner.cancel()
                    # self._global_goal = None
                    self._done_signal = True
                    self._done_signal_pub.publish(Empty())
                    rospy.loginfo("Goal reached.")
                    rospy.sleep(1.0)
                    self._alignment_cnt = 0
                    return

                planner_status = self._local_planner.status()
                if (np.linalg.norm(self._pose_xy - current_waypoint) <= 1.0
                        or planner_status == PlannerStatus.SUCCEEDED):
                    rospy.loginfo(f"Waypoint {m} achieved. Moving to next waypoint.")
                    break  # Move to next waypoint

                if planner_status == PlannerStatus.FAILED:
                    rospy.logwarn(f"Local planner failed on waypoint {m}; skipping it.")
                    break

                if elapsed_time > waypoint_timeout:
                    rospy.logwarn(f"Skipping waypoint {m} due to timeout.")
                    self._local_planner.cancel()
                    break

                if self._update_signal:
                    rospy.logwarn("New trajectory received, stopping current execution.")
                    self._local_planner.cancel()
                    self._update_signal = False
                    return  # Stop processing and wait for the new trajectory
                
                if self._alignment_cnt >= 3 and np.linalg.norm(self._current_vel) < 0.25: 
                    self._local_planner.cancel()
                    self._recovery_signal = True
                    # self._done_signal = True
                    self.reset_trajectories()
                    self._alignment_cnt = 0
                    return
                
                if self._joy_override:
                    self._local_planner.cancel()
                    return

                rospy.sleep(0.1)

        self.reset_trajectories()

        rospy.loginfo("Trajectory execution complete.")


    def recovery(self):
        # 1. stop robot
        stop_robot(self._cmd_vel_pub)

        if self._current_free_dir[0] == -1: # all free space, turn in the direction of the goal
            robot_dir = np.sign(self._local_goal[1])
        else:
            pts = np.array(self._current_free_dir) - np.pi / 2  # convert to [-pi/2, pi/2] range
            closest_to_goal = min(pts, key=lambda p: abs(self._local_goal[1] - p))
            robot_dir = np.sign(closest_to_goal)

        self.reset_trajectories()
        self._done_signal = False

        # 2. obtain the rotation direction and speed
        kp = 0.4
        vel = robot_dir * kp

        while self._current_trajectory.get('traj') is None:
            rotate_in_place(self._cmd_vel_pub, vel)

        self.reset_trajectories()
        self._recovery_signal = False
        self._alignment_cnt   = 0


    def run(self):
        if not self._joy_override:
            if self._current_trajectory.get('traj') is not None and not self._done_signal:
                self.execute_trajectory(self._current_trajectory['traj'])

            elif self._alignment_cnt >= 3 and np.linalg.norm(self._current_vel) < 0.25 and not self._done_signal:
                self.recovery()
        else:
            print('Esperanding goal papito')
            rospy.sleep(0.2)



# Main function
if __name__ == "__main__":
    rospy.init_node("selector_node")

    projector = TrajectorySelector()
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        projector.run()
        rate.sleep()
