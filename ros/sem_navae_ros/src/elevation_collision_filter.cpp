#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <grid_map_msgs/GridMap.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Matrix3x3.h>
// #include <std_msgs/Empty.h>
#include <std_msgs/Int8.h>
#include <std_msgs/UInt8MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sem_navae_ros/PathArray.h>
#include <nav_msgs/Path.h>
// #include <unordered_set>
// #include <Eigen/Dense>

#include <iostream>
#include <algorithm> 

struct Trajectory {
  std::vector<float> frozen_costs;  // Cost values for the trajectory
  float goal_dst;                   // Distance to the goal
  float cost;                       // Total cost of the trajectory
  nav_msgs::Path trajectory;
};


class GridMapSubmapNode {
public:
  GridMapSubmapNode()
    : nh_(),
      tfListener_(tfBuffer_),
      it_(nh_)
  {
    gridMapSub_  = nh_.subscribe("/elevation_mapping/elevation_map", 1, &GridMapSubmapNode::gridMapCallback, this);
    pathSub_     = nh_.subscribe("/sem_navae/trajectories", 1, &GridMapSubmapNode::pathCallback, this); // unchecked for collisions
    pathPub_     = nh_.advertise<sem_navae_ros::PathArray>("/sem_navae/trajectories_filt", 1);
    imagePub_    = it_.advertise("/elevation_image", 1);
    imgPathsPub_ = it_.advertise("/elevation_image_paths", 1);

    nh_.getParam("elevation_filter/robot_footprint_x", robot_width_);
    nh_.getParam("elevation_filter/collision_elevation_th", elevation_th_);
    nh_.getParam("elevation_filter/collision_angle_th", angle_th_);
    nh_.getParam("elevation_filter/use_odom_frame", use_odom_frame_);
    nh_.getParam("model/max_wpts", max_wpts_);
    nh_.getParam("model/n_paths",  n_paths_);

    angle_th_ = angle_th_ * M_PI / 180;

    ROS_INFO("USING 'robot_footprint': %f", robot_width_);
    ROS_INFO("USING 'elevation_th': %f", elevation_th_);
    ROS_INFO("USING 'angle_th': %f", angle_th_);
    ROS_INFO("USING 'max_wpts': %i", max_wpts_);
    ROS_INFO("USING 'n_paths': %i", n_paths_);

  }

  void gridMapCallback(const grid_map_msgs::GridMap& msg) {
    // grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(msg, map_);

    if (!map_.exists(layer_)) {
        ROS_WARN("Layer '%s' not found in the map.", layer_.c_str());
        return;
    }

    if (use_odom_frame_)
    {
      geometry_msgs::TransformStamped tf_robot;
      try {
        tf_robot = tfBuffer_.lookupTransform("odom", "base_link", ros::Time(0));
      } catch (tf2::TransformException& ex) {
        ROS_WARN("TF lookup failed: %s", ex.what());
        return;
      }

      double robot_x = tf_robot.transform.translation.x;
      double robot_y = tf_robot.transform.translation.y;
      const grid_map::Position robot_position(robot_x, robot_y);

      grid_map::Length roi_size(sub_map_size_, sub_map_size_);
      bool ss;

      if (!map_.isInside(robot_position)) {
        ROS_WARN("Robot is outside of the grid map.");
        return;
      }

      // grid_map::Index topLeftIndex;
      grid_map::GridMap map_sub;
      try {
        map_sub = map_.getSubmap(robot_position, roi_size, ss);
      } catch (...) {
        ROS_WARN("Failed to extract submap.");
        return;
      }

      const auto& data_size = map_sub.getSize();
      cv::Mat img(data_size(1), data_size(0), CV_32FC1, cv::Scalar(NAN));

      for (grid_map::GridMapIterator it(map_sub); !it.isPastEnd(); ++it) {
        const grid_map::Index idx(*it);
        float value = map_sub.at(layer_, idx);
        if (std::isfinite(value))
          img.at<float>(idx(1), idx(0)) = value; // CV row=y, col=x
      }

      // Normalize to 0–255 and convert to 8-bit image
      double minVal, maxVal;
      cv::minMaxLoc(img, &minVal, &maxVal);
      // cv::Mat img_8u;
      cv::normalize(img, img_8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);

      tf2::Quaternion q(
        tf_robot.transform.rotation.x,
        tf_robot.transform.rotation.y,
        tf_robot.transform.rotation.z,
        tf_robot.transform.rotation.w
      );
      double r, p, yaw;
      tf2::Matrix3x3(q).getRPY(r, p , yaw);

      // Center of the image
      cv::Point2f center(img_8u.cols / 2.0f, img_8u.rows / 2.0f);

      // Compute rotation matrix (rotate by -yaw to make robot point up)
      cv::Mat rot_mat = cv::getRotationMatrix2D(center, -yaw * 180.0 / M_PI, 1.0);

      // Apply the affine transformation
      // cv::Mat img_rotated;
      cv::warpAffine(img_8u, img_8u, rot_mat, img_8u.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

      sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img_8u).toImageMsg();
      out_msg->header.stamp = ros::Time::now();
      out_msg->header.frame_id = "base_link"; // Robot-centric view
      imagePub_.publish(out_msg);
    }

    else{ // assume base_link as the map frame.
      // Get size and resolution
      const auto size = map_.getSize();
      int rows = size(0);
      int cols = size(1);
      ROS_INFO_ONCE("Map size: %d x %d", rows, cols);

      // Create OpenCV image (float32 for now)
      cv::Mat img(rows, cols, CV_32FC1, cv::Scalar(NAN));

      // Fill image with elevation data
      for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
        const grid_map::Index idx(*it);
        float value = map_.at(layer_, idx);
        if (std::isfinite(value))
          img.at<float>(idx(0), idx(1)) = value;
      }
      _elv_map_img = img.clone();
      
      cv::Rect roi(img.rows / 2 - 7, img.cols - 15, 14, 15); // where tf dis came from??
      img(roi) = 0;

      cv::Mat img_inpainted = img.clone(); //fillMissingElevation(img);
      
      // Normalize to 0–255 and convert to 8-bit image
      double minVal, maxVal;
      cv::minMaxLoc(img_inpainted, &minVal, &maxVal);
      // cv::Mat img_8u;
      cv::normalize(img_inpainted, img_8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);


      sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img_8u).toImageMsg();
      out_msg->header.stamp = ros::Time::now();
      out_msg->header.frame_id = "base_link";

      imagePub_.publish(out_msg);
    }
  }


  void pathCallback(const sem_navae_ros::PathArray& msg)
  {
    cv::Mat rgb_img;
    if (img_8u.empty())
      return;
    
    cv::cvtColor(img_8u, rgb_img, cv::COLOR_GRAY2RGB);
    std::vector<std::vector<int>> collisions;
    collisions.resize(n_paths_, std::vector<int>(max_wpts_, 0));

    int p = 0;
    for (auto& path : msg.paths)
    {
      bool collision_found = false;
      size_t collision_index = path.poses.size();  // default: no collision

      // 1. Check for collisions between waypoints
      float prev_z = 0;
      for (size_t i = 2; i < path.poses.size(); ++i) 
      {
        if (collision_found) break;
        
        const auto& pose1 = path.poses[i - 1].pose;
        const auto& pose2 = path.poses[i].pose;
        
        double x1 = pose1.position.x;
        double y1 = pose1.position.y;
        double x2 = pose2.position.x;
        double y2 = pose2.position.y;

        int num_samples = 3; // waypoint_dst / robot_lenght ~ 2 or 3

        for (int j = 0; j <= num_samples; ++j) 
        {
          double t = static_cast<double>(j) / num_samples;
          double x = x1 + t * (x2 - x1);
          double y = y1 + t * (y2 - y1);

          grid_map::Position pos(x, y);
          if (!map_.isInside(pos)) continue;
          grid_map::Length roi_size(robot_length_, robot_width_);
          grid_map::GridMap map_sub;
          bool ss;

          try {
              map_sub = map_.getSubmap(pos, roi_size, ss);
            } catch (...) {
              ROS_WARN("Failed to extract submap.");
              continue;  // submap out of bounds or failed
          }

          try
          {
            /* code */
            Eigen::MatrixXf layer_data = map_sub[layer_];
            layer_data = layer_data.unaryExpr([](float v) { return std::isfinite(v) ? v : 0.0f; });
            
            if (layer_data.size() == 0 || layer_data.rows() == 0 || layer_data.cols() == 0) {
              ROS_DEBUG("layer_data is empty, skipping waypoint");
              continue;  // o return
            }

            float max_value = layer_data.maxCoeff();
            
            // ROS_INFO("Max value in submap: %f", max_value);
            // float z_world = map_.at(layer_, idx);
            
            float angle = std::atan2(max_value - prev_z, std::sqrt(std::pow(x1-x2, 2) + std::pow(y1-y2, 2)));
            prev_z = max_value;
            
            // ROS_WARN("ANGLE OF WPT [%i][%i]: %f\nHEIGHTOF WPT: %f", p, i, angle, max_value);
            
            if (std::isfinite(max_value) && max_value >= elevation_th_ && angle > angle_th_) {
              collision_found = true;
              collision_index = i;
              break;
            }
          }
          catch(...)
          {
            std::cerr << "qe" << '\n';
          }
        }
      }
        
        // 2. Plot only the waypoints
      for (size_t i = 0; i < path.poses.size(); ++i) {
        const auto& pose = path.poses[i].pose;
        double wx = pose.position.x;
        double wy = pose.position.y;

        grid_map::Position pos(wx, wy);
        if (!map_.isInside(pos)) continue;

        grid_map::Index idx;
        map_.getIndex(pos, idx);
        
        int row = idx(0);
        int col = idx(1);

        if (row >= 0 && row < rgb_img.rows && col >= 0 && col < rgb_img.cols) {
          cv::Scalar color = (i < collision_index) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);  // green or red
          cv::circle(rgb_img, cv::Point(col, row), 1, color, -1);
        }

        collisions[p][i] = (i < collision_index) ? 0 : 1;
      }
      p++;
    } //endfor patharray

    
    // get the free dirs
    std::vector<float> free_dir = get_free_dirs(_elv_map_img);

    for (const auto& dir : free_dir) 
    {
      int px = static_cast<int>(80 * std::cos(dir) + rgb_img.cols / 2); // sum the map origin offset
      int py = static_cast<int>(-80 * std::sin(dir) + rgb_img.rows); // sum the map origin offset
      if (px >= 0 && px < rgb_img.cols && py >= 0 && py < rgb_img.rows) 
        cv::circle(rgb_img, cv::Point(px, py), 2, cv::Scalar(255, 0, 0), -1); // blue circle for free direction
    }

    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_img).toImageMsg();
    out_msg->header.stamp = ros::Time::now();
    out_msg->header.frame_id = "base_link";
    auto trajs_msg = _build_path_message(msg, collisions, n_paths_, max_wpts_, free_dir);

    pathPub_.publish(trajs_msg);
    imgPathsPub_.publish(out_msg);
  }

  
  cv::Mat fillMissingElevation(const cv::Mat& elevation_map) {
    cv::Mat mask, elevation_filled;

    // Create mask where elevation == 0 (assume 0 are missing values)
    mask = (elevation_map == 0);

    // Convert to 8-bit single channel for inpainting mask
    mask.convertTo(mask, CV_8U, 255);

    // Normalize input to 0-255 for inpainting algorithm
    double minVal, maxVal;
    cv::minMaxLoc(elevation_map, &minVal, &maxVal, nullptr, nullptr, ~mask);
    
    // Avoid division by zero
    if (maxVal - minVal < 1e-3) maxVal = minVal + 1.0;

    cv::Mat elevation_norm;
    elevation_map.convertTo(elevation_norm, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Inpainting (you can choose INPAINT_NS or INPAINT_TELEA)
    cv::Mat inpainted_norm;
    cv::inpaint(elevation_norm, mask, inpainted_norm, 5, cv::INPAINT_TELEA);

    // Convert back to original scale
    inpainted_norm.convertTo(elevation_filled, CV_32F, (maxVal - minVal) / 255.0, minVal);

    // Optional smoothing to reduce artifacts
    // cv::GaussianBlur(elevation_filled, elevation_filled, cv::Size(3, 3), 0);

    return elevation_filled;
  }


  sem_navae_ros::PathArray _build_path_message(const sem_navae_ros::PathArray& trajectories_raw,
                    const std::vector<std::vector<int>>& collisions, int K, int N, const std::vector<float>& free_dir) 
  {
    sem_navae_ros::PathArray trajectories;
    trajectories.paths = trajectories_raw.paths;
    trajectories.goal  = trajectories_raw.goal;

    trajectories.free_dir.data = free_dir;
    trajectories.goal_relative = trajectories_raw.goal_relative;
    
    // Handle collisions data
    if (collisions.size() > 0) {
        std_msgs::UInt8MultiArray collisions_msg;
        
        // Flatten and populate the collision data
        int K = collisions.size();
        int N = collisions[0].size();
        collisions_msg.data.resize(K * N);
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                collisions_msg.data[i * N + j] = collisions[i][j];
            }
        }

        // Set up the layout for the MultiArray
        collisions_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        collisions_msg.layout.dim[0].label = "height";
        collisions_msg.layout.dim[0].size = K;
        collisions_msg.layout.dim[0].stride = N;

        collisions_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        collisions_msg.layout.dim[1].label = "width";
        collisions_msg.layout.dim[1].size = N;
        collisions_msg.layout.dim[1].stride = 1;

        trajectories.collisions = collisions_msg;
    } else {
        // No collision data, just pass an empty array
        std_msgs::UInt8MultiArray empty_collisions;
        trajectories.collisions = empty_collisions;
    }

    return trajectories;
  }


  std::vector<std::vector<int>> _computeCollisions(const sem_navae_ros::PathArray& msg)
  {
    std::vector<std::vector<int>> collisions(n_paths_, std::vector<int>(max_wpts_, 0));

    int p = 0;
    for (const auto& path : msg.paths)
    {
      bool collision_found = false;
      size_t collision_index = path.poses.size();  // default: no collision

      float prev_z = 0.0f;
      for (size_t i = 3; i < path.poses.size(); ++i)
      {
        if (collision_found) break;

        const auto& pose1 = path.poses[i - 1].pose;
        const auto& pose2 = path.poses[i].pose;

        double x1 = pose1.position.x;
        double y1 = pose1.position.y;
        double x2 = pose2.position.x;
        double y2 = pose2.position.y;

        int num_samples = 3;
        for (int j = 0; j <= num_samples; ++j)
        {
          double t = static_cast<double>(j) / num_samples;
          double x = x1 + t * (x2 - x1);
          double y = y1 + t * (y2 - y1);

          grid_map::Position pos(x, y);
          if (!map_.isInside(pos)) continue;

          grid_map::Length roi_size(robot_length_, robot_width_);
          grid_map::GridMap map_sub;
          bool ss;

          try {
            map_sub = map_.getSubmap(pos, roi_size, ss);
          } catch (...) {
            ROS_WARN("Failed to extract submap.");
            continue;
          }

          try {
            Eigen::MatrixXf layer_data = map_sub[layer_];
            layer_data = layer_data.unaryExpr([](float v) { return std::isfinite(v) ? v : 0.0f; });

            float max_value = layer_data.maxCoeff();
            float angle = std::atan2(max_value - prev_z, std::hypot(x2 - x1, y2 - y1));
            prev_z = max_value;

            if (std::isfinite(max_value) && max_value >= elevation_th_ && angle > angle_th_) {
              collision_found = true;
              collision_index = i;
              break;
            }
          } catch (...) {
            ROS_ERROR("Error computing collision");
          }
        }
      }

      for (size_t i = 0; i < path.poses.size(); ++i) {
        collisions[p][i] = (i < collision_index) ? 0 : 1;
      }
      p++;
    }

    return collisions;
  }


  std::vector<float> get_free_dirs(cv::Mat elevation_map)
  {
    // robot center px, py = W//2, H
    // regions to visit from -pi/2 to pi/2 with step size pi / 10
    float R_max = 8.0; // max distance to check for collisions
    float theta = 0;
    float increment = M_PI / 10;
    std::vector<float> free_dirs;
    std::vector<int> info;

    for (int i=0; i<11; i++)
    {
      theta = theta + increment;
      info = obtain_collision_distance(elevation_map, R_max, theta);
      if (!info[0]) // detected not collision
      {
        free_dirs.push_back(theta);
      }
    }
    if (free_dirs.empty())
    free_dirs.push_back(-1); // if no free dirs, return -1

    return free_dirs;
  }

  std::vector<int> obtain_collision_distance(cv::Mat map, float R_max, float theta)
  {
    float map_resolution = 0.1;
    int collision = false;
    int r;
    for (r=5; r < static_cast<int>(R_max / map_resolution); r++)
    {
      int px = r * std::cos(theta) + map.cols / 2; // sum the map origin offset
      int py = -r * std::sin(theta) + map.rows; 
      px = std::clamp(px, 0, map.cols-1);
      py = std::clamp(py, 0, map.rows-1);
      
      double minVal, maxVal;
      float value;
      try
        {value = map.at<float>(py, px);}
      catch(...)
        {ROS_WARN("Failed to extract map pixel values");}

      if (std::isnan(value))
      {
        // ROS_INFO("NaN value at px: %i, py: %i", px, py);
        continue; // skip NaN values
      }
      if (value > 0.4)
      {
        collision = true;
        break;
      }
    }

    std::vector<int> info = {collision, r};
    return info;
  }


private:
  ros::NodeHandle nh_;
  ros::Subscriber gridMapSub_;
  ros::Subscriber pathSub_;
  ros::Publisher pathPub_;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;
  image_transport::ImageTransport it_;
  image_transport::Publisher imagePub_;
  image_transport::Publisher imgPathsPub_;
  
  grid_map::GridMap map_; // the current elevation map
  cv::Mat img_8u;
  cv::Mat _elv_map_img;
  // cv::Mat img_elvn;
  std::string layer_ = "elevation";
  
  float robot_width_ = 0.65; // robot width in meters
  float robot_length_ = 0.65; // robot length in meters
  float elevation_th_ = 0.22;
  float angle_th_ = 10.0;
  int max_wpts_ = 12;
  int n_paths_ = 3;
  
  double sub_map_size_ = 12.0; 

  bool use_odom_frame_ = false;

};


int main(int argc, char** argv) {
  ros::init(argc, argv, "grid_map_submap_node");
  GridMapSubmapNode node;
  ros::spin();
  return 0;
}