cmake_minimum_required(VERSION 3.0.2)
project(sem_navae_ros)


## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Debug)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  Eigen3
  cv_bridge
  grid_map_ros
  grid_map_msgs
  grid_map_cv
  grid_map_core
  image_transport
)


################################################
## Declare ROS messages, services and actions ##
################################################


add_message_files(
  FILES
  PathArray.msg
  # SelectedPath.msg
  # ImageArray.msg
  # Debug.msg
)

# add_service_files(
#   FILES
#   Annotate.srv
#   GetCollision.srv
# )

generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
)


###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mtg_ros
  #  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime nav_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(grid_map_collisions src/elevation_collision_filter.cpp)
target_link_libraries(grid_map_collisions ${catkin_LIBRARIES} Eigen3::Eigen ${OpenCV_LIBRARIES})

