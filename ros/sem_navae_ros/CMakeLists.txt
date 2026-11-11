cmake_minimum_required(VERSION 3.0.2)
project(sem_navae_ros)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  cv_bridge
  image_transport
  grid_map_core
  grid_map_ros
  grid_map_cv
  grid_map_msgs
  tf2_ros
  tf2_geometry_msgs
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Python package (src/sem_navae_ros: local planner plugins)
catkin_python_setup()

add_message_files(FILES PathArray.msg)
generate_messages(DEPENDENCIES std_msgs nav_msgs sensor_msgs geometry_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs nav_msgs sensor_msgs geometry_msgs
)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Elevation collision filter (Eq. 6)
add_executable(grid_map_collisions src/elevation_collision_filter.cpp)
add_dependencies(grid_map_collisions ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(grid_map_collisions ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS grid_map_collisions
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
catkin_install_python(PROGRAMS
  nodes/generator_node.py
  nodes/selector_node.py
  nodes/visualizer_node.py
  nodes/tf_to_pose_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/selection_replay.test)
endif()
