cmake_minimum_required(VERSION 3.5)
project(tare_planner)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclpy REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(backward_ros REQUIRED)
find_package(nlohmann_json REQUIRED)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
  or-tools/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  /opt/ros/humble/include
  # /usr/local/include
  # /usr/include
)

# message(WARNING "CMAKE_CURRENT_SOURCE_DIR is '${CMAKE_CURRENT_SOURCE_DIR}'")
message(STATUS "PCL_INCLUDE_DIRS = ${PCL_INCLUDE_DIRS}")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectNode.msg"
  "msg/ObjectNodeList.msg"
  "msg/RoomNode.msg"
  "msg/RoomNodeList.msg"
  "msg/RoomType.msg"
  "msg/ViewpointRep.msg"
  "msg/DetectionResult.msg"
  "msg/RoomEarlyStop1.msg"
  "msg/VlmAnswer.msg"
  "msg/ObjectType.msg"
  "msg/NavigationQuery.msg"
  "msg/TargetObjectInstruction.msg"
  "msg/TargetObject.msg"
  "msg/TargetObjectWithSpatial.msg"
 DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# ## Specify additional locations for library files
# link_directories(
#   /usr/local/lib # Location when using 'make system_install'
#   /usr/lib       # More usual location (e.g. when installing using a package)
#   or-tools/lib
# )

add_library(lidar_model src/lidar_model/lidar_model.cpp)
ament_target_dependencies(lidar_model rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 visualization_msgs pcl_ros pcl_conversions)
# target_link_libraries(lidar_model ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_library(graph src/graph/graph.cpp)
ament_target_dependencies(graph rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
# target_link_libraries(graph ${catkin_LIBRARIES})

add_library(misc_utils src/utils/misc_utils.cpp)
ament_target_dependencies(misc_utils rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs OpenCV)
target_link_libraries(misc_utils ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_library(pointcloud_utils src/utils/pointcloud_utils.cpp)
ament_target_dependencies(pointcloud_utils rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(pointcloud_utils ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_library(tsp_solver src/tsp_solver/tsp_solver.cpp)
ament_target_dependencies(tsp_solver rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(tsp_solver ${catkin_LIBRARIES} ${CMAKE_CURRENT_SOURCE_DIR}/or-tools/lib/libortools.so)

add_library(viewpoint src/viewpoint/viewpoint.cpp)
ament_target_dependencies(viewpoint rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(viewpoint ${catkin_LIBRARIES} lidar_model)

add_library(rolling_grid src/rolling_grid/rolling_grid.cpp)
ament_target_dependencies(rolling_grid rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(rolling_grid ${catkin_LIBRARIES} misc_utils pointcloud_utils)

add_library(viewpoint_manager STATIC src/viewpoint_manager/viewpoint_manager.cpp)
ament_target_dependencies(viewpoint_manager rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 OpenCV)
target_link_libraries(viewpoint_manager ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} rolling_grid viewpoint grid_world)

add_library(local_coverage_planner src/local_coverage_planner/local_coverage_planner.cpp)
ament_target_dependencies(local_coverage_planner rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(local_coverage_planner ${catkin_LIBRARIES} viewpoint_manager planning_env)

add_library(grid_world STATIC src/grid_world/grid_world.cpp)
ament_target_dependencies(grid_world rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 OpenCV)
target_link_libraries(grid_world ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} viewpoint_manager tsp_solver keypose_graph exploration_path misc_utils)

add_library(pointcloud_manager src/pointcloud_manager/pointcloud_manager.cpp)
ament_target_dependencies(pointcloud_manager rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(pointcloud_manager ${catkin_LIBRARIES} misc_utils)

add_library(keypose_graph STATIC src/keypose_graph/keypose_graph.cpp)
ament_target_dependencies(keypose_graph rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(keypose_graph ${catkin_LIBRARIES} ${PCL_LIBRARIES} viewpoint_manager)

add_library(rolling_occupancy_grid src/rolling_occupancy_grid/rolling_occupancy_grid.cpp)
ament_target_dependencies(rolling_occupancy_grid rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(rolling_occupancy_grid ${catkin_LIBRARIES} misc_utils rolling_grid)

add_library(planning_env src/planning_env/planning_env.cpp)
ament_target_dependencies(planning_env rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(planning_env ${catkin_LIBRARIES} ${PCL_LIBRARIES} rolling_occupancy_grid pointcloud_manager lidar_model viewpoint_manager grid_world)

add_library(exploration_path src/exploration_path/exploration_path.cpp)
ament_target_dependencies(exploration_path rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(exploration_path ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_library(tare_visualizer src/tare_visualizer/tare_visualizer.cpp)
ament_target_dependencies(tare_visualizer rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions visualization_msgs)
target_link_libraries(tare_visualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} grid_world misc_utils planning_env)

add_library(representation src/representation/representation.cpp)
ament_target_dependencies(representation rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_conversions visualization_msgs OpenCV cv_bridge)
target_link_libraries(representation ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} misc_utils pointcloud_utils)

rosidl_target_interfaces(representation
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp")

add_library(sensor_coverage_planner_ground src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp)
ament_target_dependencies(sensor_coverage_planner_ground rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 OpenCV visualization_msgs cv_bridge)
target_link_libraries(sensor_coverage_planner_ground ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} graph planning_env keypose_graph viewpoint_manager pointcloud_manager grid_world local_coverage_planner tare_visualizer misc_utils representation misc_utils nlohmann_json::nlohmann_json)

rosidl_target_interfaces(sensor_coverage_planner_ground
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp")

add_executable(navigationBoundary src/navigation_boundary_publisher/navigationBoundary.cpp)
ament_target_dependencies(navigationBoundary rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2 pcl_ros pcl_conversions)
target_link_libraries(navigationBoundary ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(tare_planner_node src/tare_planner_node/tare_planner_node.cpp)
ament_target_dependencies(tare_planner_node rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs tf2_ros tf2_geometry_msgs tf2)
target_link_libraries(tare_planner_node ${catkin_LIBRARIES} sensor_coverage_planner_ground)


add_executable(room_segmentation src/room_segmentation/room_segmentation.cpp)

ament_target_dependencies(room_segmentation
 rclcpp
 std_msgs
 sensor_msgs
 nav_msgs
 geometry_msgs
 tf2
 tf2_ros
 tf2_geometry_msgs
 message_filters
 pcl_ros
 pcl_conversions
 OpenCV
 cv_bridge
)
target_link_libraries(room_segmentation ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} pointcloud_utils representation misc_utils)

rosidl_target_interfaces(room_segmentation
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp")

#############
## Install ##
#############

install(TARGETS
  room_segmentation
  navigationBoundary
  tare_planner_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
        DESTINATION share/${PROJECT_NAME}/
        )
install(DIRECTORY rviz/
        DESTINATION share/${PROJECT_NAME}/
        )
install(DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}
      )
install(DIRECTORY data/
        DESTINATION share/${PROJECT_NAME}
      )

ament_package()

