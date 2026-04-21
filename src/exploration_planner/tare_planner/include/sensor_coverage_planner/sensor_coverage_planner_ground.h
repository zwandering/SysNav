/**
 * @file sensor_coverage_planner_ground.h
 * @author Chao Cao (ccao1@andrew.cmu.edu)
 * @brief Class that does the job of exploration
 * @version 0.1
 * @date 2020-06-03
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <cmath>
#include <vector>
#include <unordered_set>

#include <Eigen/Core>
// ROS
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <tf2/transform_datatypes.h>
// PCL
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/pca.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
// Third parties
#include <utils/misc_utils.h>
#include <utils/pointcloud_utils.h>
// Components
#include "exploration_path/exploration_path.h"
#include "grid_world/grid_world.h"
#include "keypose_graph/keypose_graph.h"
#include "local_coverage_planner/local_coverage_planner.h"
#include "planning_env/planning_env.h"
#include "rolling_occupancy_grid/rolling_occupancy_grid.h"
#include "tare_visualizer/tare_visualizer.h"
#include "viewpoint_manager/viewpoint_manager.h"

#include "representation/representation.h"
#include "grid/grid.h"
#include "tare_planner/msg/object_node.hpp"
#include "tare_planner/msg/object_node_list.hpp"
#include "tare_planner/msg/viewpoint_rep.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "tare_planner/msg/room_node.hpp"
#include "tare_planner/msg/room_node_list.hpp"
#include "tare_planner/msg/room_type.hpp"
#include "tare_planner/msg/room_early_stop1.hpp"
#include "tare_planner/msg/vlm_answer.hpp"
#include "tare_planner/msg/navigation_query.hpp"
#include "tare_planner/msg/target_object_instruction.hpp"
#include "tare_planner/msg/target_object.hpp"
#include "tare_planner/msg/target_object_with_spatial.hpp"
#include <filesystem>
#include <unordered_map>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define cursup "\033[A"
#define cursclean "\033[2K"
#define curshome "\033[0;0H"

namespace sensor_coverage_planner_3d_ns {
const std::string kWorldFrameID = "map";
typedef pcl::PointXYZRGBNormal PlannerCloudPointType;
typedef pcl::PointCloud<PlannerCloudPointType> PlannerCloudType;
typedef misc_utils_ns::Timer Timer;

class SensorCoveragePlanner3D : public rclcpp::Node {
public:
  explicit SensorCoveragePlanner3D();
  bool initialize();
  void execute();
  ~SensorCoveragePlanner3D() = default;

private:
  // Parameters
  // String
  std::string sub_start_exploration_topic_;
  std::string sub_keypose_topic_;
  std::string sub_state_estimation_topic_;
  std::string sub_registered_scan_topic_;
  std::string sub_terrain_map_topic_;
  std::string sub_terrain_map_ext_topic_;
  std::string sub_coverage_boundary_topic_;
  std::string sub_viewpoint_boundary_topic_;
  std::string sub_viewpoint_room_boundary_topic_;
  std::string sub_nogo_boundary_topic_;
  std::string sub_joystick_topic_;
  std::string sub_reset_waypoint_topic_;

  std::string pub_exploration_finish_topic_;
  std::string pub_runtime_breakdown_topic_;
  std::string pub_runtime_topic_;
  std::string pub_waypoint_topic_;
  std::string pub_momentum_activation_count_topic_;

  // Bool
  bool kAutoStart;
  bool kRushHome;
  bool kUseTerrainHeight;
  bool kCheckTerrainCollision;
  bool kExtendWayPoint;
  bool kUseLineOfSightLookAheadPoint;
  bool kNoExplorationReturnHome;
  bool kUseMomentum;

  // Double
  double kKeyposeCloudDwzFilterLeafSize;
  double kRushHomeDist;
  double kAtHomeDistThreshold;
  double kTerrainCollisionThreshold;
  double kLookAheadDistance;
  double kExtendWayPointDistanceBig;
  double kExtendWayPointDistanceSmall;

  // Int
  int kDirectionChangeCounterThr;
  int kDirectionNoChangeCounterThr;
  int kResetWaypointJoystickAxesID;
  int previous_room_id_;

  std::shared_ptr<pointcloud_utils_ns::PCLCloud<PlannerCloudPointType>>
      keypose_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZ>>
      registered_scan_stack_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      registered_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      large_terrain_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      terrain_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      terrain_ext_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      viewpoint_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      grid_world_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      selected_viewpoint_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      exploring_cell_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      exploration_path_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      lookahead_point_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      keypose_graph_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      viewpoint_in_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      point_cloud_manager_neighbor_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      reordered_global_subspace_cloud_;

  nav_msgs::msg::Odometry keypose_;
  geometry_msgs::msg::Point robot_position_;
  geometry_msgs::msg::Point last_robot_position_;
  lidar_model_ns::LiDARModel robot_viewpoint_;
  exploration_path_ns::ExplorationPath exploration_path_;
  Eigen::Vector3d lookahead_point_;
  Eigen::Vector3d lookahead_point_direction_;
  Eigen::Vector3d moving_direction_;
  double robot_yaw_;
  bool moving_forward_;
  std::vector<Eigen::Vector3d> visited_positions_;
  int cur_keypose_node_ind_;
  Eigen::Vector3d initial_position_;

  std::shared_ptr<keypose_graph_ns::KeyposeGraph> keypose_graph_;
  std::shared_ptr<planning_env_ns::PlanningEnv> planning_env_;
  std::shared_ptr<viewpoint_manager_ns::ViewPointManager> viewpoint_manager_;
  std::shared_ptr<local_coverage_planner_ns::LocalCoveragePlanner>
      local_coverage_planner_;
  std::shared_ptr<grid_world_ns::GridWorld> grid_world_;
  std::shared_ptr<tare_visualizer_ns::TAREVisualizer> visualizer_;

  std::shared_ptr<misc_utils_ns::Marker> keypose_graph_node_marker_;
  std::shared_ptr<misc_utils_ns::Marker> keypose_graph_edge_marker_;
  std::shared_ptr<misc_utils_ns::Marker> nogo_boundary_marker_;
  std::shared_ptr<misc_utils_ns::Marker> grid_world_marker_;

  bool keypose_cloud_update_;
  bool initialized_;
  bool lookahead_point_update_;
  bool relocation_;
  bool start_exploration_;
  bool exploration_finished_;
  bool near_home_;
  bool at_home_;
  bool stopped_;
  bool test_point_update_;
  bool viewpoint_ind_update_;
  bool step_;
  bool use_momentum_;
  bool lookahead_point_in_line_of_sight_;
  bool reset_waypoint_;
  pointcloud_utils_ns::PointCloudDownsizer<pcl::PointXYZ> pointcloud_downsizer_;

  int update_representation_runtime_;
  int local_viewpoint_sampling_runtime_;
  int local_path_finding_runtime_;
  int global_planning_runtime_;
  int trajectory_optimization_runtime_;
  int overall_runtime_;
  int registered_cloud_count_;
  int keypose_count_;
  int direction_change_count_;
  int direction_no_change_count_;
  int momentum_activation_count_;

  double start_time_;
  double global_direction_switch_time_;
  double reset_waypoint_joystick_axis_value_;

  rclcpp::TimerBase::SharedPtr execution_timer_;

  // ROS subscribers
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr exploration_start_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      registered_scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      terrain_map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      terrain_map_ext_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      state_estimation_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      coverage_boundary_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      viewpoint_boundary_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      viewpoint_room_boundary_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      nogo_boundary_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joystick_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_waypoint_sub_;

  // ROS publishers
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_full_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr old_global_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      to_nearest_global_subspace_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_tsp_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr exploration_path_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr waypoint_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr exploration_finish_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr
      runtime_breakdown_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr runtime_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr
      momentum_activation_count_pub_;
  // Debug
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
      pointcloud_manager_neighbor_cells_origin_pub_;

  void ReadParameters();
  void InitializeData();

  // Callback functions
  void
  ExplorationStartCallback(const std_msgs::msg::Bool::ConstSharedPtr start_msg);
  void StateEstimationCallback(
      const nav_msgs::msg::Odometry::ConstSharedPtr state_estimation_msg);
  void RegisteredScanCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr registered_cloud_msg);
  void TerrainMapCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrain_map_msg);
  void TerrainMapExtCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr
                                 terrain_cloud_large_msg);
  void CoverageBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void ViewPointBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void NogoBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void JoystickCallback(const sensor_msgs::msg::Joy::ConstSharedPtr joy_msg);
  void ResetWaypointCallback(const std_msgs::msg::Empty::ConstSharedPtr empty_msg);

  void SendInitialWaypoint();
  void UpdateKeyposeGraph();
  int UpdateViewPoints();
  void UpdateViewPointCoverage();
  void UpdateRobotViewPointCoverage();
  void UpdateCoveredAreas(int &uncovered_point_num,
                          int &uncovered_frontier_point_num);
  void UpdateVisitedPositions();
  void UpdateGlobalRepresentation();
  void GlobalPlanning(std::vector<int> &global_cell_tsp_order,
                      exploration_path_ns::ExplorationPath &global_path);
  void PublishGlobalPlanningVisualization(
      const exploration_path_ns::ExplorationPath &global_path,
      const exploration_path_ns::ExplorationPath &local_path);
  void LocalPlanning(int uncovered_point_num, int uncovered_frontier_point_num,
                     const exploration_path_ns::ExplorationPath &global_path,
                     exploration_path_ns::ExplorationPath &local_path);
  void PublishLocalPlanningVisualization(
      const exploration_path_ns::ExplorationPath &local_path);
  exploration_path_ns::ExplorationPath ConcatenateGlobalLocalPath(
      const exploration_path_ns::ExplorationPath &global_path,
      const exploration_path_ns::ExplorationPath &local_path);

  void PublishRuntime();
  double GetRobotToHomeDistance();
  void PublishExplorationState();
  void PublishWaypoint();
  bool GetLookAheadPoint(const exploration_path_ns::ExplorationPath &local_path,
                         const exploration_path_ns::ExplorationPath &global_path,
                         Eigen::Vector3d &lookahead_point);

  void PrintExplorationStatus(std::string status, bool clear_last_line = true);
  void CountDirectionChange();


  // -------------------------------------------------------------------------------------
  // ========== ROS Subscribers ==========
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr door_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr room_mask_sub_;
  rclcpp::Subscription<tare_planner::msg::RoomNodeList>::SharedPtr room_node_list_sub_;
  rclcpp::Subscription<tare_planner::msg::RoomType>::SharedPtr room_type_sub_;
  rclcpp::Subscription<tare_planner::msg::ObjectNodeList>::SharedPtr object_node_list_sub_;
  rclcpp::Subscription<tare_planner::msg::TargetObject>::SharedPtr anchor_object_sub_;
  rclcpp::Subscription<tare_planner::msg::TargetObject>::SharedPtr target_object_sub_;
  rclcpp::Subscription<tare_planner::msg::TargetObjectInstruction>::SharedPtr target_object_instruction_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_point_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr keyboard_input_sub_;
  rclcpp::Subscription<tare_planner::msg::VlmAnswer>::SharedPtr room_navigation_answer_sub_;

  // ========== ROS Publishers ==========
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr chosen_room_boundary_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr door_normal_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr object_node_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr object_visibility_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr room_type_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viewpoint_room_id_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr viewpoint_visibility_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr room_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr door_position_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr room_anchor_point_pub_;
  rclcpp::Publisher<tare_planner::msg::RoomType>::SharedPtr room_type_pub_;
  rclcpp::Publisher<tare_planner::msg::TargetObject>::SharedPtr anchor_object_pub_;
  rclcpp::Publisher<tare_planner::msg::TargetObject>::SharedPtr target_object_pub_;
  rclcpp::Publisher<tare_planner::msg::TargetObjectWithSpatial>::SharedPtr target_object_spatial_pub_;
  rclcpp::Publisher<tare_planner::msg::NavigationQuery>::SharedPtr room_navigation_query_pub_;
  rclcpp::Publisher<tare_planner::msg::RoomEarlyStop1>::SharedPtr room_early_stop_1_pub_;
  rclcpp::Publisher<tare_planner::msg::ViewpointRep>::SharedPtr viewpoint_rep_pub_;

  // ========== VLM-Related Functions ==========
  // Viewpoint representation
  void UpdateViewpointRep();
  
  // Callback functions
  void ObjectNodeListCallback(const tare_planner::msg::ObjectNodeList::ConstSharedPtr msg);
  void DoorCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr door_cloud_msg);
  void RoomNodeListCallback(const tare_planner::msg::RoomNodeList::ConstSharedPtr room_node_list_msg);
  void GoalPointCallback(const geometry_msgs::msg::PointStamped::ConstSharedPtr goal_point_msg);
  void RoomMaskCallback(const sensor_msgs::msg::Image::ConstSharedPtr room_mask_msg);
  void CameraImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void RoomTypeCallback(const tare_planner::msg::RoomType::ConstSharedPtr msg);
  void RoomNavigationAnswerCallback(const tare_planner::msg::VlmAnswer::ConstSharedPtr msg);
  void KeyboardInputCallback(const std_msgs::msg::String::ConstSharedPtr msg);
  void TargetObjectInstructionCallback(const tare_planner::msg::TargetObjectInstruction::ConstSharedPtr msg);
  void TargetObjectCallback(const tare_planner::msg::TargetObject::ConstSharedPtr msg);
  void AnchorObjectCallback(const tare_planner::msg::TargetObject::ConstSharedPtr msg);
  
  // Utility functions
  bool CheckRayVisibilityInOccupancyGrid(const Eigen::Vector3i& start_pos, const Eigen::Vector3i& end_pos);
  bool InRange(const Eigen::Vector3i& voxel_index) const;
  std::vector<Eigen::Vector3i> Convert2Voxels(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  void GetPoseAtTime(double imageTime, float &lidarX, float &lidarY, float &lidarZ, 
                     float &lidarRoll, float &lidarPitch, float &lidarYaw);
  cv::Mat project_pcl_to_image(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_w,
                                float &lidarX, float &lidarY, float &lidarZ, 
                                float &lidarRoll, float &lidarPitch, float &lidarYaw,
                                cv::Mat &image, pcl::PointXYZI &room_center, int &room_id);
  
  // Visualization functions
  void CreateVisibilityMarkers();
  void PublishViewpointRoomIdMarkers();
  void PublishRoomTypeVisualization();
  void PublishObjectNodeMarkers();
  void PublishFreespaceCloud();
  
  // Room management functions
  void SendInRoomWaypoint();
  void SetCurrentRoomId();
  void SetRoomPosition(const int &start_room_id, const int &end_room_id);
  void SetStartAndEndRoomId();
  void UpdateRoomLabel();
  void ResetRoomInfo();
  void GetToRoomState(bool &at_room, bool &near_room_1, bool &near_room_2);
  void GetDoorNormal(const int &start_room_id, const int &end_room_id, 
                     const Eigen::Vector3d &door_center, Eigen::Vector3d &room_normal);
  void GetDoorCentroid(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr door_cloud_final, 
                       Eigen::Vector3d& door_center);
  void GetDoorCentroid(const int &start_room_id, const int &end_room_id, 
                       Eigen::Vector3d &door_center);
  void CheckDoorCloudInRange();
  double GetRobotToRoomDistance();
  
  // Object detection and tracking functions
  void UpdateObjectVisibility();
  void UpdateViewpointObjectVisibility();
  void CheckObjectFound();
  void ResetFoundObjectInfo();
  void ResetFoundAnchorObjectInfo();
  void SetFoundTargetObject();
  void SetFoundAnchorObject();
  void ProcessObjectNodes();
  void CheckAnchorObjectFound();
  
  // VLM query functions
  void PublishRoomNavigationQuery();
  void ChangeRoomQuery(const int &room_id_1, const int &room_id_2, bool enter_wrong_room = false);
  void GetAnswer();
  
  // JSON serialization functions
  void to_json(json &j, const representation_ns::ObjectNodeRep &obj) const;
  void to_json(json &j, const representation_ns::ViewPointRep &viewpoint) const;
  void to_json(json &j, const representation_ns::RoomNodeRep &room) const;
  void to_json(json &j, const representation_ns::Representation &rep) const;

  // ========== VLM-Related Data Members ==========
  // Representation core
  std::shared_ptr<representation_ns::Representation> representation_;
  
  // Viewpoint representation parameters
  double rep_threshold_;
  int rep_threshold_voxel_num_;
  bool add_viewpoint_rep_;
  int curr_viewpoint_rep_node_ind;
  std::vector<representation_ns::ViewPointRep> viewpoint_reps_;
  std::vector<int> previous_obs_voxel_inds_;
  std::vector<int> current_obs_voxel_inds_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZ>> viewpoint_rep_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>> covered_points_all_;
  tare_planner::msg::ViewpointRep viewpoint_rep_msg_;
  
  // Door and room boundary data
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr door_cloud_;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr door_cloud_final_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZRGBL>> door_cloud_vis_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZLNormal>> door_cloud_in_range_;
  Eigen::Vector3d door_position_;
  Eigen::Vector3d door_normal_;
  
  // Room state flags
  bool ask_vlm_near_room_;
  bool ask_vlm_finish_room_;
  bool ask_vlm_change_room_;
  bool transit_across_room_;
  bool at_room_;
  bool near_room_1_;
  bool near_room_2_;
  bool enter_wrong_room_;
  bool asked_in_advance_;
  bool has_candidate_room_position_;
  
  // Room data structures
  Eigen::MatrixXi adjacency_matrix;
  Eigen::Vector3i room_voxel_dimension_;
  Eigen::Vector3f shift_;
  std::vector<representation_ns::RoomNodeRep> room_nodes_;
  std::vector<representation_ns::RoomNodeRep> room_nodes_tmp;
  cv::Mat room_mask_;
  cv::Mat room_mask_old_;
  
  // Room IDs and positions
  int current_room_id_;
  int target_room_id_;
  int start_room_id_;
  int end_room_id_;
  int prev_room_id_;
  geometry_msgs::msg::Point robot_position_old_;
  geometry_msgs::msg::Point goal_position_;
  geometry_msgs::msg::Point candidate_room_position_;
  
  // Room counters and parameters
  int room_guide_counter_;
  int room_id_change_counter_;
  int room_navigation_query_counter_;
  int stayed_in_room_counter_;
  int room_finished_counter_;
  float room_resolution_;
  float occupancy_grid_resolution_;
  double kRushRoomDist_1;
  double kRushRoomDist_2;
  
  // Target object tracking
  bool found_object_;
  bool ask_found_object_;
  int found_object_id_;
  int found_object_room_id_;
  double found_object_distance_;
  geometry_msgs::msg::Point found_object_position_;
  std::string target_object_;
  
  // Anchor object tracking
  bool found_anchor_object_;
  bool ask_found_anchor_object_;
  int found_anchor_object_id_;
  int found_anchor_object_room_id_;
  double found_anchor_object_distance_;
  geometry_msgs::msg::Point found_anchor_object_position_;
  std::vector<geometry_msgs::msg::Point> found_anchor_object_viewpoint_positions_;
  std::vector<geometry_msgs::msg::Point> found_anchor_object_viewpoint_positions_visited_;
  std::string anchor_object_;
  
  // Object detection parameters
  rclcpp::Time last_object_update_time_;
  double rep_sensor_range;
  std::vector<int> object_ids_to_remove_;
  double obj_score_;
  std::set<int> considered_object_ids_;
  
  // Search and navigation conditions
  std::string room_condition_;
  std::string spatial_condition_;
  std::string attribute_condition_;
  
  // Camera and sensor data
  cv::Mat camera_image_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>> freespace_cloud_;
  
  // Odometry stack for pose interpolation
  static constexpr int kOdomStackSize = 400;
  float lidarXStack[kOdomStackSize];
  float lidarYStack[kOdomStackSize];
  float lidarZStack[kOdomStackSize];
  float lidarRollStack[kOdomStackSize];
  float lidarPitchStack[kOdomStackSize];
  float lidarYawStack[kOdomStackSize];
  double odomTimeStack[kOdomStackSize];
  int odomLastIDPointer;
  int odomFrontIDPointer;
  double odomTime;
  double imageTime;
  float odomX;
  float odomY;
  float odomZ;
  double PI;
  
  // Timing
  rclcpp::Time last_target_object_instruction_time_;
  
  // Miscellaneous flags
  bool dynamic_environment_;
  bool tmp_flag_;
};

} // namespace sensor_coverage_planner_3d_ns
