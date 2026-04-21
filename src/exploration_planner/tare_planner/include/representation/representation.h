/**
 * @file representation.h
 * @author Haokun Zhu (haokunz@andrew.cmu.edu)
 * @brief Class that implements the VLM representation
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

// #include <viewpoint/viewpoint.h>
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
// Third parties
#include <utils/pointcloud_utils.h>
#include <grid/grid.h>

#include "tare_planner/msg/object_node.hpp"
#include "tare_planner/msg/room_node.hpp"

#include <vector>
#include <set>
#include <memory>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

namespace representation_ns
{
const std::string kWorldFrameID = "map";

// Define the ViewPoint node class.
class ViewPointRep
{
public:
  explicit ViewPointRep(int id, double x, double y, double z, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp);
  explicit ViewPointRep(int id, const geometry_msgs::msg::Point &position, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp);
  ~ViewPointRep() = default;

  // Handling the object node representation connection
  void AddObjectIndex(int object_index) {
    object_indices_.insert(object_index);
  }

  void DeleteObjectIndex(int object_index) {
    object_indices_.erase(object_index);
  }

  void AddDirectObjectIndex(int object_index) {
    direct_object_indices_.insert(object_index);
  }
  void DeleteDirectObjectIndex(int object_index) {
    direct_object_indices_.erase(object_index);
  }

  const std::set<int>& GetObjectIndices() const {
    return object_indices_;
  }
  const std::set<int>& GetDirectObjectIndices() const {
    return direct_object_indices_;
  }

  void ClearObjectIndices() {
    object_indices_.clear();
  }

  bool HasObjectIndex(int object_index) const {
    return object_indices_.find(object_index) != object_indices_.end();
  }
  bool HasDirectObjectIndex(int object_index) const {
    return direct_object_indices_.find(object_index) != direct_object_indices_.end();
  }

  // Handling the room node representation connection
  void SetRoomId(int room_id) {
    room_id_ = room_id;
  }
  int GetRoomId() const {
    return room_id_;
  }
  int GetId() const {
    return id_;
  }

  // Getters for position and timestamp
  const geometry_msgs::msg::Point& GetPosition() const {
    return position_;
  }
  
  const rclcpp::Time& GetTimestamp() const {
    return timestamp_;
  }

// private:
  int id_; // Unique ID for the viewpoint
  geometry_msgs::msg::Point position_;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr covered_cloud_;
  // Add the vector to store the objects
  std::set<int> object_indices_; // Indices of objects that can be seen from this viewpoint
  std::set<int> direct_object_indices_; // Indices of objects that are directly observed from this viewpoint
  int room_id_;
  rclcpp::Time timestamp_; // Timestamp of the viewpoint
};

// Define the room node class.
class RoomNodeRep 
{
public:
  // explicit RoomNodeRep(const int id, const cv::Mat &room_mask, const geometry_msgs::msg::PolygonStamped &polygon);
  // explicit RoomNodeRep(const int id, const std::vector<cv::Point> &points, const geometry_msgs::msg::PolygonStamped &polygon);
  explicit RoomNodeRep(const int id, const std::vector<cv::Point> &points);
  // explicit RoomNodeRep(const tare_planner::msg::RoomNode msg);
  RoomNodeRep() = default;
  ~RoomNodeRep() = default;

  // void UpdateRoomNode(const cv::Mat &room_mask, const geometry_msgs::msg::PolygonStamped &polygon);
  // void UpdateRoomNode(const cv::Mat &room_mask);
  void UpdateRoomNode(const std::vector<cv::Point> &points, const geometry_msgs::msg::PolygonStamped &polygon);
  void UpdateRoomNode(const std::vector<cv::Point> &points);
  void UpdatePolygon(const geometry_msgs::msg::PolygonStamped &polygon);
  void UpdateCentroid(Eigen::Vector3f &centroid);
  void RemovePoint(const cv::Point &point);
  bool InRoom(const geometry_msgs::msg::Point &point);
  // void HeritageAttributes(const RoomNodeRep &prev_room_node);
  void UpdateRoomNode(const tare_planner::msg::RoomNode msg);
  void AddViewpointId(int viewpoint_id) {
    viewpoint_indices_.insert(viewpoint_id);
  }
  void DeleteViewpointId(int viewpoint_id) {
    viewpoint_indices_.erase(viewpoint_id);
  }
  void AddObjectIndex(int object_index)
  {
    object_indices_.insert(object_index);
  }
  void DeleteObjectIndex(int object_index)
  {
    object_indices_.erase(object_index);
  }
  void SetAnchorPoint(const geometry_msgs::msg::Point &anchor_point)
  {
    anchor_point_ = anchor_point;
  }
  const geometry_msgs::msg::Point & GetAnchorPoint() const
  {
    return anchor_point_;
  }
  void SetImage(const cv::Mat &image)
  {
    image_ = image.clone(); // Clone the image to ensure it is stored correctly
  }
  const cv::Mat &GetImage() const
  {
    return image_;
  }
  void SetRoomMask(const cv::Mat &room_mask)
  {
    cv::Mat room_mask_tmp = room_mask.clone(); // Clone the image to ensure it is stored correctly
    room_mask_tmp.convertTo(room_mask_tmp, CV_8UC1);
    // crop the room mask to only keep the non-zero part
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(room_mask_tmp, non_zero_points);
    if (non_zero_points.empty())
    {
      return;
    }
    cv::Rect rect = cv::boundingRect(non_zero_points);
    std::vector<Eigen::Vector2i> bbox(2); // 大小为2
    bbox[0] = {rect.tl().y, rect.tl().x}; // 左上角
    bbox[1] = {rect.br().y, rect.br().x}; // 右下角
    // add some margin to the bbox
    int margin = 10; // 可以根据需要调整
    bbox[0] = (bbox[0] - Eigen::Vector2i(margin, margin)).cwiseMax(Eigen::Vector2i(0, 0));
    bbox[1] = (bbox[1] + Eigen::Vector2i(margin, margin)).cwiseMin(Eigen::Vector2i(room_mask_tmp.rows - 1, room_mask_tmp.cols - 1));

    room_mask_ = room_mask_tmp.rowRange(bbox[0][0], bbox[1][0] + 1)
                              .colRange(bbox[0][1], bbox[1][1] + 1);
  }
  const cv::Mat &GetRoomMask() const
  {
    return room_mask_;
  }

  void SetLastArea(float last_area)
  {
    last_area_ = last_area;
  }
  float GetLastArea() const
  {
    return last_area_;
  }

  std::string GetRoomLabel() const
  {
    if (labels_.empty())
    {
      return "";
    }
    int max_count = 0;
    std::string label;
    for (const auto &pair : labels_)
    {
      if (pair.second > max_count)
      {
        max_count = pair.second;
        label = pair.first;
      }
    }
    return label;
  }
  void SetIsAsked()
  {
    is_asked_ = std::max(0, is_asked_ - 1);
    // MY_ASSERT(is_asked_ >= 0);
  }
  void ClearRoomLabels()
  {
    labels_.clear();
    is_labeled_ = false;
    is_asked_ = 2;
    last_area_ = 0.0f;
    voxel_num_ = 0;
    anchor_point_ = geometry_msgs::msg::Point();
    image_ = cv::Mat();
  }

  // Getters and setters for private members
  std::map<std::string, int>& GetLabelsMutable() { return labels_; }
  const std::map<std::string, int>& GetLabels() const { return labels_; }
  
  bool IsLabeled() const { return is_labeled_; }
  void SetIsLabeled(bool is_labeled) { is_labeled_ = is_labeled; }
  
  const std::set<int>& GetObjectIndices() const { return object_indices_; }
  std::set<int>& GetObjectIndicesMutable() { return object_indices_; }
  
  bool IsVisited() const { return is_visited_; }
  void SetIsVisited(bool is_visited) { is_visited_ = is_visited; }
  
  bool IsCovered() const { return is_covered_; }
  void SetIsCovered(bool is_covered) { is_covered_ = is_covered; }
  
  int GetIsAsked() const { return is_asked_; }
  void SetIsAskedValue(int is_asked) { is_asked_ = is_asked; }
  
  int GetVoxelNum() const { return voxel_num_; }
  void SetVoxelNum(int voxel_num) { voxel_num_ = voxel_num; }

  // Getters for alive and polygon
  bool IsAlive() const { return alive; }
  void SetAlive(bool is_alive) { alive = is_alive; }
  
  const geometry_msgs::msg::PolygonStamped& GetPolygon() const { return polygon_; }
  
  // Getter for id
  int GetId() const { return id_; }

  // Add methods and properties specific to RoomNodeRep here
  // For example, you might want to add methods to manage connections, properties, etc.
  int id_;
  int show_id_; // Show ID of the room, used for visualization
  // cv::Mat mask_;
  geometry_msgs::msg::PolygonStamped polygon_;
  std::vector<cv::Point> points_;
  Eigen::Vector3f centroid_;
  float area_; // Area of the room in pixels
  float last_area_; // the area of last time the room label is updated
  bool alive;
  std::set<int> neighbors_; // Set of connected room IDs

  std::set<int> viewpoint_indices_; // Set of viewpoint indices that belong to this room
  std::set<int> object_indices_; // Set of object indices that belong to this room

  // attributes should be heritable
  bool is_visited_ = false; // Whether the room has been visited
  bool is_covered_ = false; // Whether the room has been covered
  bool is_labeled_ = false; // Whether the room has been asked
  int is_asked_ = 2;        // the room node can be asked 2 times for early stop
  int voxel_num_; // Number of voxels observed in the room

  std::map<std::string, int> labels_; // Map of labels and their counts
  geometry_msgs::msg::Point anchor_point_;
  cv::Mat image_; // Image of the room
  cv::Mat room_mask_; // Room mask
  bool is_connected_;
};

// // Define the object node class.
class ObjectNodeRep 
{
public:
  explicit ObjectNodeRep(const tare_planner::msg::ObjectNode::ConstSharedPtr msg);
  ObjectNodeRep() = default;
  ~ObjectNodeRep() = default;


  // Getter methods
  int GetObjectId() const { return object_id_[0]; }
  double GetConfidence() const { return confidence_; }
  const std::string& GetLabel() const { return label_; }
  const geometry_msgs::msg::Point& GetPosition() const { return position_; }
  const sensor_msgs::msg::PointCloud2& GetCloud() const { return cloud_; }
  const bool& GetStatus() const { return status_; }

  void AddVisibleViewpoint(int viewpoint_index) {
        visible_viewpoint_indices_.insert(viewpoint_index);
    }
  const std::set<int>& GetVisibleViewpoints() const { 
        return visible_viewpoint_indices_; 
    }
  bool operator==(const ObjectNodeRep& other) const {
    return object_id_ == other.object_id_;  
  }
  // Getter for the voxels
  const std::vector<Eigen::Vector3i>& GetVoxels() const {
    return voxels_;
  }
  // Method to set the voxels
  void SetVoxels(const std::vector<Eigen::Vector3i>& voxels) {
    voxels_ = voxels;
  }
  void UpdateObjectNode(const tare_planner::msg::ObjectNode::ConstSharedPtr msg);

  int GetRoomId() const { return room_id_; }
  void SetRoomId(int room_id) { room_id_ = room_id; }
  
  // Getters and setters for is_considered flags
  bool IsConsidered() const { return is_considered_; }
  void SetIsConsidered(bool is_considered) { is_considered_ = is_considered; }
  
  bool IsConsideredStrong() const { return is_considered_strong_; }
  void SetIsConsideredStrong(bool is_considered_strong) { is_considered_strong_ = is_considered_strong; }

  // Getter for visible_viewpoint_indices_
  const std::set<int>& GetVisibleViewpointIndices() const { return visible_viewpoint_indices_; }
  std::set<int>& GetVisibleViewpointIndicesMutable() { return visible_viewpoint_indices_; }
  
  // Getter for timestamp
  const rclcpp::Time& GetTimestamp() const { return timestamp_; }

  // Public members for direct access
  std::vector<int> object_id_;
  int room_id_; // Room ID the object belongs to
  std::string label_;
  double confidence_; // Assuming confidence is a double value
  geometry_msgs::msg::Point position_;
  std::array<geometry_msgs::msg::Point, 8> bbox3d_; 
  sensor_msgs::msg::PointCloud2 cloud_;
  bool status_; // New member to track the status of the object node
  std::set<int> visible_viewpoint_indices_; // List of viewpoint indices that can see this object
  // List of voxels of the object
  std::vector<Eigen::Vector3i> voxels_; // Assuming voxels are represented as 3D integer coordinates
  rclcpp::Time timestamp_; // Timestamp of the object node
  std::string img_path_; // Path to the best image of the object
  bool is_asked_vlm_ = false; // Whether the object has been asked by VLM
  bool is_considered_ = false; // Whether the object is a target to be asked by VLM
  bool is_considered_strong_ = false; // Whether the object is a strong target to be asked by VLM
};


class Representation
{
public:
  explicit Representation(rclcpp::Node::SharedPtr nh, std::string world_frame_id = "map");
  ~Representation() = default;

  // ==================== ViewPoint Management ====================
  /**
   * @brief Add a new viewpoint to the representation
   * @return The index of the newly added viewpoint
   */
  int AddViewPointRep(const geometry_msgs::msg::Point &position, 
                      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, 
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, 
                      const rclcpp::Time &timestamp);

  /**
   * @brief Get viewpoint by index (non-const)
   * @throws std::out_of_range if index is invalid
   */
  ViewPointRep& GetViewPointRepNode(int index);

  /**
   * @brief Get viewpoint by index (const)
   * @throws std::out_of_range if index is invalid
   */
  const ViewPointRep& GetViewPointRepNode(int index) const;

  /**
   * @brief Get viewpoint position by index
   * @throws std::out_of_range if index is invalid
   */
  geometry_msgs::msg::Point GetViewPointRepNodePos(int index) const;

  /**
   * @brief Get the number of viewpoints
   */
  size_t GetViewPointCount() const;

  /**
   * @brief Get all viewpoints (const)
   */
  const std::vector<ViewPointRep>& GetViewPointReps() const;

  /**
   * @brief Get all viewpoints (non-const, use sparingly)
   */
  std::vector<ViewPointRep>& GetViewPointRepsMutable();

  /**
   * @brief Get visualization cloud for viewpoints
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr GetViewPointRepCloud() const;

  /**
   * @brief Get all covered points cloud
   */
  pcl::PointCloud<pcl::PointXYZI>::Ptr GetCoveredPointsAllCloud() const;

  // ==================== Object Node Management ====================
  /**
   * @brief Update or add object node from message
   */
  void UpdateObjectNode(const tare_planner::msg::ObjectNode::ConstSharedPtr msg);

  /**
   * @brief Get object node by ID (non-const)
   * @throws std::out_of_range if object_id not found
   */
  ObjectNodeRep& GetObjectNodeRep(int object_id);

  /**
   * @brief Get object node by ID (const)
   * @throws std::out_of_range if object_id not found
   */
  const ObjectNodeRep& GetObjectNodeRep(int object_id) const;

  /**
   * @brief Check if object node exists
   */
  bool HasObjectNode(int object_id) const;

  /**
   * @brief Get the number of object nodes
   */
  size_t GetObjectNodeCount() const;

  /**
   * @brief Get all object nodes (const)
   */
  const std::unordered_map<int, ObjectNodeRep>& GetObjectNodeRepMap() const;

  /**
   * @brief Get all object nodes (non-const, use sparingly)
   */
  std::unordered_map<int, ObjectNodeRep>& GetObjectNodeRepMapMutable();

  /**
   * @brief Get latest object node indices
   */
  const std::set<int>& GetLatestObjectNodeIndices() const;

  /**
   * @brief Get latest object node indices (mutable)
   */
  std::set<int>& GetLatestObjectNodeIndicesMutable();

  // ==================== Room Node Management ====================
  /**
   * @brief Add a new room node from RoomNode message
   * @param msg The RoomNode message
   * @return Reference to the newly created RoomNodeRep
   */
  RoomNodeRep& AddRoomNode(const tare_planner::msg::RoomNode& msg);

  /**
   * @brief Get room node by ID (non-const)
   * @throws std::out_of_range if room_id not found
   */
  RoomNodeRep& GetRoomNode(int room_id);

  /**
   * @brief Get room node by ID (const)
   * @throws std::out_of_range if room_id not found
   */
  const RoomNodeRep& GetRoomNode(int room_id) const;

  /**
   * @brief Check if room node exists
   */
  bool HasRoomNode(int room_id) const;

  /**
   * @brief Get the number of room nodes
   */
  size_t GetRoomNodeCount() const;

  /**
   * @brief Get all room nodes (const)
   */
  const std::map<int, RoomNodeRep>& GetRoomNodesMap() const;

  /**
   * @brief Get all room nodes (non-const, use sparingly)
   */
  std::map<int, RoomNodeRep>& GetRoomNodesMapMutable();

  // ==================== Relationship Management ====================
  /**
   * @brief Set the relationship between object and room
   */
  void SetObjectRoomRelation(int object_id, int new_room_id);

  /**
   * @brief Set the relationship between viewpoint and room
   */
  void SetViewpointRoomRelation(int viewpoint_id, int new_room_id);

  /**
   * @brief Update viewpoint room IDs from room mask
   */
  void UpdateViewpointRoomIdsFromMask(const cv::Mat& room_mask, 
                                      const Eigen::Vector3f& shift, 
                                      float room_resolution);

  // ==================== Serialization ====================
  /**
   * @brief Serialize the whole representation as a JSON string
   */
  std::string ToJSON() const;


private:
  // ==================== Private Members ====================
  rclcpp::Node::SharedPtr nh_;

  // Core data structures
  std::vector<ViewPointRep> viewpoint_reps_;
  std::map<int, RoomNodeRep> room_nodes_map_;
  std::unordered_map<int, ObjectNodeRep> object_node_rep_map_;
  std::set<int> latest_object_node_indices_;

  // Visualization clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr viewpoint_rep_vis_cloud_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr covered_points_all_;

  // ==================== Private Methods ====================
  void AddViewPointRepNode(const geometry_msgs::msg::Point &position, 
                          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, 
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, 
                          const rclcpp::Time &timestamp);
};

} // namespace representation_ns