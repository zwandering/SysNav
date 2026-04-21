/**
 * @file room_segmentation_node.h
 * @author Haokun Zhu (haokunz@andrew.cmu.edu)
 * @brief ROS 2 Node class header for room segmentation
 * @version 0.2
 * @date 2026-01-04
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <numeric>
#include <limits>

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/polygon_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/string.hpp"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

// Eigen
#include <Eigen/Dense>

// Custom messages
#include "tare_planner/msg/room_node.hpp"
#include "tare_planner/msg/room_node_list.hpp"
#include "representation/representation.h"


namespace room_segmentation {

// Helper structures
struct PlaneInfo {
    int id;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
    std::vector<Eigen::Vector3i> voxel_indices;
    Eigen::Vector3f normal;
    Eigen::Vector3f centroid;
    Eigen::Vector3f u_dir;
    Eigen::Vector3f v_dir;
    float width;
    float height;
    std::array<Eigen::Vector3f, 4> corners;
    bool alive = true;
    bool merged = false;
};

/**
 * @brief ROS 2 Node for room segmentation
 * 
 * This node processes laser scan data to segment rooms in an indoor environment.
 * It maintains maps of navigable space, detects walls using plane fitting,
 * and segments the environment into discrete rooms connected by doors.
 */
class RoomSegmentationNode : public rclcpp::Node {
public:
    explicit RoomSegmentationNode();
    ~RoomSegmentationNode() = default;

private:
    // ==================== Callback Functions ====================
    void laserCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void occupiedCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void freespaceCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void stateEstimationCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
    void keyboardInputCallback(const std_msgs::msg::String::ConstSharedPtr msg);
    void timerCallback();

    // ==================== Core Processing Functions ====================
    void roomSegmentation();
    cv::Mat getWall(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud);
    void updateVoxelMap(const std::vector<Eigen::Vector3f> &navigable_points);
    void updateStateVoxel();
    void updateFreespace(pcl::PointCloud<pcl::PointXYZI>::Ptr &freespace_cloud_tmp);
    void updateRooms(cv::Mat &room_mask_cropped, cv::Mat &room_mask_new,  cv::Mat &room_mask_vis_cropped, int &room_number);

    // ==================== Helper Functions ====================
    void mergePlanes(std::vector<PlaneInfo> &plane_infos, int idx_0, PlaneInfo &compare);
    bool isPlaneSame(const PlaneInfo &a, const PlaneInfo &b);
    bool isRoomConnected(const int &room_id, const int &current_room_id, 
                        const Eigen::MatrixXi &adjacency_matrix);
    geometry_msgs::msg::PolygonStamped computePolygonFromMaskCropped(const cv::Mat &mask);
    void saveImageToFile(const cv::Mat &image, const std::string &filename, bool force_save = false);
    int toIndex(int x, int y, int z);
    int toIndex(int x, int y);

    // ==================== Publishing Functions ====================
    void publishRoomNodes();
    void publishRoomPolygon();
    void publishDoorCloud();

    // ==================== ROS2 Interfaces ====================
    
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_laser_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_occupied_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_freespace_cloud_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_state_estimation_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_keyboard_input_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_explored_area_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_room_mask_vis_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_room_mask_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_door_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_1_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr pub_room_boundary_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr pub_room_boundary_tmp_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_polygon_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_wall_cloud_;
    rclcpp::Publisher<tare_planner::msg::RoomNode>::SharedPtr pub_room_node_;
    rclcpp::Publisher<tare_planner::msg::RoomNodeList>::SharedPtr pub_room_node_list_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_room_map_cloud_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // ==================== Parameters ====================
    float explored_area_voxel_size_;
    float room_resolution_;
    float room_resolution_inv_;
    float occupancy_grid_resolution_;
    float ceiling_height_;
    float ceiling_height_base_; // this is the ceiling height relative to robot base height
    float wall_thres_height_;
    float wall_thres_height_base_; // this is the wall threshold height relative to robot base height
    float outward_distance_0_; // this controls the extent to which the wall dilates along its own direction.
    float outward_distance_1_; // this controls the extent to which the wall dilates along its normal direction.
    float distance_threshold_;
    float distance_angel_threshold_;
    float angle_threshold_deg_;
    float region_growing_radius_; // radius for region growing segmentation
    int dilation_iteration_; // number of dilation iterations when pre-processing for watershed
    int min_room_size_; // minimum room size in number of pixels after dilation
    int exploredAreaDisplayInterval_; // interval for publishing explored area (x10)
    int normal_search_num_;  // Number of nearest neighbors for normal estimation
    float normal_search_radius_;  // Search radius for normal estimation
    float kViewPointCollisionMarginZPlus_;
    float kViewPointCollisionMarginZMinus_;
    bool is_debug_; // whether to save debug images
    std::vector<int> room_voxel_dimension_;

    // ==================== Point Clouds ====================
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laser_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr laser_cloud_tmp_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr explored_area_cloud_tmp_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr downsampled_explored_area_cloud_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr downsampled_explored_area_cloud_tmp_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr downsampled_ceiling_cloud_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr downsampled_ceiling_cloud_tmp_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr in_range_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr occupied_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr updated_voxel_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr freespace_cloud_;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr door_cloud_;

    // ==================== Filters ====================
    pcl::VoxelGrid<pcl::PointXYZINormal> explored_area_dwz_filter_;
    pcl::VoxelGrid<pcl::PointXYZINormal> ceiling_cloud_dwz_filter_;
    pcl::PassThrough<pcl::PointXYZINormal> ceiling_pass_filter_;  // Reusable filter for ceiling height
    pcl::PassThrough<pcl::PointXYZI> occupied_pass_filter_;  // Reusable filter for occupied cloud
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree_;

    // ==================== Maps and Data Structures ====================
    std::vector<int> navigable_voxels_; // voxels that save the point cloud number in each voxel
    std::vector<int> state_voxels_; // voxels that save the freespace state in each voxel (used to tackle lidar reflection issue)
    std::vector<int> freespace_indices_; // indices of free voxels 
    cv::Mat navigable_map_all_; // the whole top-down navigable map that aggregate along the z axis
    cv::Mat navigable_map_; // the cropped navigable_map_all_
    cv::Mat wall_hist_all_; // the whole top-down wall histogram map
    cv::Mat wall_hist_; // the cropped wall_hist_all_
    cv::Mat state_map_all_;
    cv::Mat state_map_; // cropped state_map_all_
    cv::Mat room_mask_; // the int-value room_mask (each room has a unique integer id, 0 means background and -1 means edges between rooms)
    cv::Mat room_mask_vis_; // the visualizable room mask (each room has a unique color)
    std::vector<Eigen::Vector2i> bbox_; // bounding box of the current navigable map crop
    std::vector<Eigen::Vector3f> ceiling_points_;
    std::vector<PlaneInfo> plane_infos_; // detected plane information
    std::map<int, representation_ns::RoomNodeRep> room_nodes_map_; // room id to RoomNodeRep
    int room_node_counter_; // counter to assign unique ids to room nodes
    
    // ==================== State Variables ====================
    Eigen::Vector3f shift_; // shift from the left-down or right-up corner to the center of the _all_ maps
    geometry_msgs::msg::Point robot_position_; // robot position in world frame
    int explored_area_display_count_; // counter for publishing explored area
    bool segment_flag_; // flag to trigger room segmentation
    bool demo_frozen_; // when true, stop updating and keep republishing cached results
    int demo_publish_count_; // counter for throttling publish rate while frozen
};

} // namespace room_segmentation
