/**
 * @file room_segmentation.cpp
 * @author Haokun Zhu (haokunz@andrew.cmu.edu)
 * @brief ROS 2 Node implementation for room segmentation
 * @version 0.2
 * @date 2026-01-04
 * @copyright Copyright (c) 2026
 */

#include "room_segmentation/room_segmentation_node.h"

namespace room_segmentation {

// ==================== Constructor ====================
RoomSegmentationNode::RoomSegmentationNode() 
    : Node("room_segmentation_node"),
      explored_area_voxel_size_(0.1f),
      room_resolution_(0.1f),
      occupancy_grid_resolution_(0.2f),
      ceiling_height_base_(2.0f),
      wall_thres_height_base_(0.1f),
      outward_distance_0_(0.5f),
      outward_distance_1_(0.3f),
      distance_threshold_(2.5f),
      distance_angel_threshold_(0.3f),
      angle_threshold_deg_(6.0f),
      region_growing_radius_(15.0f),
      dilation_iteration_(4),
      min_room_size_(40),
      exploredAreaDisplayInterval_(1),
      normal_search_num_(50),
      normal_search_radius_(0.5f),
      kViewPointCollisionMarginZPlus_(0.5f),
      kViewPointCollisionMarginZMinus_(0.5f),
      is_debug_(false),
      room_voxel_dimension_({200, 200, 50}),
      explored_area_display_count_(0),
      room_node_counter_(0),
      segment_flag_(false),
      demo_frozen_(false),
      demo_publish_count_(0)
{
    RCLCPP_INFO(this->get_logger(), "Initializing Room Segmentation Node...");

    // ==================== Declare and Get Parameters ====================
    this->declare_parameter<float>("exploredAreaVoxelSize", 0.1f);
    this->declare_parameter<float>("room_resolution", 0.1f);
    this->declare_parameter<float>("rolling_occupancy_grid.resolution_x", 0.2f);
    this->declare_parameter<int>("room_x", 200);
    this->declare_parameter<int>("room_y", 200);
    this->declare_parameter<int>("room_z", 50);
    this->declare_parameter<float>("ceilingHeight_", 2.0f);
    this->declare_parameter<float>("wall_thres_height_", 0.1f);
    this->declare_parameter<int>("exploredAreaDisplayInterval", 1);
    this->declare_parameter<int>("dilation_iteration", 4);
    this->declare_parameter<float>("outward_distance_0", 0.5f);
    this->declare_parameter<float>("outward_distance_1", 0.3f);
    this->declare_parameter<float>("distance_threshold", 2.5f);
    this->declare_parameter<float>("distance_angel_threshold", 0.3f);
    this->declare_parameter<float>("angle_threshold_deg", 6.0f);
    this->declare_parameter<float>("region_growing_radius", 15.0f);
    this->declare_parameter<int>("min_room_size", 40);
    this->declare_parameter<int>("normal_search_num", 50);
    this->declare_parameter<float>("normal_search_radius", 0.5f);
    this->declare_parameter<float>("kViewPointCollisionMarginZPlus", 0.5f);
    this->declare_parameter<float>("kViewPointCollisionMarginZMinus", 0.5f);
    this->declare_parameter<bool>("isDebug", false);

    this->get_parameter("exploredAreaVoxelSize", explored_area_voxel_size_);
    this->get_parameter("room_resolution", room_resolution_);
    this->get_parameter("rolling_occupancy_grid.resolution_x", occupancy_grid_resolution_);
    this->get_parameter("room_x", room_voxel_dimension_[0]);
    this->get_parameter("room_y", room_voxel_dimension_[1]);
    this->get_parameter("room_z", room_voxel_dimension_[2]);
    this->get_parameter("ceilingHeight_", ceiling_height_base_);
    this->get_parameter("wall_thres_height_", wall_thres_height_base_);
    this->get_parameter("exploredAreaDisplayInterval", exploredAreaDisplayInterval_);
    this->get_parameter("dilation_iteration", dilation_iteration_);
    this->get_parameter("outward_distance_0", outward_distance_0_);
    this->get_parameter("outward_distance_1", outward_distance_1_);
    this->get_parameter("distance_threshold", distance_threshold_);
    this->get_parameter("distance_angel_threshold", distance_angel_threshold_);
    this->get_parameter("angle_threshold_deg", angle_threshold_deg_);
    this->get_parameter("region_growing_radius", region_growing_radius_);
    this->get_parameter("min_room_size", min_room_size_);
    this->get_parameter("normal_search_num", normal_search_num_);
    this->get_parameter("normal_search_radius", normal_search_radius_);
    this->get_parameter("kViewPointCollisionMarginZPlus", kViewPointCollisionMarginZPlus_);
    this->get_parameter("kViewPointCollisionMarginZMinus", kViewPointCollisionMarginZMinus_);
    this->get_parameter("isDebug", is_debug_);

    room_resolution_inv_ = 1.0f / room_resolution_;
    ceiling_height_ = ceiling_height_base_;
    wall_thres_height_ = wall_thres_height_base_;

    // ==================== Initialize Point Clouds ====================
    laser_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    laser_cloud_tmp_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    explored_area_cloud_tmp_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    downsampled_explored_area_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    downsampled_explored_area_cloud_tmp_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    downsampled_ceiling_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    downsampled_ceiling_cloud_tmp_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    in_range_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    occupied_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    updated_voxel_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    freespace_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    door_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());

    // ==================== Initialize Filters ====================
    explored_area_dwz_filter_.setLeafSize(explored_area_voxel_size_, 
                                          explored_area_voxel_size_, 
                                          explored_area_voxel_size_);
    ceiling_cloud_dwz_filter_.setLeafSize(explored_area_voxel_size_, 
                                          explored_area_voxel_size_, 
                                          explored_area_voxel_size_);
    ceiling_pass_filter_.setFilterFieldName("z");  // Set filter field once
    occupied_pass_filter_.setFilterFieldName("z");  // Set filter field once
    tree_.reset(new pcl::search::KdTree<pcl::PointXYZINormal>());

    // ==================== Initialize Maps ====================
    shift_ = Eigen::Vector3f(room_voxel_dimension_[0] / 2.0f,
                            room_voxel_dimension_[1] / 2.0f,
                            room_voxel_dimension_[2] / 2.0f);

    navigable_voxels_.resize(room_voxel_dimension_[0] * 
                            room_voxel_dimension_[1] * 
                            room_voxel_dimension_[2], 0);
    state_voxels_.resize(room_voxel_dimension_[0] * 
                        room_voxel_dimension_[1] * 
                        room_voxel_dimension_[2], -1);

    navigable_map_all_ = cv::Mat::zeros(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_32F);
    wall_hist_all_ = cv::Mat::zeros(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_32F);
    state_map_all_ = cv::Mat::zeros(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_8U);
    room_mask_ = cv::Mat::zeros(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_32S);
    room_mask_vis_ = cv::Mat::zeros(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_8UC3);
    room_mask_vis_.setTo(cv::Scalar(255, 255, 255));

    bbox_.emplace_back(Eigen::Vector2i(0, 0));
    bbox_.emplace_back(Eigen::Vector2i(room_voxel_dimension_[0] - 1, room_voxel_dimension_[1] - 1));

    robot_position_ = geometry_msgs::msg::Point();
    robot_position_.x = 0.0;
    robot_position_.y = 0.0;
    robot_position_.z = 0.0;

    // ==================== Create Subscriptions ====================
    sub_laser_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 20,
        std::bind(&RoomSegmentationNode::laserCloudCallback, this, std::placeholders::_1));

    sub_state_estimation_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimation", 20,
        std::bind(&RoomSegmentationNode::stateEstimationCallback, this, std::placeholders::_1));

    sub_occupied_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/occupied_cloud", 20,
        std::bind(&RoomSegmentationNode::occupiedCloudCallback, this, std::placeholders::_1));

    sub_freespace_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/freespace_cloud", 20,
        std::bind(&RoomSegmentationNode::freespaceCloudCallback, this, std::placeholders::_1));

    sub_keyboard_input_ = this->create_subscription<std_msgs::msg::String>(
        "/keyboard_input", 5,
        std::bind(&RoomSegmentationNode::keyboardInputCallback, this, std::placeholders::_1));

    // ==================== Create Publishers ====================
    pub_explored_area_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/explore_areas_new", 5);
    pub_room_mask_vis_ = this->create_publisher<sensor_msgs::msg::Image>("/room_mask_vis", 5);
    pub_room_mask_ = this->create_publisher<sensor_msgs::msg::Image>("/room_mask", 5);
    pub_door_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/door_cloud", 5);
    pub_debug_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_cloud", 5);
    pub_debug_1_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/free_cloud_1", 5);
    pub_room_boundary_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/current_room_boundary", 5);
    pub_room_boundary_tmp_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/navigation_boundary", 5);
    pub_polygon_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/room_boundaries", 50);
    pub_wall_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/walls", 5);
    pub_room_node_ = this->create_publisher<tare_planner::msg::RoomNode>("/room_nodes", 50);
    pub_room_node_list_ = this->create_publisher<tare_planner::msg::RoomNodeList>("/room_nodes_list", 5);
    pub_room_map_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/room_map_cloud", 5);

    // ==================== Create Timer ====================
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&RoomSegmentationNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Room Segmentation Node initialized successfully!");
}

// ==================== Timer Callback ====================
void RoomSegmentationNode::timerCallback() {
    if (demo_frozen_) {
        // Demo mode: skip segmentation and republish the cached results at a
        // throttled rate (~2 Hz given a 100 ms timer period), matching the
        // approximate rate of the normal segmentation pipeline.
        demo_publish_count_++;
        if (demo_publish_count_ >= 5) {
            demo_publish_count_ = 0;
            publishRoomNodes();
            publishDoorCloud();
            publishRoomPolygon();
        }
        return;
    }
    if (segment_flag_) {
        segment_flag_ = false;
        roomSegmentation();
        publishRoomNodes();
        publishDoorCloud();
        publishRoomPolygon();
    }
}

// ==================== Helper Functions ====================
void RoomSegmentationNode::saveImageToFile(const cv::Mat &image, const std::string &filename, bool force_save) {
    if (is_debug_ || force_save) {
        cv::Mat flipped_image;
        cv::transpose(image, flipped_image);
        cv::flip(flipped_image, flipped_image, 0);
        if (!flipped_image.empty()) {
            if (cv::imwrite(filename, flipped_image)) {
                return;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save image: %s", filename.c_str());
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Image is empty, cannot save: %s", filename.c_str());
        }
    }
}

int RoomSegmentationNode::toIndex(int x, int y, int z) {
    return x * room_voxel_dimension_[1] * room_voxel_dimension_[2] +
           y * room_voxel_dimension_[2] + z;
}

int RoomSegmentationNode::toIndex(int x, int y) {
    return x * room_voxel_dimension_[1] + y;
}

bool RoomSegmentationNode::isPlaneSame(const PlaneInfo &a, const PlaneInfo &b) {
    float angle = std::acos(std::abs(a.normal.dot(b.normal))) * 180.0f / M_PI;
    if (angle > angle_threshold_deg_)
        return false;
    
    Eigen::Vector3f centroid_diff = b.centroid - a.centroid;
    float angel_distance = centroid_diff.dot(a.normal);
    if (std::abs(angel_distance) > distance_angel_threshold_)
        return false;
    
    float center_dist = (a.centroid - b.centroid).norm();
    float actual_dist = center_dist - (a.width + b.width) / 2.0f;
    if (actual_dist > distance_threshold_)
        return false;
    
    return true;
}

bool RoomSegmentationNode::isRoomConnected(const int &room_id, const int &current_room_id, 
                                           const Eigen::MatrixXi &adjacency_matrix) {
    int n = adjacency_matrix.rows();
    if (room_id < 0 || current_room_id < 0 || room_id >= n || current_room_id >= n) {
        return false;
    }

    std::vector<bool> visited(n, false);
    std::queue<int> q;
    q.push(current_room_id);
    visited[current_room_id] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        if (u == room_id)
            return true;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && adjacency_matrix(u, v) > 0) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    return false;
}

geometry_msgs::msg::PolygonStamped RoomSegmentationNode::computePolygonFromMaskCropped(const cv::Mat &mask) {
    std::vector<std::vector<cv::Point>> current_room_contours;
    cv::findContours(mask, current_room_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<cv::Point> largest_contour;
    if (!current_room_contours.empty()) {
        largest_contour = current_room_contours[0];
        for (const auto &contour : current_room_contours) {
            if (cv::contourArea(contour) > cv::contourArea(largest_contour)) {
                largest_contour = contour;
            }
        }
    }
    
    geometry_msgs::msg::PolygonStamped boundary_polygon;
    boundary_polygon.header.frame_id = "map";
    boundary_polygon.polygon.points.clear();
    
    for (const auto &pt : largest_contour) {
        Eigen::Vector3i pt_voxel(pt.y, pt.x, 0);
        Eigen::Vector3f pt_position = misc_utils_ns::voxel_to_point_cropped(pt_voxel, shift_, room_resolution_, bbox_);
        
        geometry_msgs::msg::Point32 point;
        point.x = pt_position.x();
        point.y = pt_position.y();
        point.z = 0.0;
        boundary_polygon.polygon.points.push_back(point);
    }
    return boundary_polygon;
}

// ==================== State Estimation Callback ====================
void RoomSegmentationNode::stateEstimationCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    robot_position_ = msg->pose.pose.position;
}

// ==================== Keyboard Input Callback ====================
void RoomSegmentationNode::keyboardInputCallback(const std_msgs::msg::String::ConstSharedPtr msg) {
    if (msg->data == "demo" && !demo_frozen_) {
        demo_frozen_ = true;
        demo_publish_count_ = 0;
        RCLCPP_INFO(this->get_logger(),
                    "Demo mode enabled: room segmentation frozen, republishing cached results.");
    }
    else if (msg->data == "resume" && demo_frozen_) {
        demo_frozen_ = false;
        demo_publish_count_ = 0;
        RCLCPP_INFO(this->get_logger(),
                    "Demo mode disabled: room segmentation resumed.");
    }
}

// ==================== Callback Functions ====================
void RoomSegmentationNode::laserCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    if (demo_frozen_) {
        return;
    }
    laser_cloud_->clear();
    laser_cloud_tmp_->clear();
    pcl::fromROSMsg(*msg, *laser_cloud_tmp_);
    // Transform PointXYZI to PointXYZINormal
    pcl::copyPointCloud(*laser_cloud_tmp_, *laser_cloud_);

    *explored_area_cloud_tmp_ += *laser_cloud_;

    explored_area_display_count_++;

    if (explored_area_display_count_ >= 5 * exploredAreaDisplayInterval_)
    {
        // downsample explored area cloud
        downsampled_explored_area_cloud_tmp_->clear();
        explored_area_dwz_filter_.setInputCloud(explored_area_cloud_tmp_);
        explored_area_dwz_filter_.filter(*downsampled_explored_area_cloud_tmp_);
        *downsampled_explored_area_cloud_ += *downsampled_explored_area_cloud_tmp_; // accumulate downsampled cloud

        ceiling_height_ = ceiling_height_base_ + robot_position_.z;
        wall_thres_height_ = wall_thres_height_base_ + robot_position_.z;

        // filter the pointcloud below the ceiling height
        ceiling_pass_filter_.setInputCloud(downsampled_explored_area_cloud_tmp_);
        ceiling_pass_filter_.setFilterLimits(-std::numeric_limits<float>::max(), ceiling_height_);
        ceiling_pass_filter_.filter(*downsampled_ceiling_cloud_tmp_);

        // before adding the new point, record the index size
        int start_idx = downsampled_ceiling_cloud_->size();
        *downsampled_ceiling_cloud_ += *downsampled_ceiling_cloud_tmp_;
        int end_idx = downsampled_ceiling_cloud_->size();

        // find the old points that will be affected by the new points
        tree_->setInputCloud(downsampled_ceiling_cloud_);
        std::unordered_set<int> affected_old_indices;

        for (const auto &pt : downsampled_ceiling_cloud_tmp_->points)
        {
            std::vector<int> indices;
            std::vector<float> dists;
            // tree_->radiusSearch(pt, normal_search_radius_, indices, dists);
            tree_->nearestKSearch(pt, normal_search_num_, indices, dists);
            for (int idx : indices)
            {
                if (idx < start_idx) // only consider old points
                    affected_old_indices.insert(idx);
            }
        }

        // create the points that need to be updated normals
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr points_to_update(new pcl::PointCloud<pcl::PointXYZINormal>);
        points_to_update->points.reserve(affected_old_indices.size() + (end_idx - start_idx));
        std::vector<int> update_indices;

        for (int idx : affected_old_indices)
        {
            points_to_update->points.push_back(downsampled_ceiling_cloud_->points[idx]);
            update_indices.push_back(idx);
        }
        for (int idx = start_idx; idx < end_idx; ++idx)
        {
            points_to_update->points.push_back(downsampled_ceiling_cloud_->points[idx]);
            update_indices.push_back(idx);
        }

        // use the kdtree to compute normals
        pcl::NormalEstimation<pcl::PointXYZINormal, pcl::Normal> ne;
        ne.setInputCloud(points_to_update);
        ne.setSearchMethod(tree_);
        ne.setKSearch(normal_search_num_);
        pcl::PointCloud<pcl::Normal>::Ptr updated_normals(new pcl::PointCloud<pcl::Normal>);
        ne.compute(*updated_normals);

        // write back the updated normals
        for (size_t i = 0; i < update_indices.size(); ++i)
        {
            int idx = update_indices[i];
            downsampled_ceiling_cloud_->points[idx].normal_x = updated_normals->points[i].normal_x;
            downsampled_ceiling_cloud_->points[idx].normal_y = updated_normals->points[i].normal_y;
            downsampled_ceiling_cloud_->points[idx].normal_z = updated_normals->points[i].normal_z;
            downsampled_ceiling_cloud_->points[idx].curvature = updated_normals->points[i].curvature;
        }

        // convert to Eigen::Vector3f for further processing
        std::vector<Eigen::Vector3f> ceilingPoint_tmp;
        ceilingPoint_tmp.reserve(downsampled_ceiling_cloud_tmp_->points.size());
        for (const auto &pt : downsampled_ceiling_cloud_tmp_->points)
        {
            ceilingPoint_tmp.emplace_back(pt.x, pt.y, pt.z);
        }

        updateVoxelMap(ceilingPoint_tmp);

        explored_area_dwz_filter_.setInputCloud(downsampled_explored_area_cloud_);
        explored_area_dwz_filter_.filter(*downsampled_explored_area_cloud_);

        ceiling_cloud_dwz_filter_.setInputCloud(downsampled_ceiling_cloud_);
        ceiling_cloud_dwz_filter_.filter(*downsampled_ceiling_cloud_);

        sensor_msgs::msg::PointCloud2 explored_area_msg;
        pcl::toROSMsg(*downsampled_ceiling_cloud_, explored_area_msg);
        explored_area_msg.header.stamp = msg->header.stamp;
        explored_area_msg.header.frame_id = "map";
        pub_explored_area_->publish(explored_area_msg);

        // get the point cloud that is within the robot's range
        in_range_cloud_->clear();
        in_range_cloud_->points.reserve(downsampled_ceiling_cloud_->points.size() / 4); // 预估容量
        float radius_squared = region_growing_radius_ * region_growing_radius_;
        for (const auto &pt : downsampled_ceiling_cloud_->points)
        {
            float dx = pt.x - robot_position_.x;
            float dy = pt.y - robot_position_.y;
            if (dx * dx + dy * dy <= radius_squared)
            {
                in_range_cloud_->points.push_back(pt);
            }
        }
        in_range_cloud_->width = in_range_cloud_->points.size();
        in_range_cloud_->height = 1;
        in_range_cloud_->is_dense = true;

        // publish the in range cloud for debugging
        sensor_msgs::msg::PointCloud2 in_range_cloud_msg;
        pcl::toROSMsg(*in_range_cloud_, in_range_cloud_msg);
        in_range_cloud_msg.header.stamp = msg->header.stamp;
        in_range_cloud_msg.header.frame_id = "map";
        pub_debug_->publish(in_range_cloud_msg);

        explored_area_display_count_ = 0;
        explored_area_cloud_tmp_->clear();
        downsampled_ceiling_cloud_tmp_->clear();

        segment_flag_ = true;
    }
}

void RoomSegmentationNode::occupiedCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    if (demo_frozen_) {
        return;
    }
    occupied_cloud_->clear();
    pcl::fromROSMsg(*msg, *occupied_cloud_);

    float height_1 = robot_position_.z - kViewPointCollisionMarginZMinus_;
    float height_2 = robot_position_.z + kViewPointCollisionMarginZPlus_;
    // filter the occupied cloud within the height range
    occupied_pass_filter_.setInputCloud(occupied_cloud_);
    occupied_pass_filter_.setFilterLimits(height_1, height_2);
    occupied_pass_filter_.filter(*occupied_cloud_);

    if (occupied_cloud_->empty())
    {
        RCLCPP_WARN(this->get_logger(), "Occupied cloud is empty after height filtering.");
        return;
    }

    updateStateVoxel();
}

void RoomSegmentationNode::freespaceCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    if (demo_frozen_) {
        return;
    }
    freespace_cloud_->clear();
    pcl::fromROSMsg(*msg, *freespace_cloud_);
    if (freespace_cloud_->empty())
    {
        RCLCPP_WARN(this->get_logger(), "Freespace cloud is empty.");
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr freespace_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    updateFreespace(freespace_cloud_tmp);
}

// ==================== Core Processing Functions ====================
cv::Mat RoomSegmentationNode::getWall(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Extract normals from PointXYZINormal
    auto t0 = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normals->resize(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        const pcl::PointXYZINormal &pt = cloud->points[i];
        (*normals)[i].normal_x = pt.normal_x;
        (*normals)[i].normal_y = pt.normal_y;
        (*normals)[i].normal_z = pt.normal_z;   
        (*normals)[i].curvature = pt.curvature;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Region Growing clustering
    t0 = std::chrono::high_resolution_clock::now();
    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZINormal, pcl::Normal> reg;
    reg.setMinClusterSize(300);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree_);
    reg.setNumberOfNeighbours(50);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(3 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);
    reg.setSmoothModeFlag(true);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    t1 = std::chrono::high_resolution_clock::now();
    
    t0 = std::chrono::high_resolution_clock::now();
    // Process clusters to extract vertical planes
    std::vector<PlaneInfo> plane_infos_new;
    plane_infos_new.reserve(clusters.size() / 4); // Reserve estimated space
    
    for (size_t i = 0; i < clusters.size(); ++i)
    {
        const auto &cluster = clusters[i];
        
        // Compute centroid and normal sum directly from indices without copying cloud
        Eigen::Vector3f centroid(0, 0, 0);
        Eigen::Vector3f normal_sum(0, 0, 0);
        int valid_points = 0;
        
        for (int idx : cluster.indices)
        {
            const pcl::PointXYZINormal &pt = cloud->points[idx];
            centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
            
            const pcl::Normal &n = normals->points[idx];
            if (std::isfinite(n.normal_x) && std::isfinite(n.normal_y) && std::isfinite(n.normal_z))
            {
                normal_sum += Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
                valid_points++;
            }
        }
        
        if (valid_points == 0 || normal_sum.norm() < 1e-3)
            continue;
            
        centroid /= static_cast<float>(cluster.indices.size());
        Eigen::Vector3f avg_normal = normal_sum.normalized();

        // Early filtering: Remove horizontal planes
        float dot = std::abs(avg_normal.dot(Eigen::Vector3f::UnitZ()));
        if (dot > std::cos(80.0f * M_PI / 180.0f))
            continue;
        
        // Compute variance using online algorithm (single pass)
        float mean_dist = 0.0f;
        float m2 = 0.0f;
        int n = 0;
        
        for (int idx : cluster.indices)
        {
            const pcl::PointXYZINormal &pt = cloud->points[idx];
            Eigen::Vector3f p(pt.x - centroid.x(), pt.y - centroid.y(), pt.z - centroid.z());
            float dist = p.dot(avg_normal);
            
            n++;
            float delta = dist - mean_dist;
            mean_dist += delta / n;
            float delta2 = dist - mean_dist;
            m2 += delta * delta2;
        }
        
        float variance = (n > 1) ? (m2 / n) : 0.0f;
        if (variance > 0.1f)
            continue;

        avg_normal = (avg_normal - avg_normal.dot(Eigen::Vector3f::UnitZ()) * Eigen::Vector3f::UnitZ()).normalized();

        std::vector<Eigen::Vector3i> voxel_indices;
        voxel_indices.reserve(cluster.indices.size());
        
        Eigen::Vector3f u_dir = avg_normal.cross(Eigen::Vector3f::UnitZ()).normalized();
        Eigen::Vector3f v_dir = avg_normal.cross(u_dir).normalized();
        
        // Compute bounding rectangle directly from original cloud using indices
        float u_min = FLT_MAX, u_max = -FLT_MAX;
        float v_min = FLT_MAX, v_max = -FLT_MAX;
        
        for (int idx : cluster.indices)
        {
            const pcl::PointXYZINormal &point = cloud->points[idx];
            Eigen::Vector3f p(point.x, point.y, point.z);
            Eigen::Vector3f relative = p - centroid;
            float u = relative.dot(u_dir);
            float v = relative.dot(v_dir);

            u_min = std::min(u_min, u);
            u_max = std::max(u_max, u);
            v_min = std::min(v_min, v);
            v_max = std::max(v_max, v);

            voxel_indices.emplace_back(misc_utils_ns::point_to_voxel(p, shift_, room_resolution_inv_));
        }
        
        float height = v_max - v_min;
        // Early filtering: check height before further computation
        if (height < 1.5f)
            continue;
        
        // Now create cluster_cloud only for planes that passed all filters
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        cluster_cloud->points.reserve(cluster.indices.size());
        for (int idx : cluster.indices)
        {
            cluster_cloud->points.push_back(cloud->points[idx]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;
        
        // Compute corners of the bounding rectangle
        std::array<Eigen::Vector3f, 4> corners = {
            centroid + u_dir * u_min + v_dir * v_min,
            centroid + u_dir * u_max + v_dir * v_min,
            centroid + u_dir * u_max + v_dir * v_max,
            centroid + u_dir * u_min + v_dir * v_max};

        centroid = (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0f;
        float width = u_max - u_min;
            
        plane_infos_new.push_back({static_cast<int>(i),
                                   cluster_cloud,
                                   std::move(voxel_indices),
                                   avg_normal,
                                   centroid,
                                   u_dir,
                                   v_dir,
                                   width,
                                   height,
                                   corners,
                                   true,
                                   false});
    }

    t1 = std::chrono::high_resolution_clock::now();

    // Extract in-range planes from plane_infos_
    std::vector<int> in_range_plane_indices;
    for (size_t i = 0; i < plane_infos_.size(); ++i)
    {
        const auto &plane = plane_infos_[i];
        if (plane.cloud->size() > 1000)
            continue;

        float dist = std::hypot(plane.centroid.x() - robot_position_.x, plane.centroid.y() - robot_position_.y);
        if (dist < region_growing_radius_)
        {
            in_range_plane_indices.push_back(i);
            plane_infos_[i].alive = false;
        }
    }

    // Merge new planes with existing planes
    for (size_t i = 0; i < plane_infos_new.size(); ++i)
    {
        bool found = false;
        int found_idx = -1;
        for (size_t j = 0; j < plane_infos_.size(); ++j)
        {
            if (isPlaneSame(plane_infos_new[i], plane_infos_[j]))
            {
                found = true;
                found_idx = static_cast<int>(j);
                break;
            }
        }
        if (!found)
        {
            // if no similar plane found, add as a new plane
            plane_infos_new[i].id = static_cast<int>(plane_infos_.size());
            plane_infos_.push_back(plane_infos_new[i]);
        }
        else
        {
            // merge with the found plane
            mergePlanes(plane_infos_, found_idx, plane_infos_new[i]);
            plane_infos_[found_idx].alive = true;
        }
    }

    // Merge similar planes
    for (size_t i = 0; i < plane_infos_.size(); ++i)
    {
        if (!plane_infos_[i].alive)
            continue;
        for (size_t j = i + 1; j < plane_infos_.size(); ++j)
        {
            if (!plane_infos_[j].alive)
                continue;
            if (isPlaneSame(plane_infos_[i], plane_infos_[j]))
            {
                mergePlanes(plane_infos_, static_cast<int>(i), plane_infos_[j]);
                plane_infos_[j].alive = false;
                plane_infos_[i].alive = true;
            }
        }
    }

    // Remove planes where most voxels are free
    for (auto &plane : plane_infos_)
    {
        if (!plane.alive)
            continue;
        int free_count = 0;
        int total_count = 0;
        for (const auto &voxel_index : plane.voxel_indices)
        {
            if (voxel_index[0] < 0 || voxel_index[0] >= room_voxel_dimension_[0] ||
                voxel_index[1] < 0 || voxel_index[1] >= room_voxel_dimension_[1] ||
                voxel_index[2] < 0 || voxel_index[2] >= room_voxel_dimension_[2])
                continue;
            if (state_map_all_.at<uchar>(voxel_index[0], voxel_index[1]) == 1)
            {
                free_count++;
            }
            total_count++;
        }
        if (free_count > total_count * 0.33f)
        {
            plane.alive = false;
        }
    }

    // Remove non-alive planes
    plane_infos_.erase(std::remove_if(plane_infos_.begin(), plane_infos_.end(),
                                     [](const PlaneInfo &plane)
                                     { return (!plane.alive); }),
                      plane_infos_.end());

    // Visualize merged planes
    t0 = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Calculate total points for reservation
    size_t total_points = 0;
    for (const auto &plane : plane_infos_)
        total_points += plane.cloud->points.size();
    merged_cloud->points.reserve(total_points);
    
    for (const auto &plane : plane_infos_)
    {
        uint8_t r = static_cast<uint8_t>(rand() % 256);
        uint8_t g = static_cast<uint8_t>(rand() % 256);
        uint8_t b = static_cast<uint8_t>(rand() % 256);
        
        for (const auto& pt : plane.cloud->points)
        {
            merged_cloud->points.emplace_back(pt.x, pt.y, pt.z, r, g, b);
        }
    }
    merged_cloud->width = merged_cloud->points.size();
    merged_cloud->height = 1;
    merged_cloud->is_dense = true;

    sensor_msgs::msg::PointCloud2 merged_cloud_msg;
    pcl::toROSMsg(*merged_cloud, merged_cloud_msg);
    merged_cloud_msg.header.frame_id = "map";
    pub_wall_cloud_->publish(merged_cloud_msg);

    t1 = std::chrono::high_resolution_clock::now();

    auto t_end = std::chrono::high_resolution_clock::now();

    // Project walls to 2D map (combined loop for both operations)
    t0 = std::chrono::high_resolution_clock::now();
    cv::Mat wall_mask(room_voxel_dimension_[0], room_voxel_dimension_[1], CV_8U, cv::Scalar(0));

    for (const auto &plane : plane_infos_)
    {
        if (plane.merged)
            continue;

        // First pass: basic polygon
        std::vector<cv::Point> polygon_2d;
        polygon_2d.reserve(4);
        Eigen::Vector3f outward_1 = plane.normal * outward_distance_1_;

        for (int i = 0; i < 4; i++)
        {
            Eigen::Vector3f corner = plane.corners[i];
            if (i==0)
                corner = corner - outward_1;
            else if (i==1)
                corner = corner - outward_1;
            else if (i==2)
                corner = corner + outward_1;
            else
                corner = corner + outward_1;
            
            Eigen::Vector3i idx = misc_utils_ns::point_to_voxel(corner, shift_, room_resolution_inv_);
            polygon_2d.emplace_back(idx[1], idx[0]);
        }
        cv::fillPoly(wall_mask, std::vector<std::vector<cv::Point>>{polygon_2d}, cv::Scalar(255));

        // Second pass: extension along u_dir
        polygon_2d.clear();
        polygon_2d.reserve(4);

        for (int i = 0; i<2; ++i)
        {
            Eigen::Vector3f corner = plane.corners[i];
            Eigen::Vector3f pt_0 = corner;
            Eigen::Vector3f pt_1;
            if (i == 0) {
                pt_1 = corner - plane.u_dir * outward_distance_0_;
            } else {
                pt_1 = corner + plane.u_dir * outward_distance_0_;
            }
            
            Eigen::Vector3i idx_0 = misc_utils_ns::point_to_voxel(pt_0, shift_, room_resolution_inv_);
            Eigen::Vector3i idx_1 = misc_utils_ns::point_to_voxel(pt_1, shift_, room_resolution_inv_);
            cv::Point pt_2d_0(idx_0[1], idx_0[0]);
            cv::Point pt_2d_1(idx_1[1], idx_1[0]);
            
            cv::LineIterator it(wall_mask, pt_2d_0, pt_2d_1, 8);

            cv::Point pt_found = pt_2d_1;
            for (int j = 0; j < it.count; ++j, ++it)
            {
                cv::Point pt = it.pos();
                if (pt.x <= 0 || pt.x >= wall_mask.cols - 1 || pt.y <= 0 || pt.y >= wall_mask.rows - 1)
                {
                    pt_found = pt;
                    break;
                }
                if (wall_mask.at<uchar>(pt) == 255)
                {
                    pt_found = pt;
                    break;
                }
            }
            if (pt_found.x >= 0 && pt_found.y >= 0)
            {
                Eigen::Vector3i idx_found(pt_found.y, pt_found.x, 0);
                Eigen::Vector3f pt_found_world = misc_utils_ns::voxel_to_point(idx_found, shift_, room_resolution_);
                Eigen::Vector3f pt_outward_0 = pt_found_world + plane.normal * outward_distance_1_;
                Eigen::Vector3f pt_outward_1 = pt_found_world - plane.normal * outward_distance_1_;
                Eigen::Vector3i idx_outward_0 = misc_utils_ns::point_to_voxel(pt_outward_0, shift_, room_resolution_inv_);
                Eigen::Vector3i idx_outward_1 = misc_utils_ns::point_to_voxel(pt_outward_1, shift_, room_resolution_inv_);
                polygon_2d.emplace_back(idx_outward_0[1], idx_outward_0[0]);
                polygon_2d.emplace_back(idx_outward_1[1], idx_outward_1[0]);
            }
        }
        if (!polygon_2d.empty())
            cv::fillPoly(wall_mask, std::vector<std::vector<cv::Point>>{polygon_2d}, cv::Scalar(255));
    }
    
    t1 = std::chrono::high_resolution_clock::now();

    // Crop wall_mask using bbox_
    wall_mask = wall_mask.rowRange(bbox_[0][0], bbox_[1][0] + 1)
                    .colRange(bbox_[0][1], bbox_[1][1] + 1);
    saveImageToFile(wall_mask, "wall_mask_from_planes.png");

    return wall_mask;
}

void RoomSegmentationNode::updateVoxelMap(const std::vector<Eigen::Vector3f> &navigable_points) {
    // Traverse all points and accumulate to voxels
    for (const auto &pt : navigable_points)
    {
        auto idx = misc_utils_ns::point_to_voxel(pt, shift_, room_resolution_inv_);
        // Boundary clipping
        idx[0] = std::clamp(idx[0], 0, room_voxel_dimension_[0] - 1);
        idx[1] = std::clamp(idx[1], 0, room_voxel_dimension_[1] - 1);
        idx[2] = std::clamp(idx[2], 0, room_voxel_dimension_[2] - 1);
        
        if (navigable_voxels_[toIndex(idx[0], idx[1], idx[2])] == 0 &&
            state_map_all_.at<uchar>(idx[0], idx[1]) != 1) // Don't update free space again
        {
            navigable_voxels_[toIndex(idx[0], idx[1], idx[2])] = 1;
            navigable_map_all_.at<float>(idx[0], idx[1]) += 1.0f; // Accumulate to 2D map
            
            if (wall_thres_height_ < pt.z() && pt.z() < ceiling_height_)
            {
                // If point height is within wall range, mark corresponding voxel as wall
                wall_hist_all_.at<float>(idx[0], idx[1]) += 1.0f;
            }
        }
    }

    // Get the bounding box of non-zero area in navigable_map
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(navigable_map_all_, non_zero_points);
    if (non_zero_points.empty())
    {
        return;
    }
    
    cv::Rect rect = cv::boundingRect(non_zero_points);
    bbox_[0] = Eigen::Vector2i(rect.tl().y, rect.tl().x); // Top-left corner
    bbox_[1] = Eigen::Vector2i(rect.br().y, rect.br().x); // Bottom-right corner
    
    // Add some margin to the bbox
    int margin = 20;
    bbox_[0] = (bbox_[0] - Eigen::Vector2i(margin, margin)).cwiseMax(Eigen::Vector2i(0, 0));
    bbox_[1] = (bbox_[1] + Eigen::Vector2i(margin, margin)).cwiseMin(Eigen::Vector2i(room_voxel_dimension_[0] - 1, room_voxel_dimension_[1] - 1));

    // crop the navigable_map_all using the bbox to get the navigable_map
    navigable_map_ = navigable_map_all_.rowRange(bbox_[0][0], bbox_[1][0] + 1).colRange(bbox_[0][1], bbox_[1][1] + 1);
    wall_hist_ = wall_hist_all_.rowRange(bbox_[0][0], bbox_[1][0] + 1).colRange(bbox_[0][1], bbox_[1][1] + 1);
    state_map_ = state_map_all_.rowRange(bbox_[0][0], bbox_[1][0] + 1).colRange(bbox_[0][1], bbox_[1][1] + 1);
}

void RoomSegmentationNode::updateStateVoxel() {
    // use the idx store in freespace_indices_ to update the state_map_all_
    for (const auto &pt : occupied_cloud_->points)
    {
        if (pt.intensity != 0) // Occupied
        {
            continue; // Only process occupied space
        }
        auto idx = misc_utils_ns::point_to_voxel(
            Eigen::Vector3f(pt.x - room_resolution_ / 2.0 - 1e-4, 
                            pt.y - room_resolution_ / 2.0 - 1e-4, 
                            pt.z - room_resolution_ / 2.0 - 1e-4), 
            shift_, room_resolution_inv_);
        // Boundary clipping
        idx[0] = std::clamp(idx[0], 0, room_voxel_dimension_[0] - 1);
        idx[1] = std::clamp(idx[1], 0, room_voxel_dimension_[1] - 1);
        idx[2] = std::clamp(idx[2], 0, room_voxel_dimension_[2] - 1);
        
        // Check if all surrounding cells are in freespace_indices
        bool flag = true;
        for (int dx = 0; dx <= 1; ++dx)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                int nx = idx[0] + dx;
                int ny = idx[1] + dy;
                if (nx >= 0 && nx < room_voxel_dimension_[0] &&
                    ny >= 0 && ny < room_voxel_dimension_[1])
                {
                    if (std::find(freespace_indices_.begin(), freespace_indices_.end(), 
                                 toIndex(nx, ny)) == freespace_indices_.end())
                    {
                        flag = false; // If any point is not in freespace_indices, don't mark as free space
                        break;
                    }
                }
            }
        }
        
        if (flag)
        {
            // Mark surrounding area as free space
            for (int dx = -1; dx <= 2; ++dx)
            {
                for (int dy = -1; dy <= 2; ++dy)
                {
                    int nx = idx[0] + dx;
                    int ny = idx[1] + dy;
                    if (nx >= 0 && nx < room_voxel_dimension_[0] &&
                        ny >= 0 && ny < room_voxel_dimension_[1])
                    {
                        state_map_all_.at<uchar>(nx, ny) = 1; // Mark as free space
                        navigable_map_all_.at<float>(nx, ny) = 1.0f;
                        wall_hist_all_.at<float>(nx, ny) = 1.0f; // Clear wall history
                        
                        for (int z = 0; z < room_voxel_dimension_[2]; ++z)
                        {
                            int index = toIndex(nx, ny, z);
                            navigable_voxels_[index] = 0;
                        }

                        Eigen::Vector3f pt_position = misc_utils_ns::voxel_to_point(
                            Eigen::Vector3i(nx, ny, 0), shift_, room_resolution_);
                        pt_position.z() = robot_position_.z; // Keep z-axis height constant
                        pcl::PointXYZI pt_new;
                        pt_new.x = pt_position.x();
                        pt_new.y = pt_position.y();
                        pt_new.z = pt_position.z();
                        pt_new.intensity = 0; // Mark as free space
                        updated_voxel_cloud_->points.push_back(pt_new);
                    }
                }
            }
            updated_voxel_cloud_->points.push_back(pt);
        }
    }

    if (!updated_voxel_cloud_->empty())
    {
        // Publish the updated_voxel_cloud using pub_debug_
        sensor_msgs::msg::PointCloud2 occupied_cloud_msg;
        pcl::toROSMsg(*updated_voxel_cloud_, occupied_cloud_msg);
        occupied_cloud_msg.header.stamp = this->now();
        occupied_cloud_msg.header.frame_id = "map";
        pub_debug_->publish(occupied_cloud_msg);
    }
}

void RoomSegmentationNode::updateFreespace(pcl::PointCloud<pcl::PointXYZI>::Ptr &freespace_cloud_tmp)
{
    // this function only updates the navigable_voxels_, navigable_map_all_, wall_hist_all_ based on the freespace_cloud_
    // this function will store the freespace voxel indices in freespace_indices_, and it is used in updateStateVoxel() to update the state_map_all_
    freespace_indices_.clear();
    for (auto &pt : freespace_cloud_->points)
    {
        auto idx = misc_utils_ns::point_to_voxel(Eigen::Vector3f(pt.x + 1e-4, pt.y + 1e-4, pt.z + 1e-4), shift_, room_resolution_inv_);
        // check if the point is within bounds
        idx[0] = std::clamp(idx[0], 0, room_voxel_dimension_[0] - 1);
        idx[1] = std::clamp(idx[1], 0, room_voxel_dimension_[1] - 1);
        idx[2] = std::clamp(idx[2], 0, room_voxel_dimension_[2] - 1);

        for (int dx = 0; dx <= 1; ++dx)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                int nx = idx[0] + dx;
                int ny = idx[1] + dy;
                if (nx >= 0 && nx < room_voxel_dimension_[0] &&
                    ny >= 0 && ny < room_voxel_dimension_[1] &&
                    state_map_all_.at<uchar>(nx, ny) != 1)
                {
                    Eigen::Vector3f pt_position = misc_utils_ns::voxel_to_point(Eigen::Vector3i(nx, ny, idx[2]), shift_, room_resolution_);
                    pcl::PointXYZI pt_xyz;
                    pt_xyz.x = pt_position.x();
                    pt_xyz.y = pt_position.y();
                    pt_xyz.z = robot_position_.z; // 使用机器人的z坐标作为高度
                    pt_xyz.intensity = 1.0f;      // 设置强度为1.0f
                    freespace_cloud_tmp->points.push_back(pt_xyz);

                    freespace_indices_.push_back(toIndex(nx, ny)); // 记录free space的体素索引
                    for (int dz = -2; dz <= 5; ++dz)               // 只在z轴上扩展
                    {
                        int nz = idx[2] + dz;
                        if (nz >= 0 && nz < room_voxel_dimension_[2])
                        {
                            if (navigable_voxels_[toIndex(nx, ny, nz)] == 1)
                            {
                                navigable_voxels_[toIndex(nx, ny, nz)] = 0;
                                navigable_map_all_.at<float>(nx, ny) -= 1.0f;
                                float pt_z = pt.z + dz * room_resolution_; // 计算实际的z坐标
                                if (wall_thres_height_ < pt_z && pt_z < ceiling_height_)
                                    wall_hist_all_.at<float>(nx, ny) -= 1.0f; // 更新墙体地图
                            }
                        }
                    }
                }
            }
        }
    }

    // use pub_debug_1_ to publish the freespace cloud
    sensor_msgs::msg::PointCloud2 freespace_cloud_msg;
    pcl::toROSMsg(*freespace_cloud_tmp, freespace_cloud_msg);
    freespace_cloud_msg.header.stamp = this->now();
    freespace_cloud_msg.header.frame_id = "map";
    pub_debug_1_->publish(freespace_cloud_msg);
}

void RoomSegmentationNode::mergePlanes(std::vector<PlaneInfo> &plane_infos, int idx_0, PlaneInfo &compare) {
    // Merge planes
    PlaneInfo &base = plane_infos[idx_0];
    compare.merged = true;

    size_t size_0 = base.cloud->size();
    size_t size_1 = compare.cloud->size();
    base.centroid = (base.centroid * size_0 + compare.centroid * size_1) / (size_0 + size_1);
    base.cloud->insert(base.cloud->end(), compare.cloud->begin(), compare.cloud->end());
    
    // Downsample the merged cloud
    explored_area_dwz_filter_.setInputCloud(base.cloud);
    explored_area_dwz_filter_.filter(*base.cloud);
    
    // Use RANSAC to fit the normal
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*base.cloud, *merged_cloud);
    
    pcl::SACSegmentation<pcl::PointXYZINormal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setInputCloud(merged_cloud);
    
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;
    seg.segment(inliers, coefficients);
    
    if (inliers.indices.size() > 0)
    {
        Eigen::Vector3f normal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
        normal = (normal - normal.dot(Eigen::Vector3f::UnitZ()) * Eigen::Vector3f::UnitZ()).normalized(); // Keep only component perpendicular to z-axis
        base.normal = normal;
        
        // Store the inliers as the point cloud
        base.cloud->clear();
        base.voxel_indices.clear();
        for (const auto &index : inliers.indices)
        {
            pcl::PointXYZINormal pt = merged_cloud->points[index];
            base.cloud->push_back(pt);
            Eigen::Vector3f pt_f(pt.x, pt.y, pt.z);
            Eigen::Vector3i voxel_index = misc_utils_ns::point_to_voxel(pt_f, shift_, room_resolution_inv_);
            base.voxel_indices.push_back(voxel_index);
        }
    }
    
    base.u_dir = base.normal.cross(Eigen::Vector3f::UnitZ()).normalized();
    base.v_dir = base.normal.cross(base.u_dir).normalized();
    
    // Update bounding box
    // Treat point cloud as points on a plane, compute the four corner points
    Eigen::Vector4f centroid4f;
    pcl::compute3DCentroid(*base.cloud, centroid4f);
    base.centroid = centroid4f.head<3>();

    float u_min = FLT_MAX, u_max = -FLT_MAX;
    float v_min = FLT_MAX, v_max = -FLT_MAX;
    for (const auto &point : base.cloud->points)
    {
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f relative = p - base.centroid;
        float u = relative.dot(base.u_dir);
        float v = relative.dot(base.v_dir);

        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }
    
    // Four corner points (counter-clockwise, starting from bottom-left)
    std::array<Eigen::Vector3f, 4> corners = {
        base.centroid + base.u_dir * u_min + base.v_dir * v_min,
        base.centroid + base.u_dir * u_max + base.v_dir * v_min,
        base.centroid + base.u_dir * u_max + base.v_dir * v_max,
        base.centroid + base.u_dir * u_min + base.v_dir * v_max};

    base.centroid = (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0f;
    base.corners = corners;

    // Compute plane width and height
    base.width = u_max - u_min;
    base.height = v_max - v_min;
}

void RoomSegmentationNode::updateRooms(cv::Mat &room_mask_cropped, cv::Mat &room_mask_new,
                                       cv::Mat &room_mask_vis_cropped, int &room_number) {
    // Step 1: Only keep regions in room_mask_cropped that are non-zero in room_mask_new
    cv::Mat mask = (room_mask_new != 0);
    room_mask_cropped.setTo(0, mask == 0);

    // Create a set of all room IDs from 1 to room_number that need processing
    std::set<int> room_need_process_ids;
    for (int i = 1; i <= room_number; ++i)
    {
        room_need_process_ids.insert(i);
    }
    
    Eigen::Vector3d bg_color_ = misc_utils_ns::idToColor(0); // Background color is white
    cv::Vec3b bg_color = cv::Vec3b(bg_color_[0], bg_color_[1], bg_color_[2]);
    Eigen::Vector3d color_;
    cv::Vec3b color;

    room_mask_vis_cropped.setTo(bg_color, mask == 0);
    
    // Check each room node, three cases:
    // 1. If most of room's mask in room_mask_new is 0, the room is invalid or merged, should be deleted
    // 2. If room's mask in room_mask_new has only 1 non-zero value, compare old and new masks
    // 3. If room's mask in room_mask_new has multiple non-zero values, the old room was split
    
    if (room_mask_new.size() != room_mask_cropped.size())
    {
        throw std::runtime_error("room_mask_new and room_mask must have the same size");
    }

    std::set<int> room_ids_to_remove;
    std::vector<representation_ns::RoomNodeRep> room_nodes_new;
    
    for (auto &id_room_node_pair : room_nodes_map_)
    {   
        int room_id = id_room_node_pair.first;
        representation_ns::RoomNodeRep &room_node = id_room_node_pair.second;
        std::vector<cv::Point> non_zero_points_old = room_node.points_;
        
        if (non_zero_points_old.empty())
        {
            room_node.SetAlive(false);
            room_ids_to_remove.insert(room_id);
            std::cout << "Room " << room_id << " erase 1" << std::endl;
            continue;
        }

        std::vector<cv::Point> non_zero_points_old_cropped;
        non_zero_points_old_cropped.reserve(non_zero_points_old.size());
        cv::Mat mask_old = cv::Mat::zeros(room_mask_cropped.size(), CV_8U);
        
        for (const auto &pt : non_zero_points_old)
        {
            mask_old.at<uchar>(pt.y - bbox_[0][0], pt.x - bbox_[0][1]) = 1;
            cv::Point pt_cropped(pt.x - bbox_[0][1], pt.y - bbox_[0][0]);
            non_zero_points_old_cropped.push_back(pt_cropped);
        }
        int value_old = room_node.GetId();

        bool all_zero = true;
        int zero_count = 0;
        std::set<int> values_set = {};
        std::unordered_map<int, int> value_count_map = {};
        std::set<int> values_set_old;
        
        for (const auto &pt : non_zero_points_old_cropped)
        {   
            int value = room_mask_new.at<int>(pt.y, pt.x);
            int value_old_tmp = room_mask_cropped.at<int>(pt.y, pt.x);
            if (value > 0)
            {
                values_set.insert(value);
                value_count_map[value]++;
                all_zero = false;
            }
            else
            {
                zero_count++;
            }
            if (value_old_tmp > 0)
            {
                values_set_old.insert(value_old_tmp);
            }
        }
        
        // Case 1: Room is mostly zero or invalid
        if (all_zero || zero_count > non_zero_points_old.size() * 0.8 || 
            values_set_old.find(value_old) == values_set_old.end())
        {
            room_mask_cropped.setTo(0, mask_old);
            room_node.SetAlive(false);
            room_ids_to_remove.insert(room_id);
            std::cout << "Room " << room_id << " erase 2" << std::endl;
            RCLCPP_INFO(this->get_logger(),
                        "Room %d: all_zero = %d, zero_count = %d, value_found = %d.",
                        room_id, all_zero, (zero_count > non_zero_points_old.size() * 0.8), 
                        (values_set_old.find(value_old) == values_set_old.end()));
        }
        // Case 2: Room has single non-zero value
        else if (values_set.size() == 1 && *values_set.begin() != 0)
        {
            int value_new = *values_set.begin();
            cv::Mat mask_new = (room_mask_new == value_new);
            
            if (cv::countNonZero(mask_new ^ mask_old) == 0)
            {
                room_mask_new.setTo(0, mask_old);
                room_need_process_ids.erase(value_new);
                continue;
            }
            else
            {
                room_mask_cropped.setTo(0, mask_old);
                room_mask_vis_cropped.setTo(bg_color, mask_old);
                
                std::vector<cv::Point> non_zero_points_new;
                cv::findNonZero(mask_new, non_zero_points_new);

                room_mask_cropped.setTo(value_old, mask_new);
                color_ = misc_utils_ns::idToColor(value_old);
                color = cv::Vec3b(color_[0], color_[1], color_[2]);
                room_mask_vis_cropped.setTo(color, mask_new);
                room_mask_new.setTo(0, mask_new);

                for (auto &pt : non_zero_points_new)
                {
                    pt.x += bbox_[0][1];
                    pt.y += bbox_[0][0];
                }
                room_node.UpdateRoomNode(non_zero_points_new);
                room_need_process_ids.erase(value_new);
            }
        }
        // Case 3: Room has multiple non-zero values (split)
        else if (values_set.size() > 1)
        {
            std::vector<int> non_zero_values;
            for (const auto &value : values_set)
            {
                if (value != 0)
                    non_zero_values.push_back(value);
            }
            
            // Find value with maximum count
            int max_area_value = 0;
            int max_area = 0;
            for (const auto &pair : value_count_map)
            {
                if (pair.second > max_area)
                {
                    max_area_value = pair.first;
                    max_area = pair.second;
                }
            }
            
            cv::Mat mask_new = (room_mask_new == max_area_value);
            room_mask_cropped.setTo(0, mask_old);
            room_mask_vis_cropped.setTo(bg_color, mask_old);

            room_mask_cropped.setTo(value_old, mask_new);
            color_ = misc_utils_ns::idToColor(value_old);
            color = cv::Vec3b(color_[0], color_[1], color_[2]);
            room_mask_vis_cropped.setTo(color, mask_new);
            room_mask_new.setTo(0, mask_new);

            std::vector<cv::Point> non_zero_points_new;
            cv::findNonZero(mask_new, non_zero_points_new);
            for (auto &pt : non_zero_points_new)
            {
                pt.x += bbox_[0][1];
                pt.y += bbox_[0][0];
            }
            room_node.UpdateRoomNode(non_zero_points_new);
            room_need_process_ids.erase(max_area_value);

            // Process other values
            for (const auto &value : non_zero_values)
            {
                if (value == max_area_value)
                    continue;

                cv::Mat mask_other;
                cv::bitwise_and((room_mask_new == value), (room_mask_cropped == 0), mask_other);
                std::vector<cv::Point> non_zero_points_other;
                cv::findNonZero(mask_other, non_zero_points_other);

                cv::Mat mask_other_tmp = (room_mask_new == value);
                std::vector<cv::Point> non_zero_points_other_tmp;
                cv::findNonZero(mask_other_tmp, non_zero_points_other_tmp);
                
                if (((float)non_zero_points_other.size() / (non_zero_points_other_tmp.size() + 1e-5) > 0.5) && 
                    non_zero_points_other.size() > 0)
                {
                    room_node_counter_++;
                    int new_room_id = room_node_counter_;
                    room_mask_cropped.setTo(new_room_id, mask_other_tmp);
                    Eigen::Vector3d new_color_ = misc_utils_ns::idToColor(new_room_id);
                    cv::Vec3b new_color = cv::Vec3b(new_color_[0], new_color_[1], new_color_[2]);
                    room_mask_vis_cropped.setTo(new_color, mask_other_tmp);
                    room_mask_new.setTo(0, mask_other_tmp);

                    std::vector<cv::Point> non_zero_points_new;
                    cv::findNonZero(mask_other_tmp, non_zero_points_new);
                    for (auto &pt : non_zero_points_new)
                    {
                        pt.x += bbox_[0][1];
                        pt.y += bbox_[0][0];
                    }

                    representation_ns::RoomNodeRep new_room_node(new_room_id, non_zero_points_new);
                    room_nodes_new.push_back(new_room_node);
                    room_need_process_ids.erase(value);
                }
            }
        }
    }

    // Add new room nodes
    for (auto &new_room_node : room_nodes_new)
    {
        int new_room_id = new_room_node.GetId();
        if (room_nodes_map_.find(new_room_id) == room_nodes_map_.end())
        {
            room_nodes_map_[new_room_id] = new_room_node;
        }
        else
        {
            throw std::runtime_error("New room id already exists in room_nodes_map: " + std::to_string(new_room_id));
        }
    }
    
    // Process remaining room IDs
    for (const auto &room_id : room_need_process_ids)
    {
        cv::Mat mask_new = (room_mask_new == room_id);
        if (cv::countNonZero(mask_new) > 0)
        {
            room_node_counter_++;
            int new_room_id = room_node_counter_;
            room_mask_cropped.setTo(new_room_id, mask_new);
            Eigen::Vector3d new_color_ = misc_utils_ns::idToColor(new_room_id);
            cv::Vec3b new_color = cv::Vec3b(new_color_[0], new_color_[1], new_color_[2]);
            room_mask_vis_cropped.setTo(new_color, mask_new);
            room_mask_new.setTo(0, mask_new);

            std::vector<cv::Point> non_zero_points_new;
            cv::findNonZero(mask_new, non_zero_points_new);
            for (auto &pt : non_zero_points_new)
            {
                pt.x += bbox_[0][1];
                pt.y += bbox_[0][0];
            }

            representation_ns::RoomNodeRep new_room_node(new_room_id, non_zero_points_new);
            room_nodes_map_[new_room_id] = new_room_node;
        }
    }
    
    // Remove all inactive room nodes
    for (auto it = room_nodes_map_.begin(); it != room_nodes_map_.end();)
    {
        assert(it->second.id_ == it->first);
        if (!it->second.alive) {
            it = room_nodes_map_.erase(it);
        } else {
            ++it;
        }
    }

    // Verify all room IDs processed
    int non_zero_count = cv::countNonZero(room_mask_new);
    room_mask_new *= 255;
    saveImageToFile(room_mask_new, "room_mask_new.png");
    assert(non_zero_count == 0);

    // Renumber rooms and update visualization
    cv::Mat room_mask_vis_cropped_new = room_mask_vis_.rowRange(bbox_[0][0], bbox_[1][0] + 1)
                                                      .colRange(bbox_[0][1], bbox_[1][1] + 1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr room_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    int counter = 1;
    for (auto &id_room_node_pair : room_nodes_map_)
    {
        representation_ns::RoomNodeRep &room_node = id_room_node_pair.second;
        cv::Mat mask_new = (room_mask_cropped == room_node.GetId());
        room_node.SetRoomMask(mask_new);
        room_node.show_id_ = counter;
        counter++;

        geometry_msgs::msg::PolygonStamped new_room_polygon = computePolygonFromMaskCropped(mask_new);
        room_node.UpdatePolygon(new_room_polygon);
        room_node.area_ = cv::countNonZero(mask_new) * room_resolution_ * room_resolution_;

        Eigen::Vector3f room_centroid(0.0, 0.0, 0.0);
        if (!room_node.points_.empty())
        {
            for (const auto &pt : room_node.points_)
            {
                Eigen::Vector3i pt_voxel(pt.y, pt.x, 0);
                Eigen::Vector3f pt_position = misc_utils_ns::voxel_to_point(pt_voxel, shift_, room_resolution_);
                pcl::PointXYZRGB point;
                point.x = pt_position.x();
                point.y = pt_position.y();
                point.z = robot_position_.z;
                Eigen::Vector3d color_ = misc_utils_ns::idToColor(room_node.GetId());
                point.b = static_cast<uint8_t>(color_[0]);
                point.g = static_cast<uint8_t>(color_[1]);
                point.r = static_cast<uint8_t>(color_[2]);
                room_cloud->points.push_back(point);

                room_centroid.x() += pt_position.x();
                room_centroid.y() += pt_position.y();
            }
        }
        room_centroid.x() /= room_node.points_.size();
        room_centroid.y() /= room_node.points_.size();
        room_centroid.z() = robot_position_.z;
        room_node.centroid_ = room_centroid;
        room_node.neighbors_.clear();
    }
    
    sensor_msgs::msg::PointCloud2 room_cloud_msg;
    pcl::toROSMsg(*room_cloud, room_cloud_msg);
    room_cloud_msg.header.stamp = this->now();
    room_cloud_msg.header.frame_id = "map";
    pub_room_map_cloud_->publish(room_cloud_msg);
}

// TODO: Copy implementation from original file lines 1600-2100
void RoomSegmentationNode::roomSegmentation()
{
    auto t_all_0 = std::chrono::high_resolution_clock::now();
    auto t_0 = std::chrono::high_resolution_clock::now();
    
    // Extract the room boundary from the navigable map
    cv::Mat hist_full = navigable_map_.clone();
    cv::normalize(hist_full, hist_full, 0, 255, cv::NORM_MINMAX);
    cv::Mat outside_boundary = cv::Mat::zeros(hist_full.size(), CV_8U);
    cv::threshold(hist_full, outside_boundary, 0, 255, cv::THRESH_BINARY);
    saveImageToFile(outside_boundary, "full_map_1.png");

    if (outside_boundary.type() != CV_8U)
        outside_boundary.convertTo(outside_boundary, CV_8U);

    auto t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Extract Boundary: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;
    
    // --------------------------- Pre-Processing: Exacting the wall ---------------------------
    t_0 = std::chrono::high_resolution_clock::now();
    cv::Mat wall_from_plane = getWall(in_range_cloud_);
    t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Get Wall from Plane: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;

    // --------------------------- Extract walls from histogram ---------------------------
    t_0 = std::chrono::high_resolution_clock::now();
    cv::Mat wall_from_hist = wall_hist_.clone();
    cv::normalize(wall_from_hist, wall_from_hist, 0, 1, cv::NORM_MINMAX);
    wall_from_hist.convertTo(wall_from_hist, CV_8U);
    saveImageToFile(wall_from_hist, "walls_skeleton_hist_1_raw.png");
    double max_val_wall_from_hist = 0;
    cv::minMaxLoc(wall_from_hist, nullptr, &max_val_wall_from_hist);
    double threshold_wall_from_hist = max_val_wall_from_hist * 0.5;
    cv::threshold(wall_from_hist, wall_from_hist, threshold_wall_from_hist, 255, cv::THRESH_BINARY);
    cv::normalize(wall_from_plane, wall_from_plane, 0, 255, cv::NORM_MINMAX);
    wall_from_plane.convertTo(wall_from_plane, CV_8U);
    cv::threshold(wall_from_plane, wall_from_plane, 0, 255, cv::THRESH_BINARY);
    // save 2 histograms speparately
    saveImageToFile(wall_from_plane, "wall_from_plane.png");
    saveImageToFile(wall_from_hist, "wall_from_hist.png");
    saveImageToFile(state_map_, "state_map.png");

    // Combine the two histograms
    cv::Mat walls_skeleton_hist_connected = wall_from_hist.clone();
    wall_from_plane = wall_from_plane | wall_from_hist;

    wall_from_plane.setTo(0, state_map_);

    t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Extract Wall from Histogram: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;

    // --------------------------- Contour Processing ---------------------------
    t_0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(outside_boundary.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    double min_hole_area = 400.0;
    outside_boundary = cv::Mat::zeros(outside_boundary.size(), CV_8U);

    for (int i = 0; i < contours.size(); ++i)
    {
        // 外轮廓
        if (hierarchy[i][3] == -1)
        {
            cv::drawContours(outside_boundary, contours, i, cv::Scalar(255), cv::FILLED);

            int child = hierarchy[i][2];
            while (child != -1)
            {
                double area = cv::contourArea(contours[child]);
                if (area >= min_hole_area)
                {
                    // 仅当内轮廓面积足够大时才绘制
                    cv::drawContours(outside_boundary, contours, child, cv::Scalar(0), cv::FILLED);
                }
                child = hierarchy[child][0]; // 下一个兄弟
            }
        }
    }

    saveImageToFile(wall_from_plane, "wall_all.png");

    cv::Mat outside_boundary_connected = outside_boundary.clone();

    // Remove wall positions from boundary
    outside_boundary.setTo(0, wall_from_plane);
    outside_boundary_connected.setTo(0, walls_skeleton_hist_connected);

    // Filter connected components by area
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(outside_boundary, labels, stats, centroids, 8);

    cv::Mat area_mask = cv::Mat::zeros(outside_boundary.size(), CV_8U);
    for (int i = 1; i < num_labels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > 100)
        {
            cv::Mat label_mask = (labels == i);
            area_mask.setTo(255, label_mask);
        }
    }
    outside_boundary = area_mask.clone();

    // filter small connected components in outside_boundary_connected
    num_labels = cv::connectedComponentsWithStats(outside_boundary_connected, labels, stats, centroids, 8);
    area_mask = cv::Mat::zeros(outside_boundary_connected.size(), CV_8U);
    for (int i = 1; i < num_labels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > 100)
        {
            cv::Mat label_mask = (labels == i);
            area_mask.setTo(255, label_mask);
        }
    }
    outside_boundary_connected = area_mask.clone();

    cv::Mat full_map, full_map_connected;
    cv::bitwise_not(outside_boundary, full_map);
    cv::bitwise_not(outside_boundary_connected, full_map_connected);
    cv::Mat boundary_mask = full_map.clone();

    saveImageToFile(full_map, "full_map.png");
    saveImageToFile(full_map_connected, "full_map_connected.png");

    t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Contour Processing: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;

    // --------------------------- Boundary Dilation ---------------------------
    t_0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Point2d> found_region_centers;
    std::vector<cv::Mat> found_region_masks;

    cv::Mat kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat new_boundary;
    cv::dilate(boundary_mask, new_boundary, kernel_, cv::Point(-1, -1), dilation_iteration_);

    cv::Mat new_boundary_inv;
    cv::bitwise_not(new_boundary, new_boundary_inv);

    saveImageToFile(new_boundary, "new_boundary.png");

    cv::Mat new_labels, new_stats, centers;
    int num_rooms = cv::connectedComponentsWithStats(new_boundary_inv, new_labels, new_stats, centers, 8);

    for (int label = 1; label < num_rooms; ++label)
    {
        cv::Mat mask = (new_labels == label);
        int area = new_stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > min_room_size_) // TODO
        {
            double cy = centers.at<double>(label, 0);
            double cx = centers.at<double>(label, 1);
            // add the bbox back to cx and cy
            cx = cx + bbox_[0][0];
            cy = cy + bbox_[0][1];
            found_region_centers.emplace_back(cx, cy);
            // 用mask直接重写new_boundary
            new_boundary.setTo(255, mask);

            mask.convertTo(mask, CV_8U, 255);
            cv::Mat mask_dilated;
            // cv::dilate(mask, mask_dilated, kernel_, cv::Point(-1, -1), dilation_iteration / 2);
            found_region_masks.push_back(mask);
        }
    }

    // --------------------------- Watershed Segmentation ---------------------------
    cv::Mat markers = cv::Mat::zeros(boundary_mask.size(), CV_32S);
    // Set background markers
    cv::Mat bg_mask = (full_map_connected != 0);
    markers.setTo(found_region_masks.size() + 1, bg_mask);

    // Set foreground markers
    for (size_t i = 0; i < found_region_masks.size(); ++i)
    {
        cv::Mat mask = (found_region_masks[i] == 255);
        markers.setTo(static_cast<int>(i + 1), mask);
    }

    cv::Mat full_map_color;
    cv::cvtColor(full_map_connected, full_map_color, cv::COLOR_GRAY2BGR);
    saveImageToFile(full_map_color, "full_map_color.png");
    cv::watershed(full_map_color, markers);

    // --------------------------- Merge Rooms ---------------------------
    auto t0_merge_room = std::chrono::high_resolution_clock::now();
    // Convert cropped markers to original image size
    cv::Mat door_mask = (markers == -1);
    
    // Mark all doors on image boundaries as 0
    for (int c = 0; c < door_mask.cols; ++c)
    {
        if (door_mask.at<uchar>(0, c) != 0)
            door_mask.at<uchar>(0, c) = 0;
        if (door_mask.at<uchar>(door_mask.rows - 1, c) != 0)
            door_mask.at<uchar>(door_mask.rows - 1, c) = 0;
    }

    // Left and right columns
    for (int r = 0; r < door_mask.rows; ++r)
    {
        if (door_mask.at<uchar>(r, 0) != 0)
            door_mask.at<uchar>(r, 0) = 0;
        if (door_mask.at<uchar>(r, door_mask.cols - 1) != 0)
            door_mask.at<uchar>(r, door_mask.cols - 1) = 0;
    }

    // Set all markers with value found_region_masks.size() + 1 and -1 to 0
    markers.setTo(0, markers == found_region_masks.size() + 1);
    markers.setTo(0, markers == -1);
    
    cv::Mat room_mask_cropped = room_mask_.rowRange(bbox_[0][0], bbox_[1][0] + 1)
                                          .colRange(bbox_[0][1], bbox_[1][1] + 1);
    cv::Mat room_mask_vis_cropped = room_mask_vis_.rowRange(bbox_[0][0], bbox_[1][0] + 1)
                                                   .colRange(bbox_[0][1], bbox_[1][1] + 1);
    int room_number = found_region_masks.size();
    updateRooms(room_mask_cropped, markers, room_mask_vis_cropped, room_number);

    auto t1_merge_room = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Merge Rooms: " << std::chrono::duration<float>(t1_merge_room - t0_merge_room).count() << " s" << std::endl;

    t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Dilation: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;
    
    // --------------------------- Door Detection and Room Connectivity ---------------------------
    t_0 = std::chrono::high_resolution_clock::now();

    Eigen::Vector3i robot_idx = misc_utils_ns::point_to_voxel_cropped(
        Eigen::Vector3f(robot_position_.x, robot_position_.y, robot_position_.z),
        shift_,
        room_resolution_inv_,
        bbox_);
    int current_room_label = room_mask_cropped.at<int>(robot_idx[0], robot_idx[1]);
    
    if ((current_room_label > 0 && current_room_label <= static_cast<int>(found_region_masks.size())))
    {
        cv::Mat current_room_mask = (room_mask_cropped == current_room_label);
        geometry_msgs::msg::PolygonStamped boundary_polygon = computePolygonFromMaskCropped(current_room_mask);
        pub_room_boundary_->publish(boundary_polygon);
    }

    door_cloud_->clear();
    // Assert room_mask_cropped has the same size as door_mask
    assert(room_mask_cropped.size() == door_mask.size());
    
    // Filter door points
    for (int r = 1; r < door_mask.rows - 1; ++r)
    {
        for (int c = 1; c < door_mask.cols - 1; ++c)
        {
            if (door_mask.at<uchar>(r, c) != 0)
            {
                bool not_door = false;
                std::set<int> neighborLabels;
                neighborLabels.clear();
                
                for (int dr = -1; dr <= 1; ++dr)
                {
                    for (int dc = -1; dc <= 1; ++dc)
                    {
                        if (r + dr < 0 || r + dr >= door_mask.rows ||
                            c + dc < 0 || c + dc >= door_mask.cols || (dr == 0 && dc == 0))
                        {
                            continue;
                        }
                        int label = room_mask_cropped.at<int>(r + dr, c + dc);
                        if (label > 0)
                        {
                            neighborLabels.insert(label);
                        }
                    }
                }
                
                if (neighborLabels.size() == 1)
                {
                    not_door = true;
                    door_mask.at<uchar>(r, c) = 0;
                }
                if (neighborLabels.size() > 2)
                {
                    not_door = true;
                    for (int dr = -1; dr <= 1; ++dr)
                    {
                        for (int dc = -1; dc <= 1; ++dc)
                        {
                            if (r + dr < 0 || r + dr >= door_mask.rows ||
                                c + dc < 0 || c + dc >= door_mask.cols)
                            {
                                continue;
                            }
                            door_mask.at<uchar>(r + dr, c + dc) = 0;
                        }
                    }
                }
            }
        }
    }

    // Find the largest key in the room_nodes_map
    int max_room_label = 0;
    for (const auto &id_room_node_pair : room_nodes_map_)
    {
        if (id_room_node_pair.first > max_room_label)
        {
            max_room_label = id_room_node_pair.first;
        }
    }
    assert(max_room_label == room_node_counter_);
    
    // Initialize adjacency matrix
    Eigen::MatrixXi adjacency_matrix = Eigen::MatrixXi::Zero(max_room_label, max_room_label);

    // Find connected components in door_mask
    num_labels = cv::connectedComponentsWithStats(door_mask, labels, stats, centroids, 8);
    
    std::vector<cv::Point> non_zero_points;
    for (int label = 1; label < num_labels; ++label)
    {
        cv::Mat label_mask = (labels == label);
        non_zero_points.clear();
        cv::findNonZero(label_mask, non_zero_points);
        
        std::set<int> neighborLabels;
        neighborLabels.clear();
        
        for (const auto &point : non_zero_points)
        {
            int r = point.y;
            int c = point.x;
            
            for (int dr = -1; dr <= 1; ++dr)
            {
                for (int dc = -1; dc <= 1; ++dc)
                {
                    if (r + dr < 0 || r + dr >= door_mask.rows ||
                        c + dc < 0 || c + dc >= door_mask.cols || (dr == 0 && dc == 0))
                    {
                        continue;
                    }
                    int label = room_mask_cropped.at<int>(r + dr, c + dc);
                    neighborLabels.insert(label);
                }
            }
        }
        
        neighborLabels.erase(0);
        if (neighborLabels.empty())
        {
            continue;
        }
        if (neighborLabels.size() != 2)
        {
            RCLCPP_ERROR(this->get_logger(), "Door mask has more than two labels or less than two labels, skipping this door.");
            continue;
        }

        int room_label_1 = *neighborLabels.begin();
        int room_label_2 = *std::next(neighborLabels.begin());

        if (room_label_1 > room_label_2)
        {
            std::swap(room_label_1, room_label_2);
        }

        // Add neighbors to room nodes
        room_nodes_map_[room_label_1].neighbors_.insert(room_label_2);
        room_nodes_map_[room_label_2].neighbors_.insert(room_label_1);

        int door_id = adjacency_matrix(room_label_1 - 1, room_label_2 - 1);
        adjacency_matrix(room_label_1 - 1, room_label_2 - 1) += 1;
        adjacency_matrix(room_label_2 - 1, room_label_1 - 1) += 1;

        for (const auto &point : non_zero_points)
        {
            pcl::PointXYZRGBL door_point;
            Eigen::Vector3i door_idx(point.y, point.x, 0);
            Eigen::Vector3f door_position = misc_utils_ns::voxel_to_point_cropped(door_idx, shift_, room_resolution_, bbox_);
            door_point.x = door_position.x();
            door_point.y = door_position.y();
            door_point.z = robot_position_.z;
            door_point.r = room_label_1;
            door_point.g = room_label_2;
            door_point.b = 0;
            door_point.label = door_id;
            door_cloud_->points.push_back(door_point);

            room_mask_vis_cropped.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 0, 255);
        }
    }
    
    // Update room connectivity
    if (current_room_label > 0)
    {
        for (auto &id_room_node_pair : room_nodes_map_)
        {
            int room_id = id_room_node_pair.first;
            auto &room_node = id_room_node_pair.second;
            bool is_connected = isRoomConnected(room_id - 1, current_room_label - 1, adjacency_matrix);
            room_node.is_connected_ = is_connected;
        }
    }

    cv::Mat output_image = room_mask_vis_cropped.clone();

    // Draw robot position
    cv::circle(
        output_image,
        cv::Point(robot_idx[1], robot_idx[0]),
        3,
        cv::Scalar(0, 0, 0),
        -1);

    // Publish room mask
    sensor_msgs::msg::Image markers_msg;
    markers_msg.header.stamp = this->get_clock()->now();
    markers_msg.header.frame_id = "map";
    markers_msg.height = room_mask_.rows;
    markers_msg.width = room_mask_.cols;
    markers_msg.encoding = "32SC1";
    markers_msg.is_bigendian = false;
    markers_msg.step = room_mask_.cols * sizeof(int);
    markers_msg.data.resize(room_mask_.total() * room_mask_.elemSize());
    std::memcpy(markers_msg.data.data(), room_mask_.data, markers_msg.data.size());
    pub_room_mask_->publish(markers_msg);

    saveImageToFile(output_image, "room_segmentation_visualization.png");

    // Flip and transpose for visualization
    cv::Mat flipped_image;
    cv::transpose(output_image, flipped_image);
    cv::flip(flipped_image, flipped_image, 0);
    output_image = flipped_image;

    // Convert to ROS image message
    sensor_msgs::msg::Image output_image_msg;
    output_image_msg.header.stamp = this->get_clock()->now();
    output_image_msg.header.frame_id = "map";
    output_image_msg.height = output_image.rows;
    output_image_msg.width = output_image.cols;
    output_image_msg.encoding = "bgr8";
    output_image_msg.is_bigendian = false;
    output_image_msg.step = output_image.cols * output_image.elemSize();
    output_image_msg.data.resize(output_image.total() * output_image.elemSize());
    std::memcpy(output_image_msg.data.data(), output_image.data, output_image_msg.data.size());
    pub_room_mask_vis_->publish(output_image_msg);

    t_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Post Processing: " << std::chrono::duration<float>(t_1 - t_0).count() << " s" << std::endl;

    auto t_all_1 = std::chrono::high_resolution_clock::now();
    // std::cout << "[Time] Room segmentation: " << std::chrono::duration<float>(t_all_1 - t_all_0).count() << " s" << std::endl;
}


// ==================== Publishing Functions ====================
void RoomSegmentationNode::publishRoomNodes() {
    tare_planner::msg::RoomNodeList room_node_list_msg;
    room_node_list_msg.header.stamp = this->now();
    room_node_list_msg.header.frame_id = "map";
    room_node_list_msg.nodes.reserve(room_nodes_map_.size());
    
    for (const auto &id_room_node_pair : room_nodes_map_) {
        const representation_ns::RoomNodeRep &room_node = id_room_node_pair.second;
        tare_planner::msg::RoomNode room_node_msg;
        room_node_msg.id = room_node.GetId();
        room_node_msg.show_id = room_node.show_id_;
        room_node_msg.polygon = room_node.GetPolygon();
        room_node_msg.centroid.x = room_node.centroid_.x();
        room_node_msg.centroid.y = room_node.centroid_.y();
        room_node_msg.centroid.z = room_node.centroid_.z();
        
        for (const auto &neighbor : room_node.neighbors_) {
            room_node_msg.neighbors.push_back(neighbor);
        }
        room_node_msg.is_connected = room_node.is_connected_;
        room_node_msg.area = room_node.area_;
        room_node_msg.room_mask = *cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", room_node.room_mask_).toImageMsg();
        
        room_node_list_msg.nodes.push_back(room_node_msg);
    }
    pub_room_node_list_->publish(room_node_list_msg);
}

void RoomSegmentationNode::publishRoomPolygon() {
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.ns = "polygon_group";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);
    
    for (auto &id_room_node_pair : room_nodes_map_) {
        const auto &room_node = id_room_node_pair.second;
        const auto &poly = room_node.GetPolygon();

        visualization_msgs::msg::Marker marker;
        marker.header = poly.header;
        marker.ns = "polygon_group";
        marker.id = static_cast<int>(room_node.GetId());
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        Eigen::Vector3d color_ = misc_utils_ns::idToColor(room_node.GetId());
        marker.scale.x = 0.13;
        marker.color.b = color_[0] / 255.0;
        marker.color.g = color_[1] / 255.0;
        marker.color.r = color_[2] / 255.0;
        marker.color.a = 1.0;

        for (const auto &pt : poly.polygon.points) {
            geometry_msgs::msg::Point p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = robot_position_.z;
            marker.points.push_back(p);
        }

        if (!marker.points.empty()) {
            marker.points.push_back(marker.points.front());
        }
        marker_array.markers.push_back(marker);
    }
    pub_polygon_->publish(marker_array);
}

void RoomSegmentationNode::publishDoorCloud() {
    sensor_msgs::msg::PointCloud2 door_msg;
    pcl::toROSMsg(*door_cloud_, door_msg);
    door_msg.header.stamp = this->now();
    door_msg.header.frame_id = "map";
    pub_door_cloud_->publish(door_msg);
}

} // namespace room_segmentation

// ==================== Main Function ====================
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<room_segmentation::RoomSegmentationNode>();
    
    RCLCPP_INFO(node->get_logger(), "Room Segmentation Node is running...");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}