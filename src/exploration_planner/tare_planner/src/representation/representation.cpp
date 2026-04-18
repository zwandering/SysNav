/**
 * @file representation.cpp
 * @author Haokun Zhu (haokunz@andrew.cmu.edu)
 * @brief Class that implements the VLM representation
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "representation/representation.h"
#include <utils/misc_utils.h>

using namespace std::chrono_literals;

namespace representation_ns {
    ViewPointRep::ViewPointRep(int id, double x, double y, double z, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp)
        : position_(), room_id_(-1)
    {
        id_ = id; // Set the unique ID for the viewpoint
        position_.x = x;
        position_.y = y;
        position_.z = z;
        cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>(*cloud);
        covered_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*covered_cloud);
        timestamp_ = timestamp;

        room_id_ = -1; // Initialize room ID to -1 (not assigned)
    }

    ViewPointRep::ViewPointRep(int id, const geometry_msgs::msg::Point &position, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp)
        : ViewPointRep(id, position.x, position.y, position.z, cloud, covered_cloud, timestamp)
    {
    }

    RoomNodeRep::RoomNodeRep(const int id, const std::vector<cv::Point> &points)
        : id_(id), show_id_(0), points_(points), alive(true), centroid_(0.0, 0.0, 0.0), area_(0.0), last_area_(0.0), is_visited_(false), is_covered_(false), is_labeled_(false), is_asked_(2), voxel_num_(0), is_connected_(false)
    {
        viewpoint_indices_ = std::set<int>(); // Initialize the viewpoint indices set
        object_indices_ = std::set<int>();    // Initialize the object indices set
        anchor_point_ = geometry_msgs::msg::Point();
    }

    void RoomNodeRep::UpdateRoomNode(const std::vector<cv::Point> &points, const geometry_msgs::msg::PolygonStamped &polygon)
    {
        points_ = points;
        polygon_ = polygon;
        alive = true; // Mark the room as alive
    }
    
    void RoomNodeRep::UpdateRoomNode(const std::vector<cv::Point> &points)
    {
        points_ = points;
        alive = true; // Mark the room as alive
    }

    void RoomNodeRep::UpdateCentroid(Eigen::Vector3f &centroid)
    {
        centroid_ = centroid;
    }

    void RoomNodeRep::UpdatePolygon(const geometry_msgs::msg::PolygonStamped &polygon)
    {
        polygon_ = polygon;
        alive = true; // Mark the room as alive
    }

    bool RoomNodeRep::InRoom(const geometry_msgs::msg::Point &point)
    {
        return misc_utils_ns::PointInPolygon(point, polygon_.polygon);
    }

    void RoomNodeRep::RemovePoint(const cv::Point &point)
    {
        auto it = std::remove(points_.begin(), points_.end(), point);
        if (it != points_.end())
        {
            points_.erase(it, points_.end());
        }
    }

    void RoomNodeRep::UpdateRoomNode(const tare_planner::msg::RoomNode msg)
    {
        id_ = msg.id;
        show_id_ = msg.show_id; // Update the show ID from the message
        polygon_ = msg.polygon;
        centroid_ = Eigen::Vector3f(msg.centroid.x, msg.centroid.y, msg.centroid.z);
        neighbors_.clear();
        for (const auto &neighbor : msg.neighbors)
        {
            neighbors_.insert(neighbor);
        }
        alive = true; // Mark the room as alive
        is_connected_ = msg.is_connected;
        area_ = msg.area;
        // update room_mask_
        room_mask_ = cv_bridge::toCvCopy(msg.room_mask, "mono8")->image;
    }

    ObjectNodeRep::ObjectNodeRep(const tare_planner::msg::ObjectNode::ConstSharedPtr msg)
    : room_id_(-1)
    {
        object_id_ = msg->object_id;
        label_ = msg->label;
        position_ = msg->position;
        cloud_ = msg->cloud;
        status_ = msg->status; // Initialize the status from the message
        timestamp_ = msg->header.stamp; // Initialize the timestamp from the message
        visible_viewpoint_indices_ = std::set<int>(); // Initialize the visible viewpoint indices set
        room_id_ = -1;                                // Initialize room ID to -1 (not assigned)
    }

    void ObjectNodeRep::UpdateObjectNode(const tare_planner::msg::ObjectNode::ConstSharedPtr msg)
    {
        object_id_ = msg->object_id; 
        label_ = msg->label;
        position_ = msg->position;
        cloud_ = msg->cloud;
        status_ = msg->status;          // Initialize the status from the message
        timestamp_ = msg->header.stamp; // Initialize the timestamp from the message
        img_path_ = msg->img_path;
        is_asked_vlm_ = msg->is_asked_vlm;
        bbox3d_ = msg->bbox3d;
        if (!is_asked_vlm_ && !is_considered_strong_ && is_considered_)
        {
            is_considered_ = false;
        }
    }

    Representation::Representation(rclcpp::Node::SharedPtr nh, std::string world_frame_id)
    : nh_(nh)
    {
        viewpoint_rep_vis_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        viewpoint_reps_ = std::vector<ViewPointRep>();
        room_nodes_map_ = std::map<int, RoomNodeRep>(); // Use an unordered map for room nodes
        object_node_rep_map_ = std::unordered_map<int, ObjectNodeRep>();
        latest_object_node_indices_ = std::set<int>();
        covered_points_all_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    }

    void Representation::AddViewPointRepNode(const geometry_msgs::msg::Point &position, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp)
    {   
        int id = viewpoint_reps_.size(); // Use the current size as the ID
        ViewPointRep viewpoint_rep(id, position, cloud, covered_cloud, timestamp);
        viewpoint_reps_.push_back(viewpoint_rep);
        // translate the viewpoint_rep to the visualization cloud (from geometry_msgs::msg::Point to pcl::PointXYZ)
        pcl::PointXYZ point;
        point.x = position.x;
        point.y = position.y;
        point.z = position.z;
        viewpoint_rep_vis_cloud_->points.push_back(point);
        *covered_points_all_ += *covered_cloud;
    }

    int Representation::AddViewPointRep(const geometry_msgs::msg::Point &position, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &covered_cloud, const rclcpp::Time &timestamp)
    {
        if (viewpoint_reps_.size() == 0){
            AddViewPointRepNode(position, cloud, covered_cloud, timestamp);
            return 0; // Successfully added the first viewpoint
        } 
        else {
            // check if the current position is within a certain distance from the existing viewpoints
            double distance_thres = 2.0; // Define a minimum distance threshold
            double min_dist = DBL_MAX;
            int min_dist_ind = -1;
            for (int i = 0; i < viewpoint_reps_.size(); i++)
            {
                double dist = misc_utils_ns::PointXYZDist<geometry_msgs::msg::Point, geometry_msgs::msg::Point>(viewpoint_reps_[i].GetPosition(), position);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_dist_ind = i;
                }
            }
            if (min_dist > distance_thres){
                AddViewPointRepNode(position, cloud, covered_cloud, timestamp);
                return viewpoint_reps_.size() - 1; // Successfully added a new viewpoint
            } else {
                std::cout << "Viewpoint is too close to an existing viewpoint, not adding it." << std::endl;
                return min_dist_ind; // Viewpoint not added due to proximity, return the index of the closest viewpoint
            }
        }
    }

    void Representation::UpdateViewpointRoomIdsFromMask(const cv::Mat& room_mask, const Eigen::Vector3f& shift, float room_resolution) 
    {
        for (auto& viewpoint : viewpoint_reps_) {
            Eigen::Vector3f pos_float(viewpoint.GetPosition().x, viewpoint.GetPosition().y, viewpoint.GetPosition().z);
            Eigen::Vector3i voxel = misc_utils_ns::point_to_voxel(pos_float, shift, 1.0 / room_resolution);

            int old_room_id = viewpoint.GetRoomId();
            
            // Check bounds
            if (voxel.x() >= 0 && voxel.x() < room_mask.rows && 
                voxel.y() >= 0 && voxel.y() < room_mask.cols) {
                int new_room_id = room_mask.at<int>(voxel.x(), voxel.y());
                SetViewpointRoomRelation(viewpoint.GetId(), new_room_id); 
            } else {
                RCLCPP_WARN(rclcpp::get_logger("representation"), 
                            "Viewpoint at (%.2f, %.2f, %.2f) is out of room mask bounds! Voxel: (%d, %d), Mask size: (%d, %d)", 
                            viewpoint.GetPosition().x, viewpoint.GetPosition().y, viewpoint.GetPosition().z,
                            voxel.x(), voxel.y(), room_mask.rows, room_mask.cols);
                // Set to unknown room instead of leaving unchanged
                viewpoint.SetRoomId(-1);
            }
        }
    }

    std::string Representation::ToJSON() const
    {
        // Convert the representation to a JSON string

    }

    void Representation::UpdateObjectNode(const tare_planner::msg::ObjectNode::ConstSharedPtr msg)
    {
        if (msg->object_id.size() == 0) {
            RCLCPP_WARN(nh_->get_logger(), "Received ObjectNode message with empty object_id");
            return;
        }
        int object_id = msg->object_id[0];
        representation_ns::ObjectNodeRep &object_node = object_node_rep_map_[object_id];
        object_node.UpdateObjectNode(msg);
        if (msg->viewpoint_id >= 0)
        {
            object_node.AddVisibleViewpoint(msg->viewpoint_id);
            viewpoint_reps_[msg->viewpoint_id].AddDirectObjectIndex(object_id);
            object_node.SetIsConsidered(false);
        }
    }

    // ==================== Const Getter Implementations ====================
    const ViewPointRep& Representation::GetViewPointRepNode(int index) const
    {
        if (index < 0 || index >= viewpoint_reps_.size()) {
            throw std::out_of_range("Viewpoint index out of range");
        }
        return viewpoint_reps_[index];
    }

    ViewPointRep& Representation::GetViewPointRepNode(int index)
    {
        if (index < 0 || index >= viewpoint_reps_.size()) {
            throw std::out_of_range("Viewpoint index out of range");
        }
        return viewpoint_reps_[index];
    }

    geometry_msgs::msg::Point Representation::GetViewPointRepNodePos(int index) const
    {
        if (index < 0 || index >= viewpoint_reps_.size()) {
            throw std::out_of_range("Viewpoint index out of range");
        }
        return viewpoint_reps_[index].position_;
    }

    const ObjectNodeRep& Representation::GetObjectNodeRep(int object_id) const
    {
        auto it = object_node_rep_map_.find(object_id);
        if (it == object_node_rep_map_.end()) {
            throw std::out_of_range("Object ID not found: " + std::to_string(object_id));
        }
        return it->second;
    }

    ObjectNodeRep& Representation::GetObjectNodeRep(int object_id)
    {
        auto it = object_node_rep_map_.find(object_id);
        if (it == object_node_rep_map_.end()) {
            throw std::out_of_range("Object ID not found: " + std::to_string(object_id));
        }
        return it->second;
    }

    RoomNodeRep& Representation::AddRoomNode(const tare_planner::msg::RoomNode& msg)
    {
        RoomNodeRep new_room;
        new_room.id_ = msg.id;
        new_room.UpdateRoomNode(msg);
        room_nodes_map_[msg.id] = new_room;
        return room_nodes_map_[msg.id];
    }

    const RoomNodeRep& Representation::GetRoomNode(int room_id) const
    {
        auto it = room_nodes_map_.find(room_id);
        if (it == room_nodes_map_.end()) {
            throw std::out_of_range("Room ID not found: " + std::to_string(room_id));
        }
        return it->second;
    }

    RoomNodeRep& Representation::GetRoomNode(int room_id)
    {
        auto it = room_nodes_map_.find(room_id);
        if (it == room_nodes_map_.end()) {
            throw std::out_of_range("Room ID not found: " + std::to_string(room_id));
        }
        return it->second;
    }

    size_t Representation::GetViewPointCount() const
    {
        return viewpoint_reps_.size();
    }

    const std::vector<ViewPointRep>& Representation::GetViewPointReps() const
    {
        return viewpoint_reps_;
    }

    std::vector<ViewPointRep>& Representation::GetViewPointRepsMutable()
    {
        return viewpoint_reps_;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Representation::GetViewPointRepCloud() const
    {
        return viewpoint_rep_vis_cloud_;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr Representation::GetCoveredPointsAllCloud() const
    {
        return covered_points_all_;
    }

    bool Representation::HasObjectNode(int object_id) const
    {
        return object_node_rep_map_.find(object_id) != object_node_rep_map_.end();
    }

    size_t Representation::GetObjectNodeCount() const
    {
        return object_node_rep_map_.size();
    }

    const std::unordered_map<int, ObjectNodeRep>& Representation::GetObjectNodeRepMap() const
    {
        return object_node_rep_map_;
    }

    std::unordered_map<int, ObjectNodeRep>& Representation::GetObjectNodeRepMapMutable()
    {
        return object_node_rep_map_;
    }

    const std::set<int>& Representation::GetLatestObjectNodeIndices() const
    {
        return latest_object_node_indices_;
    }

    std::set<int>& Representation::GetLatestObjectNodeIndicesMutable()
    {
        return latest_object_node_indices_;
    }

    bool Representation::HasRoomNode(int room_id) const
    {
        return room_nodes_map_.find(room_id) != room_nodes_map_.end();
    }

    size_t Representation::GetRoomNodeCount() const
    {
        return room_nodes_map_.size();
    }

    const std::map<int, RoomNodeRep>& Representation::GetRoomNodesMap() const
    {
        return room_nodes_map_;
    }

    std::map<int, RoomNodeRep>& Representation::GetRoomNodesMapMutable()
    {
        return room_nodes_map_;
    }

    void Representation::SetObjectRoomRelation(int object_id, int new_room_id)
    {
        if (object_node_rep_map_.find(object_id) == object_node_rep_map_.end()) {
            RCLCPP_ERROR(nh_->get_logger(), "Object ID %d not found in object_node_rep_map_", object_id);
            return;
        }
        if (room_nodes_map_.find(new_room_id) == room_nodes_map_.end()) {
            RCLCPP_ERROR(nh_->get_logger(), "Room ID %d not found in room_nodes_map_", new_room_id);
            return;
        }
        int old_room_id = object_node_rep_map_[object_id].GetRoomId();
        if (room_nodes_map_.find(old_room_id) != room_nodes_map_.end() && old_room_id != new_room_id)
        {
            // Remove the object index from the old room
            room_nodes_map_[old_room_id].DeleteObjectIndex(object_id);
        }
        object_node_rep_map_[object_id].SetRoomId(new_room_id);
        room_nodes_map_[new_room_id].AddObjectIndex(object_id);
    }

    void Representation::SetViewpointRoomRelation(int viewpoint_id, int new_room_id)
    {
        if (viewpoint_id < 0 || viewpoint_id >= viewpoint_reps_.size()) {
            RCLCPP_ERROR(nh_->get_logger(), "Viewpoint ID %d is out of bounds for viewpoint_reps_", viewpoint_id);
            return;
        }
        if (room_nodes_map_.find(new_room_id) == room_nodes_map_.end()) {
            RCLCPP_ERROR(nh_->get_logger(), "Room ID %d is out of bounds for room_nodes_", new_room_id);
            return;
        }
        int old_room_id = viewpoint_reps_[viewpoint_id].GetRoomId();
        if (room_nodes_map_.find(old_room_id) != room_nodes_map_.end() && old_room_id != new_room_id)
        {
            // Remove the viewpoint index from the old room
            room_nodes_map_[old_room_id].DeleteViewpointId(viewpoint_id);
        }
        viewpoint_reps_[viewpoint_id].SetRoomId(new_room_id);
        room_nodes_map_[new_room_id].AddViewpointId(viewpoint_id);
    }

} // namespace representation_ns
