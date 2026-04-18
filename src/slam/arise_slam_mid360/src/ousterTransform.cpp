// #include "rclcpp/rclcpp.hpp"

// #include <pcl/filters/voxel_grid.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl_conversions/pcl_conversions.h>

// #include <nav_msgs/msg/odometry.hpp>
// #include <sensor_msgs/msg/point_cloud2.hpp>
// #include "arise_slam_mid360/sensor_data/pointcloud/point_os.h"
// #include "arise_slam_mid360/common.h"
// #include <tf2_ros/transform_listener.h>
// #include <tf2_eigen/tf2_eigen.h>

// #include "arise_slam_mid360/utils/Twist.h"
// #include <queue>

// namespace arise_slam {
// class ousterTransform : public rclcpp::Node {
// public:

//     ousterTransform() : Node("ouster_transform_node") {
//         frame_count_ = 0;
//         skip_frame_ = 1;
//         ousterCloud.reset(new pcl::PointCloud<PointType>());
//         ousterCloudOut.reset(new pcl::PointCloud<PointType>());

//         if (!readParameters()) {
//             RCLCPP_ERROR(this->get_logger(), "[arise_slamtry::ousterTransform] Could not read parameters. Exiting...");
//             rclcpp::shutdown();
//         }

//         subLaserCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
//             "/ouster_cloud_2", 2,
//             std::bind(&ousterTransform::laserCloudCB, this,
//                         std::placeholders::_1));
        
//         pubLaserCloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
//             "/ouster_cloud_registered", 1);
        
//         pubOdom = this->create_publisher<nav_msgs::msg::Odometry>(
//             "/ouster_aft_mapped", 1);
        
//         process();
//     }

//     // void
//     // setUpROS(ros::NodeHandle *pub_node, ros::NodeHandle *private_node);

//     bool readParameters() {
//         if (!this->get_parameter("skip_point_count", skip_count_))
//             return false;

//         return true;
//     }

//     void laserCloudCB(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg) {
//         frame_count_ = (frame_count_ + 1) % skip_frame_;
//         if (frame_count_ != 0)
//             return;

//         // clear queue if too big.
//         if(ouster_buf_.size()>2){
//             std::queue<sensor_msgs::msg::PointCloud2::SharedPtr>().swap(ouster_buf_);
//         }

//         ouster_buf_.push(laserCloudMsg);
//     }

//     void process() {
//         rclcpp::Rate r(100.0);

//         if (ouster_buf_.size() > 0)
//         {

//             double ouster_timestamp = ouster_buf_.front()->header.stamp.sec + ouster_buf_.front()->header.stamp.nanosec*1e-9;
//             rclcpp::Time ouster_time = ouster_buf_.front()->header.stamp;

//             ousterCloud->clear();
//             pcl::fromROSMsg(*ouster_buf_.front(), *ousterCloud);
//             ouster_buf_.pop();

//             ousterCloudOut->clear();

//             auto &laserCloudIn = *ousterCloud;


//             RCLCPP_WARN(this->get_logger(), "ouster_timestamp: %f", ouster_timestamp);

//             tf2::Stamped<tf2::Transform> transform;
//             try
//             {
//                 listener_->waitForTransform("sensor_init_rot", "aft_mapped",
//                             rclcpp::Time(ouster_time), rclcpp::Duration(1.0));
//                 listener_->lookupTransform("sensor_init_rot", "aft_mapped",
//                                             rclcpp::Time(ouster_time), transform);

//                 Eigen::Quaterniond orientation = transform.getRotation();
//                 Eigen::Vector3d translation = transform.getOrigin();
//                 // tf2::vectorTFToEigen(transform.getOrigin(), translation);
//                 // tf2::quaternionTFToEigen(transform.getRotation(), orientation);

//                 Transformd T_map(orientation, translation);

//                 RCLCPP_WARN_STREAM(this->get_logger(), "T_map: " << T_map);
//                 int point_count = 0;
//                 for (auto &point : laserCloudIn)
//                 {
//                     if(point_count==0)
//                     {
//                         Eigen::Vector3d pt{point.x, point.y, point.z};
//                         pt = T_map * pt;

//                         point.x = pt.x();
//                         point.y = pt.y();
//                         point.z = pt.z();

//                         ousterCloudOut->points.push_back(point);

//                     }                        
                    

//                     point_count = (point_count + 1) % skip_count_;
//                 }

//                 RCLCPP_WARN(this->get_logger(), "ouster pointcloud size: %ld", ousterCloudOut->points.size());

//                 // Publish transformed pointcloud
//                 sensor_msgs::msg::PointCloud2 ousterRegistered;
//                 pcl::toROSMsg(*ousterCloudOut, ousterRegistered);
//                 ousterRegistered.header.stamp =
//                     rclcpp::Time(ouster_timestamp);
//                 ousterRegistered.header.frame_id = "sensor_init_rot";
//                 pubLaserCloud->publish(ousterRegistered);

//                 // publishOdom
//                 nav_msgs::msg::Odometry odom;
//                 odom.header.frame_id = "sensor_init_rot";
//                 odom.child_frame_id = "aft_mapped";
//                 odom.header.stamp = rclcpp::Time(ouster_timestamp);

//                 odom.pose.pose.orientation.x = orientation.x();
//                 odom.pose.pose.orientation.y = orientation.y();
//                 odom.pose.pose.orientation.z = orientation.z();
//                 odom.pose.pose.orientation.w = orientation.w();

//                 odom.pose.pose.position.x = translation.x();
//                 odom.pose.pose.position.y = translation.y();
//                 odom.pose.pose.position.z = translation.z();
//                 pubOdom->publish(odom);
//             }
//             catch (tf2::TransformException ex)
//             {
//                 RCLCPP_ERROR(this->get_logger(),"%s", ex.what());
//                 rclcpp::sleep_for(1);
//             }

//         }

//         r.sleep();

//     }

// private:
//     // ROS Interface
//     // Subscribers
//     rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;

//     // Publishers
//     rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;
//     rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom;

//     int frame_count_;
//     std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> ouster_buf_;
//     pcl::PointCloud<PointType>::Ptr ousterCloud;
//     pcl::PointCloud<PointType>::Ptr ousterCloudOut;

//     // params
//     int skip_frame_;

//     std::shared_ptr<tf2_ros::TransformListener> listener_;

//     int skip_count_;

// };

//     // ousterTransform::ousterTransform()
//     // {
//     // frame_count_ = 0;

//     // skip_frame_ = 1;

//     // ousterCloud.reset(new pcl::PointCloud<PointType>());
//     // ousterCloudOut.reset(new pcl::PointCloud<PointType>());
//     // }

// } /* namespace arise_slam */

// int main(int argc, char **argv)
// {
//     // ros::init(argc, argv, "ousterTransform_node");
//     rclcpp::init(argc,argv);
//     auto node =  std::make_shared<arise_slam::ousterTransform>();
//     // ros::NodeHandle node;
//     // ros::NodeHandle privateNode("~");
//     RCLCPP_INFO(node->get_logger(), "\033[1;32m----> LaserMapping Started.\033[0m");

//     // arise_slam::ousterTransform OT;

//     // OT.setUpROS(&node, &privateNode);

//     // std::thread mapping_process{&arise_slam::laserMapping::process, &node};

//   // LM.

//     // OT.process();
//     rclcpp::spin(node);
//     rclcpp::shutdown();

//     // mapping_process.join();

//     return 0;
// }