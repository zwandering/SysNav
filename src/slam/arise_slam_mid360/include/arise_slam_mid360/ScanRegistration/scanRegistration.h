//
// Created by shibo zhao on 2020-09-27.
//

#ifndef arise_slam_mid360_SCANREGISTRATION_H
#define arise_slam_mid360_SCANREGISTRATION_H

#include <cmath>
#include <string>
#include <vector>

#include <sophus/so3.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <arise_slam_mid360_msgs/msg/laser_feature.hpp>

#include "arise_slam_mid360/common.h"
#include "arise_slam_mid360/config/parameter.h"
#include "arise_slam_mid360/container/MapRingBuffer.h"
#include "arise_slam_mid360/sensor_data/imu/imu_data.h"
#include "arise_slam_mid360/sensor_data/pointcloud/point_os.h"
#include "arise_slam_mid360/tic_toc.h"
#include "arise_slam_mid360/utils/Twist.h"
#include <mutex>
#include <iomanip>

namespace arise_slam
{

    struct bounds_t
    {
        double blindFront;
        double blindBack;
        double blindRight;
        double blindLeft;
        double blindUp;
        double blindDown;

    };
    struct scan_registration_config{
        bounds_t box_size;
        int skipFrame;
        int N_SCANS;
        std::string IMU_TOPIC;
        std::string LASER_TOPIC;
        std::string ODOM_TOPIC;
        float min_range;
        float max_range;
        int downsampleRate;
        int skip_point;
        SensorType sensor;
    };

    using std::atan2;
    using std::cos;
    using std::sin;

    float cloudCurvature[400000];
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000];
    int cloudLabel[400000];
    // int N_SCANS = 0;

    bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

    class scanRegistration : public rclcpp::Node {

    public:
        // int skipFrame = 1;
        static constexpr double scanPeriod = 0.100859904 - 20.736e-6;
        static constexpr double columnTime = 55.296e-6;
        static constexpr double laserTime = 2.304e-6;

        scanRegistration(const rclcpp::NodeOptions & options);

        void initInterface();

        // void
        // setUpROS(ros::NodeHandle *pub_node, ros::NodeHandle *private_node);

        template <typename PointT>
        void removeClosestFarestPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                            pcl::PointCloud<PointT> &cloud_out, float min_range, float max_range);
        template <typename Meas>
        bool synchronize_measurements(MapRingBuffer<Meas> &measureBuf,
                                        MapRingBuffer<pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr> &lidarBuf, bool remove_laserscan);
        void imuRemovePointDistortion(double lidar_start_time, double lidar_end_time, MapRingBuffer<Imu::Ptr> &imuBuf,
                                    pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg);
        void feature_extraction(double lidar_start_time, pcl::PointCloud<point_os::PointcloudXYZITR> &laserCloudIn);

        void vioRemovePointDistortion(double lidar_start_time, double lidar_end_time, MapRingBuffer<const std::shared_ptr<nav_msgs::msg::Odometry>> &vioBuf,
                                    pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &lidar_msg);
        void undistortionAndscanregistration();

        void imu_Handler(const sensor_msgs::msg::Imu::SharedPtr msg_in);

        void visual_odom_Handler(const nav_msgs::msg::Odometry::SharedPtr visualOdometry);

        void laserCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg);

        void assignTimeforPointCloud(pcl::PointCloud<PointType>::Ptr laserCloudIn_ptr_);

        sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame);

        sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<pcl::PointXYZHSV>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame);

        bool readParameters();

        bool checkExclusionZone(const PointType& p);

        // struct bounds_t
        // {
        //     double blindFront;
        //     double blindBack;
        //     double blindRight;
        //     double blindLeft;
        // };
        // bounds_t box_size; 

        MapRingBuffer<Imu::Ptr> imuBuf;
        MapRingBuffer<pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr> lidarBuf;
        MapRingBuffer<nav_msgs::msg::Odometry::SharedPtr> visualOdomBuf;

    private:
        // ROS Interface
        // Subscribers
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;

        // Publishers
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPointsSharp;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPointsLessSharp;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfPointsFlat;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfPointsLessFlat;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRemovePoints;
        rclcpp::Publisher<arise_slam_mid360_msgs::msg::LaserFeature>::SharedPtr pubLaserFeatureInfo;
        
        int delay_count_;
        std::mutex m_buf;
        int frameCount = 0;

        bool PUB_EACH_LINE = false;
        bool LASER_IMU_SYNC_SCCUESS = false;
        bool LASER_CAMERA_SYNC_SUCCESS = false;

        arise_slam_mid360_msgs::msg::LaserFeature laserFeature;
        std_msgs::msg::Header FeatureHeader;
        Eigen::Quaterniond q_w_original_l;
        Eigen::Vector3d t_w_original_l;
        std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pubEachScan;
        pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr pointCloudwithTime = nullptr;
        pcl::PointCloud<point_os::OusterPointXYZIRT>::Ptr tmpOusterCloudIn = nullptr ;

        scan_registration_config config_;
    };

} // namespace arise_slam

#endif //arise_slam_mid360_SCANREGISTRATION_H
