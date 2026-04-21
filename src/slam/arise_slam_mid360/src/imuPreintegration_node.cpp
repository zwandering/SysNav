//
// Created by shibo zhao on 2020-09-27.
//
#include "rclcpp/rclcpp.hpp"
#include "arise_slam_mid360/ImuPreintegration/imuPreintegration.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions options;
    options.arguments({"imu_preintegration_node"});

    std::shared_ptr<arise_slam::imuPreintegration> imuPreintegration =
        std::make_shared<arise_slam::imuPreintegration>(options);
    
    imuPreintegration->imuBuf.allocate(1000);
    imuPreintegration->lidarOdomBuf.allocate(100);
    imuPreintegration->visualOdomBuf.allocate(5000);
    imuPreintegration->initInterface();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(imuPreintegration);
    executor.spin();
    // rclcpp::spin(imuPreintegration->get_node_base_interface());
    rclcpp::shutdown();
    
    return 0;
}
