//
// Created by shibo zhao on 2020-09-27.
//
#include "rclcpp/rclcpp.hpp"
#include "arise_slam_mid360/ScanRegistration/scanRegistration.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions options;
    options.arguments({"scan_registration_node"});
    
    std::shared_ptr<arise_slam::scanRegistration> scanRegistration =
        std::make_shared<arise_slam::scanRegistration>(options);

    scanRegistration->imuBuf.allocate(5000);
    scanRegistration->visualOdomBuf.allocate(200);
    scanRegistration->lidarBuf.allocate(50);
    scanRegistration->initInterface();

    rclcpp::spin(scanRegistration->get_node_base_interface());
    rclcpp::shutdown();

    return 0;
}
