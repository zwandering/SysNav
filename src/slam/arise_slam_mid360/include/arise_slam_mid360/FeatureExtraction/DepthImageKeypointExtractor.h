//
// Created by shiboz on 2021-09-04.
//

#ifndef arise_slam_mid360_DEPTHIMAGEKEYPOINTEXTRACTOR_H
#define arise_slam_mid360_DEPTHIMAGEKEYPOINTEXTRACTOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

#include "rclcpp/rclcpp.hpp"
#include "arise_slam_mid360/sensor_data/pointcloud/point_os.h"

//points covariance class
class Double2d{
public:
    int id;
    double value;
    Double2d(int id_in, double value_in);
};

enum DepthType:uint8_t{ UP_DEPTH=0, DOWN_DEPTH=1, LIVOX=2};

class DepthKeypointExtractor
{
public:
    DepthKeypointExtractor();
    void uniformfeatureExtraction(const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &pc_in, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf, int skip_num, float block_range);
    
    void uniformfeatureExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf,int skip_num);
    void featureExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_edge, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_surf);
    void featureExtractionFromSector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, std::vector<Double2d>& cloudCurvature, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_edge, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_surf);

};


#endif //arise_slam_mid360_DEPTHIMAGEKEYPOINTEXTRACTOR_H
