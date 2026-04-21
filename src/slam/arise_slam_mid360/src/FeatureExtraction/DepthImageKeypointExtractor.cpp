//
// Created by shiboz on 2021-09-04.
//


#include "arise_slam_mid360/FeatureExtraction/DepthImageKeypointExtractor.h"


void DepthKeypointExtractor::uniformfeatureExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf, int skip_num) {

    for (int i = 0; i < (int) pc_in->points.size(); i+= skip_num)
    {   
        
        if(pc_in->points[i].z<-9.0||pc_in->points[i].z>9.0)
            continue;
        pcl::PointXYZI point;
        point.x=pc_in->points[i].x;
        point.y=pc_in->points[i].y;
        point.z=pc_in->points[i].z;
        point.intensity=pc_in->points[i].z;
        pc_out_surf->push_back(point);
    }

}

void DepthKeypointExtractor::uniformfeatureExtraction(const pcl::PointCloud<point_os::PointcloudXYZITR>::Ptr &pc_in, 
pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf, int skip_num, float block_range)
    {   
        for (uint i=1; i <(int)pc_in->points.size(); i+=skip_num)
        {   
            pcl::PointXYZI point;
            point.x=pc_in->points[i].x;
            point.y=pc_in->points[i].y;
            point.z=pc_in->points[i].z;
            point.intensity=pc_in->points[i].time;

            if ((abs(pc_in->points[i].x - pc_in->points[i-1].x) > 1e-7)
                || (abs(pc_in->points[i].y - pc_in->points[i-1].y) > 1e-7)
                || (abs(pc_in->points[i].z - pc_in->points[i-1].z) > 1e-7)
                && (pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y + pc_in->points[i].z * pc_in->points[i].z > (block_range * block_range)))
            {
                pc_out_surf->push_back(point);
            }
        
        }
        
    }


void DepthKeypointExtractor::featureExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_edge, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_surf){

#if 0
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, indices);

    //step1: coordinate transform
    for (int i = 0; i < (int) pc_in->points.size(); i++){
        double new_x = pc_in->points[i].z;
        double new_y = -pc_in->points[i].x;
        double new_z = -pc_in->points[i].y;
        pc_in->points[i].x = new_x;
        pc_in->points[i].y = new_y;
        pc_in->points[i].z = new_z;
    }
#endif

    // RCLCPP_WARN(this->get_logger(), "input feature %d", pc_in->points.size());
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> laserCloudScans;

    // step1: calculate the first horizontal angle
    double last_angle = atan2(pc_in->points[0].z,pc_in->points[0].y) * 180 / M_PI;
    int count =0;
    int point_size = pc_in->points.size()-1;

    for (int i = 0; i < (int) pc_in->points.size(); i++)
    {
        // step2: remove the unreliable feature points
        if(pc_in->points[i].z<-9.0||pc_in->points[i].z>9.0)
            continue;

        // step3: classify the feature points according to the vertical angle

        int scanID=0;
        double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y + pc_in->points[i].z * pc_in->points[i].z);
        double angle = atan2(pc_in->points[i].x,pc_in->points[i].z) * 180 / M_PI;

        //float angle = atan(pc_in->points[i].x / sqrt(pc_in->points[i].y * pc_in->points[i].y + pc_in->points[i].z  * pc_in->points[i].z)) * 180 / M_PI;
        count++;

        // RCLCPP_WARN(this->get_logger(), "last_angle %lf, anlge %lf, diff %f", last_angle,angle,fabs(angle - last_angle));
        // we assume the vertical angle resolution is 0.05
        if(fabs(angle - last_angle)>0.05){
            if(count>30){                            // we only regard the number of point which is larger than 30 as one scan
                pcl::PointCloud<pcl::PointXYZ>::Ptr pc_temp(new pcl::PointCloud<pcl::PointXYZ>());
                for(int k=0;k<count;k++){
                    pc_temp->push_back(pc_in->points[i-count+k+1]);
                }
                if(pc_temp->points.size()>0)
                    laserCloudScans.push_back(pc_temp);
            }
            count =0;
            last_angle = angle;
        }

    }

    // RCLCPP_WARN(this->get_logger(), "total points array %d", laserCloudScans.size());

    // step4: calculate the curvature
    for(int i = 0; i < laserCloudScans.size(); i++){

        std::vector<Double2d> cloudCurvature;
        int total_points = laserCloudScans[i]->points.size()-10;
        for(int j = 5; j < (int)laserCloudScans[i]->points.size() - 5; j++){  //calculate the difference for in one laser scan

#if 0
            double angle_difference = fabs((atan2(laserCloudScans[i]->points[j-5].y,laserCloudScans[i]->points[j-5].z)-
                    atan2(laserCloudScans[i]->points[j+5].y,laserCloudScans[i]->points[j+5].z)) * 180 / M_PI);
            if(angle_difference>5){  // TODO: 为什么大于５度就划分成平面点？
                //consider as a surf points
                pc_out_surf->push_back(laserCloudScans[i]->points[j]);
                continue;
            }
#endif

            double diffX = laserCloudScans[i]->points[j - 5].x + laserCloudScans[i]->points[j - 4].x + laserCloudScans[i]->points[j - 3].x + laserCloudScans[i]->points[j - 2].x + laserCloudScans[i]->points[j - 1].x - 10 * laserCloudScans[i]->points[j].x + laserCloudScans[i]->points[j + 1].x + laserCloudScans[i]->points[j + 2].x + laserCloudScans[i]->points[j + 3].x + laserCloudScans[i]->points[j + 4].x + laserCloudScans[i]->points[j + 5].x;
            double diffY = laserCloudScans[i]->points[j - 5].y + laserCloudScans[i]->points[j - 4].y + laserCloudScans[i]->points[j - 3].y + laserCloudScans[i]->points[j - 2].y + laserCloudScans[i]->points[j - 1].y - 10 * laserCloudScans[i]->points[j].y + laserCloudScans[i]->points[j + 1].y + laserCloudScans[i]->points[j + 2].y + laserCloudScans[i]->points[j + 3].y + laserCloudScans[i]->points[j + 4].y + laserCloudScans[i]->points[j + 5].y;
            double diffZ = laserCloudScans[i]->points[j - 5].z + laserCloudScans[i]->points[j - 4].z + laserCloudScans[i]->points[j - 3].z + laserCloudScans[i]->points[j - 2].z + laserCloudScans[i]->points[j - 1].z - 10 * laserCloudScans[i]->points[j].z + laserCloudScans[i]->points[j + 1].z + laserCloudScans[i]->points[j + 2].z + laserCloudScans[i]->points[j + 3].z + laserCloudScans[i]->points[j + 4].z + laserCloudScans[i]->points[j + 5].z;

            Double2d distance(j,diffX * diffX + diffY * diffY + diffZ * diffZ);
            cloudCurvature.push_back(distance);

        }

        // Extract feature for each scan
        featureExtractionFromSector(laserCloudScans[i],cloudCurvature, pc_out_edge, pc_out_surf);


    }


    //remove ground point
    /*
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for(int i=0;i<pc_out_edge->points.size();i++){
        if(pc_out_edge->points[i].z<=-0.55)
            inliers->indices.push_back(i);
    }
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(pc_out_edge);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*pc_out_edge);
*/

}


void DepthKeypointExtractor::featureExtractionFromSector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_in, std::vector<Double2d>& cloudCurvature, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_edge, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_out_surf){

    // step1 : make the  curvature in order
    std::sort(cloudCurvature.begin(), cloudCurvature.end(), [](const Double2d & a, const Double2d & b)
    {
        return a.value < b.value;
    });


    int largestPickedNum = 0;
    std::vector<int> picked_points;
    int point_info_count =0;


    for (int i = cloudCurvature.size()-1; i >= 0; i--)
    {
        int ind = cloudCurvature[i].id;
        if(std::find(picked_points.begin(), picked_points.end(), ind)==picked_points.end()){

            if(cloudCurvature[i].value <= 0.1){   //TODO: necessary?
                break;
            }

            largestPickedNum++;
            picked_points.push_back(ind);

            // step 2: select the most top 10 curvature points as edge point
            if (largestPickedNum <= 10){
                pc_out_edge->push_back(pc_in->points[ind]);           //TODO: necessary?
                point_info_count++;
            }else{
                break;
            }

            // step 3: select the plannar features
            for(int k=1;k<=5;k++){
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05){
                    break;
                }
                picked_points.push_back(ind+k);
            }

            for(int k=-1;k>=-5;k--){
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05){
                    break;
                }
                picked_points.push_back(ind+k);
            }

        }
    }


    for (int i = 0; i <= (int)cloudCurvature.size()-1; i++)
    {
        int ind = cloudCurvature[i].id;
        if( std::find(picked_points.begin(), picked_points.end(), ind)==picked_points.end())
        {
            pc_out_surf->push_back(pc_in->points[ind]);
        }
    }


}

DepthKeypointExtractor::DepthKeypointExtractor(){

}

Double2d::Double2d(int id_in, double value_in){
    id = id_in;
    value =value_in;
};


