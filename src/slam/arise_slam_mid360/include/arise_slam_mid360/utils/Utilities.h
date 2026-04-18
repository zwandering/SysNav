//
// Created by ubuntu on 2020/9/16.
//

#ifndef UTILITIES_H
#define UTILITIES_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <unordered_map>

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

#include <pcl/point_cloud.h>

#include "arise_slam_mid360/utils/Twist.h"

// Set cout to print floating values with fixed precision of a given number of decimals
#define SET_COUT_FIXED_PRECISION(decimals)                                                                             \
    std::streamsize ss = std::cout.precision();                                                                        \
    std::cout << std::fixed << std::setprecision(decimals);

// Reset cout float printing state back to before
// NOTE : RESET_COUT_FIXED_PRECISION in the same scope as SET_COUT_FIXED_PRECISION
#define RESET_COUT_FIXED_PRECISION                                                                                     \
    std::cout << std::setprecision(ss);                                                                                \
    std::cout.unsetf(std::ios::fixed | std::ios_base::floatfield);

// Anonymous namespace to avoid multiple-definitions
namespace {
//------------------------------------------------------------------------------
/*!
 * @brief Sort a vector and return sorted indices
 * @param v The vector to sort
 * @param ascending If true, sort in ascending (increasing) order
 * @return The sorted indices such that the first index is the biggest input
 *         value and the last the smallest.
 */

    template<typename T>
    std::vector<size_t> SortIdx(const std::vector<T> &v, bool ascending = false) {
        // Initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // Sort indices based on comparing values in v
        if (ascending)
            std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
        else
            std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

        return idx;
    }

//==============================================================================
//   Geometry helpers
//==============================================================================

//------------------------------------------------------------------------------
/*!
 * @brief Convert from radians to degrees
 * @param rad The angle value, in radians
 * @return The angle value, in degrees
 */
    template<typename T>
    inline constexpr T Rad2Deg(const T &rad) { return rad / M_PI * 180.; }

//------------------------------------------------------------------------------
/*!
 * @brief Convert from degrees to radians
 * @param deg The angle value, in degrees
 * @return The angle value, in radians
 */
    template<typename T>
    inline constexpr T Deg2Rad(const T &deg) { return deg / 180. * M_PI; }

    template<typename PointT>
    inline void TransformPoint(PointT &p, const Transformd &transform) {
        Eigen::Vector3d temp = p.getVector3fMap().template cast<double>();
        p.getVector3fMap() = (transform * temp).template cast<float>();
    }

    template<typename PointT>
    inline PointT TransformPointd(const PointT &p, const Transformd &transform) {
        PointT out(p);
        TransformPoint(out, transform);
        return out;
    }
//------------------------------------------------------------------------------
/*!
 * @brief Copy pointcloud metadata to an other cloud
 * @param[in] from The pointcloud to copy info from
 * @param[out] to The pointcloud to copy info to
 */
    template<typename PointT, typename PointU>
    inline void CopyPointCloudMetadata(const pcl::PointCloud<PointT> &from, pcl::PointCloud<PointU> &to) {
        to.header = from.header;
        to.is_dense = from.is_dense;
        to.sensor_orientation_ = from.sensor_orientation_;
        to.sensor_origin_ = from.sensor_origin_;
    }

    inline void TranfromPointXYZITR2PointXYZTIIdCloud(const pcl::PointCloud<PointXYZITR> &from,
                                                      pcl::PointCloud<PointXYZTIId> &to) {
        to.header = from.header;
        to.is_dense = from.is_dense;
        to.sensor_orientation_ = from.sensor_orientation_;
        to.sensor_origin_ = from.sensor_origin_;

        for (const auto &old_point : from.points) {
            PointXYZTIId new_point;
            new_point.x = old_point.x;
            new_point.y = old_point.y;
            new_point.z = old_point.z;
            new_point.intensity = old_point.intensity * 255;
            //    LOG(INFO) << "new_point.intensity: " << int(new_point.intensity);
            new_point.time = old_point.time;
            new_point.laserId = old_point.ring;
            to.push_back(new_point);
        }
    }

    inline void TransfromPointXYZICloud2PointXYZTIIdCloud(const pcl::PointCloud<pcl::PointXYZI> &from,
                                                          pcl::PointCloud<PointXYZTIId> &to,
                                                          int NbrLaser) {
        to.header = from.header;
        to.is_dense = from.is_dense;
        to.sensor_orientation_ = from.sensor_orientation_;
        to.sensor_origin_ = from.sensor_origin_;

        size_t Npts = from.points.size();

        // variable used to detect a laser jump
        double old_thetaProj = 0;
        int laser_id = 0;

        for (size_t i = 0; i < Npts; i++) {
            const auto &old_point = from.points[i];

            PointXYZTIId new_point;
            new_point.x = old_point.x;
            new_point.y = old_point.y;
            new_point.z = old_point.z;
            new_point.intensity = old_point.intensity;

            //    LOG(INFO) << "old_point.intensity: " << old_point.intensity;
            double thetaProj = 180 / M_PI * std::atan2(old_point.y, old_point.x);
            double azimut = 180 / M_PI * std::atan2(old_point.x, old_point.y);

            if (azimut < 0)
                azimut = 360 + azimut;

            // 由负到到正,新的scan生成
            if (old_thetaProj < 0 && thetaProj >= 0) {
                laser_id++;
                if (laser_id >= NbrLaser) {
                    std::cout << "An error occur while parsing the frame, more than 64 lasers where detected. "
                                 "The last point won't be processed"
                              << std::endl;
                    break;
                }
            }

            double time = azimut / 360.0;

            // LOG(INFO) << "old_point.time: " << time;
            new_point.laserId = static_cast<uint8_t>(laser_id);
            new_point.time = time;
            to.push_back(new_point);
        }  // 每一个激光点处理完毕
    }  // TransfromPointXYZICloud2PointXYZTIIdCloud function end

//------------------------------------------------------------------------------
/*!
 * @brief Compute PCA of Nx3 data array and mean value
 * @param[in] data Nx3 array (e.g. stacked 3D points)
 * @param[out] mean Where to store mean value
 * @return The PCA
 */
    inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
    ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3> &data,
               Eigen::Vector3d &mean) {
        mean = data.colwise().mean();
        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        Eigen::Matrix3d varianceCovariance = centered.transpose() * centered;

        return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(varianceCovariance);
    }

//------------------------------------------------------------------------------
/*!
 * @brief Compute PCA of Nx3 data array and mean value
 * @param data Nx3 array (e.g. stacked 3D points)
 * @return The PCA
 */
    inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
    ComputePCA(const Eigen::Matrix<double, Eigen::Dynamic, 3> &data) {
        Eigen::Vector3d mean;
        return ComputePCA(data, mean);
    }

//==============================================================================
//   Processing duration measurements
//==============================================================================

//------------------------------------------------------------------------------
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> startTimestamps;
    std::unordered_map<std::string, double> totalDurations;
    std::unordered_map<std::string, size_t> totalCalls;

//------------------------------------------------------------------------------
/*!
 * @brief Init a timer.
 * @param timer The name of the timer
 */
    inline void InitTime(const std::string &timer) { startTimestamps[timer] = std::chrono::steady_clock::now(); }

//------------------------------------------------------------------------------
/*!
 * @brief Get the timer value.
 * @param timer The name of the timer
 * @return The duration value, in seconds, since the initialization of the timer
 *
 * NOTE : This returns 0 if the counter has not been initialized yet.
 */

    inline double GetTime(const std::string &timer) {
        std::chrono::duration<double> chrono_ms = std::chrono::steady_clock::now() - startTimestamps[timer];

        return chrono_ms.count();
    }

//------------------------------------------------------------------------------
/*!
 * @brief Print a given timer value and its average value
 * @param timer The name of the timer
 *
 * NOTE : This returns 0 if the counter has not been initialized yet.
 */
    inline void StopTimeAndDisplay(const std::string &timer) {
        const double currentDuration = GetTime(timer);
        totalDurations[timer] += currentDuration;
        totalCalls[timer]++;
        double meanDurationMs = totalDurations[timer] * 1000. / totalCalls[timer];

        std::cout << " ->" << timer << " took : " << currentDuration * 1000. << "ms (average : " << meanDurationMs
                  << " ms)"
                  << std::endl;
    }

    inline Eigen::Matrix3d GetRotationMatrix(Eigen::Matrix<double, 6, 1> T) {
        return Eigen::Matrix3d(
                Eigen::AngleAxisd(T(2), Eigen::Vector3d::UnitZ())     /* rotation around Z-axis */
                * Eigen::AngleAxisd(T(1), Eigen::Vector3d::UnitY())     /* rotation around Y-axis */
                * Eigen::AngleAxisd(T(0), Eigen::Vector3d::UnitX()));   /* rotation around X-axis */
    }


    inline bool compare_pair_first(const std::pair<float, int> a, const std::pair<float, int> b) // sort from big to small
    {
        return a.first > b.first;
    }

}  // end of anonymous namespace
#endif  // UTILITIES_H
