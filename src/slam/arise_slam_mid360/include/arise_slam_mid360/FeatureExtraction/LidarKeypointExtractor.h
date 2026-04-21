//
// Created by ubuntu on 2020/9/14.
//

#ifndef LIDARKEYPOINTEXTRACTOR_H
#define LIDARKEYPOINTEXTRACTOR_H

#include <bitset>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include "arise_slam_mid360/config/parameter.h"
#include "arise_slam_mid360/sensor_data/pointcloud/LidarPoint.h"

//! Label of a point as a keypoint
//! We use binary flags as each point can have different keypoint labels.
using KeypointFlags = std::bitset<3>;
enum Keypoint
{
  EDGE = 0,  ///< edge keypoint (sharp local structure)
  PLANE = 1, ///< plane keypoint (flat local structure)
  BLOB = 2   ///< blob keypoint (spherical local structure)
};

// clang-format off
#define SetMacro(name, type)    void set##name(type _arg) { name = _arg; }
#define GetMacro(name, type)    type get##name() const { return name; }
// clang-format on

class LidarKeypointExtractor
{
public:
  using Point = PointXYZTIId;
  using Ptr = std::shared_ptr<LidarKeypointExtractor>;

  LidarKeypointExtractor() = default;

  LidarKeypointExtractor(int nLasers, std::vector<size_t>* pLaserIdMapping = nullptr)
  {
    this->NLasers = nLasers;
    if (pLaserIdMapping)
    {
      this->LaserIdMapping = *pLaserIdMapping;
    }
    else
    {
      this->LaserIdMapping.resize(this->NLasers);
      std::iota(this->LaserIdMapping.begin(), this->LaserIdMapping.end(), 0);
    }
  }

  void setNLasers(int nLasers)
  {
    this->NLasers = nLasers;
    if (this->LaserIdMapping.size() != static_cast<size_t>(nLasers))
    {
      this->LaserIdMapping.resize(this->NLasers);
      std::iota(this->LaserIdMapping.begin(), this->LaserIdMapping.end(), 0);
    }
  }

  // clang-format off
  GetMacro(NbThreads, int)
  SetMacro(NbThreads, int)

  GetMacro(NeighborWidth, int)
  SetMacro(NeighborWidth, int)

  GetMacro(MinDistanceToSensor, double)
  SetMacro(MinDistanceToSensor, double)

  GetMacro(MaxDistanceToSensor, double)
  SetMacro(MaxDistanceToSensor, double)

  GetMacro(AngleResolution, double)
  SetMacro(AngleResolution, double)

  GetMacro(PlaneSinAngleThreshold, double)
  SetMacro(PlaneSinAngleThreshold, double)

  GetMacro(EdgeSinAngleThreshold, double)
  SetMacro(EdgeSinAngleThreshold, double)

  GetMacro(EdgeDepthGapThreshold, double)
  SetMacro(EdgeDepthGapThreshold, double)

  GetMacro(EdgeSaliencyThreshold, double)
  SetMacro(EdgeSaliencyThreshold, double)

  GetMacro(EdgeIntensityGapThreshold, double)
  SetMacro(EdgeIntensityGapThreshold, double)

  GetMacro(NLasers, int)

  pcl::PointCloud<Point>::Ptr GetEdgePoints() { return this->EdgesPoints; }
  pcl::PointCloud<Point>::Ptr GetPlanarPoints() { return this->PlanarsPoints; }
  pcl::PointCloud<Point>::Ptr GetBlobPoints() { return this->BlobsPoints; }

  // Extract keypoints from the pointcloud. 
  void ComputeKeyPoints(pcl::PointCloud<Point>::Ptr pc, int N_scans, bool dynamic_mask);

  // Function to enable to have some inside on why a given point was detected as a keypoint
  std::unordered_map<std::string, std::vector<double> > GetDebugArray();

private:
  // Reset all mumbers variables that are
  // used during the process of a frame.
  void PrepareDateForNextFrame();

  // Convert the input vtk-format pointcloud
  void ConvertAndSortScanLines();

  // Compute the curvature of the scan lines
  void ComputeCurvature();

  // Invalid the points with bad criteria 
  void InvalidPointWithBadCriteria(bool dynamic_mask);

  // Labelizes point to be a keypoints or not
  void SetKeyPointsLabels();


  // check if scanline is almost empty
  inline bool IsScanLineAlmostEmpty(int nScanLinePts) const {return nScanLinePts < 2*this->NeighborWidth+1; }

public:
  // ---------------------------------------------------------------------------
  //   Parameters
  // ---------------------------------------------------------------------------

  // Max number of threads to use to process points in parallel
  int NbThreads = 1;

  // Width of the neighborhood used to compute discrete differential operators
  int NeighborWidth = 4;

  // Minimal point/sensor sensor to consider a point as valid
  double MinDistanceToSensor;  // [m]

  // Maxm point/sensor sensor to consider a point as valid
  double MaxDistanceToSensor; // [m]

  // Maximal angle resolution of the lidar azimutal resolution.
  // (default value to VLP-16. We add an extra 20%)
  double AngleResolution = DEG2RAD(0.4);  // [rad]

  // Sharpness threshold to select a planar keypoint
  double PlaneSinAngleThreshold = 0.6;  // sin(45°) (selected if sin angle is less than threshold)

  // Sharpness threshold to select an edge keypoint
  double EdgeSinAngleThreshold = 0.86;  // ~sin(60°) (selected, if sin angle is more than threshold)
  double DistToLineThreshold = 0.20;  // [m]

  // Threshold upon depth gap in neighborhood to select an edge keypoint
  double EdgeDepthGapThreshold = 0.15;  // [m]

  // Threshold upon saliency of a neighborhood to select an edge keypoint
  double EdgeSaliencyThreshold = 1.5;  // [m]

  // Threshold upon intensity gap to select an edge keypoint
  double EdgeIntensityGapThreshold = 50.;

  // Threshold upon sphericity of a neighborhood to select a blob point
  double SphericityThreshold = 0.35;  // CHECK : unused

  // Coef to apply to the incertitude radius of the blob neighborhood
  double IncertitudeCoef = 3.0;  // CHECK : unused

  // ---------------------------------------------------------------------------
  //   Internal variables
  // ---------------------------------------------------------------------------

  // Mapping of the lasers id
  std::vector<size_t> LaserIdMapping;

  // Number of lasers scan lines composing the pointcloud
  unsigned int NLasers = 0;

  // Curvature and over differntial operations
  std::vector<std::vector<double> > Angles;
  std::vector<std::vector<double> > DepthGap;
  std::vector<std::vector<double> > Saliency;
  std::vector<std::vector<double> > IntensityGap;
  std::vector<std::vector<KeypointFlags> > IsPointValid;
  std::vector<std::vector<KeypointFlags> > Label;

  pcl::PointCloud<Point>::Ptr EdgesPoints;
  pcl::PointCloud<Point>::Ptr PlanarsPoints;
  pcl::PointCloud<Point>::Ptr BlobsPoints;
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr RawPoints;

  // Current point cloud stored in two differents
  // formats
  pcl::PointCloud<Point>::Ptr pclCurrentFrame;
  std::vector<pcl::PointCloud<Point>::Ptr>
    pclCurrentFrameByScan;
  // clang-format on
};

#endif // LIDARKEYPOINTEXTRACTOR_H
