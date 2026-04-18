//
// Created by ubuntu on 2020/9/26.
//

#ifndef ROLLINGGRID_H
#define ROLLINGGRID_H

#include <memory>
#include "arise_slam_mid360/sensor_data/pointcloud/LidarPoint.h"


#define SetMacro(name, type)                                                                       \
  void set##name(type _arg) { name = _arg; }
#define GetMaroc(name, type)                                                                       \
  type get##name() const { return name; }

class RollingGrid
{
public:
  // Usefull types
  using Point = PointXYZTIId;
  using PointCloud = pcl::PointCloud<Point>;

  using Ptr = std::shared_ptr<RollingGrid>;

  RollingGrid();

  explicit RollingGrid(const Eigen::Vector3d& pos);
  //! Roll the grid to enable adding new point cloud
  void Roll(const Eigen::Vector3d& T);

  //! Get point near T
  PointCloud::Ptr Get(const Eigen::Vector3d& T) const;

  //! Get all points
  PointCloud::Ptr Get() const;

  //! Add some points to the grid
  void Add(const PointCloud::Ptr& pointcloud);

  //! Remove all points from all voxels
  void Clear();

  //! Reset map (clear voxels, rest position, ...)
  void Reset();

  //! Set min and max keypoints bouds of the current frame
  void SetMinMaxPoints(const Eigen::Array3d& minPoint, const Eigen::Array3d& maxPoint);

  //! Set grid size and clear all points from voxels
  void setGridSize(int size);
  GetMaroc(GridSize, int);

  SetMacro(VoxelResolution, double);
  GetMaroc(VoxelResolution, double);

  SetMacro(LeafSize, double);
  GetMaroc(LeafSize, double);

private:
  //! [voxels] Size of the voxel grid: nxnxn voxels
  int GridSize = 50;

  //! [m/voxel] Resolution of a voxel
  double VoxelResolution = 10.0;

  //! [m] size of the leaf used to downsample the pointcloud with a VoxelGrid filter within each
  //! voxel
  double LeafSize = 0.2;
  //! VoxelGrid of pointcloud
  std::vector<std::vector<std::vector<PointCloud::Ptr> > > Grid;

  //! [voxel, voxel, voxel] Current position of the center of the VoxelGrid
  Eigen::Array3i VoxelGridPosition;

  //! [voxels] Minimum ans maximum coordinates in voxel grid
  Eigen::Array3i MinPoint, MaxPoint;
};
#endif // ROLLINGGRID_H
