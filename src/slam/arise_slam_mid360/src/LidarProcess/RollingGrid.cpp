
#include "arise_slam_mid360/LidarProcess/RollingGrid.h"

// a new PCL Point is added so we need to recompile PCL to able to use
// filters (pcl::VoxelGrid) with this new type

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/filters/voxel_grid.h>

RollingGrid::RollingGrid()
{
  this->Reset();
} // func

RollingGrid::RollingGrid(const Eigen::Vector3d& pos)
{
  this->Reset();

  // Initialize VoxelGrid center position
  this->VoxelGridPosition = (pos / this->VoxelResolution).array().floor().cast<int>();
} // function RollingGrid end

void RollingGrid::Reset()
{
  // Clear/reset empty voxel grid to right size
  this->setGridSize(this->GridSize);
  // Reset VoxelGrid center position
  this->VoxelGridPosition.setZero();
  // Reset min and max points of current frame
  int halfGridSize = std::ceil(this->GridSize / 2);
  this->MinPoint = -halfGridSize;
  this->MaxPoint = halfGridSize;
} // function Reset end

void RollingGrid::Clear()
{

  for (int x = 0; x < this->GridSize; x++)
  {
    for (int y = 0; y < this->GridSize; y++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        // if voxel is not already initilized, allocated memory
        auto& voxel = this->Grid[x][y][z];
        if (voxel)
          voxel->clear();
        else
          voxel.reset(new PointCloud);
      }
    }
  }
} // function Clear end

void RollingGrid::Roll(const Eigen::Vector3d& T)
{
  // very basic implementation where the grid is not circular.
  // This only moves VoxelGrid so that current frame can entirely
  // fit in rolled map.

  // Compute the position of the new frame center in the grid.
  Eigen::Array3i frameCenter = (T / this->VoxelResolution).array().floor().cast<int>();

  // Halt size of the VoxelGrid, rounded up.
  int halfGridSize = (this->GridSize + 1) / 2;

  // TODO:
  // Shift the voxel grid to the -X direction
  while (frameCenter.x() + this->MinPoint.x() < this->VoxelGridPosition.x() - halfGridSize)
  {
    for (int y = 0; y < this->GridSize; y++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        for (int x = this->GridSize - 1; x > 0; x--)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x - 1][y][z]);
        }
        this->Grid[0][y][z].reset(new PointCloud());
      }
    }
    this->VoxelGridPosition.x()--;
  } // Shift the voxel grid to the -X direction end

  // Shift the voxel grid to the +X direction
  while (frameCenter.x() + this->MaxPoint.x() > this->VoxelGridPosition.x() + halfGridSize)
  {
    for (int y = 0; y < this->GridSize; y++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        for (int x = 0; x < this->GridSize - 1; x++)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x + 1][y][z]);
        }
        this->Grid[this->GridSize - 1][y][z].reset(new PointCloud());
      }
    }
    this->VoxelGridPosition.x()++;
  } // Shift the voxel grid to the +X direction end

  // Shift the voxel grid to the -Y direction
  while (frameCenter.y() + this->MinPoint.y() < this->VoxelGridPosition.y() - halfGridSize)
  {
    for (int x = 0; x < this->GridSize; x++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        for (int y = this->GridSize - 1; y > 0; y--)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x][y - 1][z]);
        }
        this->Grid[x][0][z].reset(new PointCloud());
      }
    }
    this->VoxelGridPosition.y()--;
  } // Shift the voxel grid to the -Y direction end

  // Shift the voxel grid to the +Y direction.
  while (frameCenter.y() + this->MaxPoint.y() > this->VoxelGridPosition.y() + halfGridSize)
  {
    for (int x = 0; x < this->GridSize; x++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        for (int y = 0; y < this->GridSize - 1; y++)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x][y + 1][z]);
        }
        this->Grid[x][this->GridSize - 1][z].reset(new PointCloud());
      }
    }
    this->VoxelGridPosition.y()++;
  } // Shift the voxel grid to the +Y direction end

  // Shift the voxel grid to the -Z direction
  while (frameCenter.z() + this->MinPoint.z() < this->VoxelGridPosition.z() - halfGridSize)
  {
    for (int x = 0; x < this->GridSize; x++)
    {
      for (int y = 0; y < this->GridSize; y++)
      {
        for (int z = this->GridSize - 1; z > 0; z--)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x][y][z - 1]);
        }
        this->Grid[x][y][this->GridSize - 1].reset(new PointCloud());
      }
    }
    this->VoxelGridPosition.z()--;
  } // Shift the voxel grid to the -Z direction end

  // Shift the voxel grid to the +Z direction.
  while (frameCenter.z() + this->MaxPoint.z() > this->VoxelGridPosition.z() + halfGridSize)
  {
    for (int x = 0; x < this->GridSize; x++)
    {
      for (int y = 0; y < this->GridSize; y++)
      {
        for (int z = 0; z < this->GridSize - 1; z++)
        {
          this->Grid[x][y][z] = std::move(this->Grid[x][y][z + 1]);
        }
        this->Grid[x][y][this->GridSize - 1].reset(new PointCloud());
      }
    }

    this->VoxelGridPosition.z()++;
  } // Shift the voxel grid to the +Z direction.
} // function Roll end

RollingGrid::PointCloud::Ptr RollingGrid::Get(const Eigen::Vector3d& T) const
{
  // Compute the position of the new frame center in the grid
  Eigen::Array3i frameCenter = (T / this->VoxelResolution).array().floor().cast<int>() -
    (this->VoxelGridPosition - this->GridSize / 2);

  // Get sub-VoxelGrid bouds
  Eigen::Array3i intersectionMin = (frameCenter + this->MinPoint).max(0);
  Eigen::Array3i intersectionMax = (frameCenter + this->MaxPoint).min(this->GridSize - 1);

  // Get all voxel in intersection
  PointCloud::Ptr intersection(new PointCloud);

  for (int x = intersectionMin.x(); x <= intersectionMax.x(); x++)
  {
    for (int y = intersectionMin.y(); y <= intersectionMax.y(); y++)
    {
      for (int z = intersectionMin.z(); z < intersectionMax.z(); z++)
      {
        *intersection += *(this->Grid[x][y][z]);
      }
    }
  }
} // function RollingGrid::Get end

RollingGrid::PointCloud::Ptr RollingGrid::Get() const
{
  // Merge all points into a single pointcloud
  PointCloud::Ptr intersection(new PointCloud);
  for (int x = 0; x < this->GridSize; x++)
  {
    for (int y = 0; y < this->GridSize; y++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        *intersection += *(this->Grid[x][y][z]);
      }
    }
  }

  return intersection;
} // RollingGrid::Get() end

void RollingGrid::Add(const PointCloud::Ptr& pointcloud)
{
  if (pointcloud->empty())
  {
    std::cerr << "[WARNING] PointCLoud is empty, voxel grid not updated. \n";
    return;
  }

  // Voxels to filter because new points were added
  std::vector<std::vector<std::vector<uint8_t> > > voxelToFilter(this->GridSize,
    std::vector<std::vector<uint8_t> >(this->GridSize, std::vector<uint8_t>(this->GridSize, 0)));

  // 栅格的左上角
  // Compute the position of the origin of the VoxelGrid
  Eigen::Array3i voxelGridOrigin = this->VoxelGridPosition - this->GridSize / 2;

  // Add points in the rolling grid
  for (const Point& point : *pointcloud)
  {
    // Find the voxel containing this point
    Eigen::Array3i cubeIdx =
      (point.getArray3fMap() / this->VoxelResolution).floor().cast<int>() - voxelGridOrigin;

    // Add point to grid if it is indeed within bounds
    if (((0 <= cubeIdx) and (cubeIdx < this->GridSize)).all())
    {
      voxelToFilter[cubeIdx.x()][cubeIdx.y()][cubeIdx.z()] = 1;
      this->Grid[cubeIdx.x()][cubeIdx.y()][cubeIdx.z()]->push_back(point);
    }
  }

  // Filter the modifiled pointcloud
  pcl::VoxelGrid<Point> downSizeFilter;
  downSizeFilter.setLeafSize(this->LeafSize, this->LeafSize, this->LeafSize);

  for (int x = 0; x < this->GridSize; x++)
  {
    for (int y = 0; y < this->GridSize; y++)
    {
      for (int z = 0; z < this->GridSize; z++)
      {
        if (voxelToFilter[x][y][z])
        {
          PointCloud::Ptr tmp(new PointCloud());
          downSizeFilter.setInputCloud(this->Grid[x][y][z]);
          downSizeFilter.filter(*tmp);
          this->Grid[x][y][z] = tmp;
        }
      }
    }
  }
} // RollingGrid::Add(const PointCloud::Ptr& pointcloud)

void RollingGrid::SetMinMaxPoints(const Eigen::Array3d& minPoint, const Eigen::Array3d& maxPoint)
{
  this->MinPoint = (minPoint / this->VoxelResolution).floor().cast<int>();
  this->MaxPoint = (maxPoint / this->VoxelResolution).ceil().cast<int>();
} // function  RollingGrid::SetMinMaxPoints end

void RollingGrid::setGridSize(int size)
{
  this->GridSize = size;
  // Resize voxel grid
  this->Grid.resize(this->GridSize);
  for (int x = 0; x < this->GridSize; x++)
  {
    this->Grid[x].resize(this->GridSize);
    for (int y = 0; y < this->GridSize; y++)
    {
      this->Grid[x][y].resize(this->GridSize);
    }
  }

  // Clear current voxel grid and allocate new voxels
  this->Clear();
} // RollingGrid::SetGridSize
