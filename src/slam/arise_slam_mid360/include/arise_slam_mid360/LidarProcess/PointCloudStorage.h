//
// Created by ubuntu on 2020/9/17.
//

#ifndef POINTCLOUDSTORAGE_H
#define POINTCLOUDSTORAGE_H

// A new PCL Point is added so we need to recompile PCL to be able to use
// filters (pcl::io::OctreePointCloudCompression) with this new type
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>

#include <glog/logging.h>

//! PCD file data format
enum PCDFormat
{
  ASCII = 0,
  BINARY = 1,
  BINARY_COMPRESSED = 2
};

//! PointCloudStorage data type
enum PointCloudStorageType
{
  PCL_CLOUD = 0,
  OCTREE_COMPRESSED = 1,
  PCD_ASCII = 2,
  PCD_BINARY = 3,
  PCD_BINARY_COMPRESSED = 4
};

//------------------------------------------------------------------------------
/*!
 * @brief Save pointcloud to PCD file according to data format.
 * @param[in] path The path to PCD file to write pointcloud to.
 * @param[in] cloud The pointcloud to save.
 * @param[in] pcdDataFormat The PCD file data format to use.
 * @return 0 if ok, negative error number otherwise.
 */

template<typename PointT>
int savePointCloudToPCD(const std::string& path, pcl::PointCloud<PointT> const& cloud,
  PCDFormat pcdDataFormat, bool verbose = false)
{
  if (cloud.empty())
    return -3;
  switch (pcdDataFormat)
  {
    case ASCII:
      if (verbose)
      {
        LOG(INFO) << "Saving pointcloud to ascii PCD file at " << path;
      }

      return pcl::io::savePCDFileASCII<PointT>(path, cloud);

    case BINARY:
      if (verbose)
      {
        LOG(INFO) << "Saving pointcloud to binary PCD file at " << path;
      }

      return pcl::io::savePCDFileBinary<PointT>(path, cloud);

    case BINARY_COMPRESSED:
      if (verbose)
      {
        LOG(INFO) << "Saving pointcloud to binary_compressed PCD file at " << path;
      }

      return pcl::io::savePCDFileBinaryCompressed<PointT>(path, cloud);

    default:
      LOG(ERROR) << "[ERROR] Unknown PCDFormat value (" << pcdDataFormat
                 << "). Unable to save pointcloud.";

      return -4;
  }
}

//------------------------------------------------------------------------------
/*!
 * @brief Base abstract class to store a PCL pointcloud under different formats.
 */

template<typename PointT>
struct PointCloudData
{
  using CloudT = pcl::PointCloud<PointT>;
  using CloudTPtr = typename CloudT::Ptr;

  virtual ~PointCloudData() = default; ///< Fill with new cloud
  virtual CloudTPtr GetCloud() = 0;    ///< Get stored cloud
  virtual size_t GetMemorySize() = 0;  ///< Approximate memory usage by pointcloud data.
};

//------------------------------------------------------------------------------
/*!
 * @brief Store PCL pointcloud without any compression in RAM.
 */

template<typename PointT>
struct PCLPointCloud final : public PointCloudData<PointT>
{
  using CloudT = typename PointCloudData<PointT>::CloudT;
  using CloudTPtr = typename PointCloudData<PointT>::CloudTPtr;

  PCLPointCloud(const CloudTPtr& cloud)
    : Cloud(cloud)
  {
  }

  void SetCloud(const CloudTPtr& cloud) override { this->Cloud = cloud; }
  CloudTPtr GetCloud() override { return this->Cloud; }
  size_t GetMemorySize() override
  {
    return sizeof(*this->Cloud) + (sizeof(PointT) * this->Cloud->size());
  }

private:
  CloudTPtr Cloud; ///< Raw uncompressed pointcloud.
};

#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/io/impl/octree_pointcloud_compression.hpp>

//------------------------------------------------------------------------------
/*!
 * @brief Compress (with small loss) pointcloud with octree, and store pointcloud as binary data in
 * RAM.
 */

template<typename PointT>
struct OctreeCompressedPointCloud final : public PointCloudData<PointT>
{
  using CloudT = pcl::PointCloud<PointT>;
  using CloudTPtr = typename CloudT::Ptr;

  OctreeCompressedPointCloud(const CloudTPtr& cloud)
  {
    // Compress pointcoud data
    this->SetCloud(cloud);
  }

  void SetCloud(const CloudTPtr& cloud) override
  {
    // Octree compression
    // Octree compression
    pcl::io::OctreePointCloudCompression<PointT> compression(pcl::io::MANUAL_CONFIGURATION, false,
      0.001, // pointResolution
      0.05,  // octreeResolution
      false, // doVoxelGridDownDownSampling
      100,   // iFrameRate
      1,     // colorBitResolution
      false  // doColorEncoding
    );
    compression.encodePointCloud(cloud, this->CompressedData);
  }

  CloudTPtr GetCloud() override
  {
    // Decode compressed pointcloud
    CloudTPtr cloud(new CloudT);
    pcl::io::OctreePointCloudCompression<PointT> compression;

    try
    {
      compression.decodePointCloud(this->CompressdData, cloud);
    }
    catch (std::logic_error e)
    {
      LOG(ERROR) << "[ERROR] Decompression failed. Returning empty pointcloud. ";
    }

    // Set back compressed data read position to beginning (missing in
    // OctreePointCloudCompression::decodePointCloud())
    this->CompressdData.seekg(0, std::ios::beg);
  }

  size_t GetMemorySize() override
  {
    std::streampos current = this->CompressdData.tellp();
    this->CompressdData.seekp(0, std::ios::end);
    size_t size = this->CompressdData.tellp();
    this->CompressdData.seekp(current, std::ios::beg);
    return size;
  }

private:
  std::stringstream CompressdData; ///< Binary compressed pointcloud data
};

//------------------------------------------------------------------------------
/*!
 * @brief Store PCL pointcloud on disk as PCD file.
 */

template<typename PointT, PCDFormat pcdFormat>
struct PCDFilePointCloud final : public PointCloudData<PointT>
{
  using CloudT = pcl::PointCloud<PointT>;
  using CloudTPtr = typename CloudT::Ptr;

  PCDFilePointCloud(const CloudTPtr& cloud, const std::string& pcdDirPath = "point_cloud_log/")
  {
    boost::filesystem::create_directory(pcdDirPath);
    this->PCDFilePath = pcdDirPath + std::to_string(this->PCDFileIndex) + ".pcd";
    this->PCDFileIndex++;
    this->SetCloud(cloud);
  }
  // Define default constructor/assignement/move semantics as ~PCDFilePointCloud
  // is specified
  PCDFilePointCloud() = default;
  PCDFilePointCloud(const PCDFilePointCloud&) = default;
  PCDFilePointCloud(PCDFilePointCloud&&) = default;
  PCDFilePointCloud& operator=(const PCDFilePointCloud&) = default;
  PCDFilePointCloud& operator=(PCDFilePointCloud&&) = default;

  ~PCDFilePointCloud() override
  {
    if (std::remove(this->PCDFilePath.c_str()) != 0)
    {
      LOG(ERROR) << "[WARNING] Unable to delete PCD file at " << this->PCDFilePath;
    }
  }

  void SetCloud(const CloudTPtr& cloud) override
  {
    if (savePointCloudToPCD(this->PCDFilePath, *cloud, pcdFormat) != 0)
    {
      LOG(ERROR) << "[ERROR] Failed to write binary PCD file to " << this->PCDFilePath;
    }
  }

  CloudTPtr GetCloud() override
  {
    CloudTPtr cloud(new CloudT);
    if (pcl::io::loadPCDFile(this->PCDFilePath, *cloud) != 0)
    {
      LOG(ERROR) << "[ERROR] PCD file loading failed. Returning empty pointcloud.";
    }
    return cloud;
  }

  size_t GetMemorySize() override
  {
    std::ifstream in(this->PCDFilePath, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
  }

protected:
  static size_t PCDFileIndex; ///< The index of the PCD file currently being written/
  std::string PCDFilePath;    ///< Path to PCD file
};

template<typename PointT, PCDFormat pcdFormat>
size_t PCDFilePointCloud<PointT, pcdFormat>::PCDFileIndex = 0;

template<typename PointT>
using AsciiPCDFilePointCloud = PCDFilePointCloud<PointT, PCDFormat::ASCII>;

template<typename PointT>
using BinaryPCDFilePointCloud = PCDFilePointCloud<PointT, PCDFormat::BINARY>;

template<typename PointT>
using BinaryCompressedPCDFilePointCloud = PCDFilePointCloud<PointT, PCDFormat::BINARY_COMPRESSED>;

//------------------------------------------------------------------------------
/*!
 * @brief Structure used to log pointclouds either under uncompressed/compressed format.
 */
template<typename PointT>
struct PointCloudStorage
{
  using CloudT = pcl::PointCloud<PointT>;
  using CloudTPtr = typename CloudT::Ptr;

  inline PointCloudStorageType StorageType() const {return this->Storage;}
  inline size_t PointsSize() const {return this->Points;}
  inline size_t MemorySize() const {return this->Data->GetMemorySize();}

  void SetCloud(const CloudTPtr& cloud, PointCloudStorageType storage)
  {
    this->Storage = storage;
    this->Points = cloud->size();

    switch (storage)
    {
      case PCL_CLOUD:
        this->Data.reset(new PCLPointCloud<PointT>(cloud));
        break;
      case OCTREE_COMPRESSED:
        this->Data.reset(new OctreeCompressedPointCloud<PointT>(cloud));
        break;

      case PCD_ASCII:
        this->Data.reset(new AsciiPCDFilePointCloud<PointT>(cloud));
        break;

      case PCD_BINARY:
        this->Data.reset(new BinaryCompressedPCDFilePointCloud<PointT>(cloud));
        break;
      case PCD_BINARY_COMPRESSED:
        this->Data.reset(new BinaryCompressedPCDFilePointCloud<PointT>(cloud));
        break;

      default:
        LOG(INFO) << "[ERROR] Unknow PointCloudStorageType (" << storage << ").\n";
        break;
    }
  }

  CloudTPtr GetCloud() const { return this->Data->GetCloud(); }

private:
  size_t Points;                                 ///< Number of points in stored pointcloud.
  PointCloudStorageType Storage;                 ///< How is stored pointcloud data.
  std::unique_ptr<PointCloudData<PointT> > Data; ///< Pointcloud data.
};
#endif // POINTCLOUDSTORAGE_H
