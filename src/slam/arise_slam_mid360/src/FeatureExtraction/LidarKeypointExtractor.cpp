//
// Created by shibo zhao on 2020-09-27.
//

// std
#include <numeric>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

// self
#include "arise_slam_mid360/FeatureExtraction/LidarKeypointExtractor.h"
#include "arise_slam_mid360/utils/Utilities.h"
#include "arise_slam_mid360/utils/EigenTypes.h"

//-----------------------------------------------------------------------------
namespace {
    class LineFitting {
    public:

        bool FitPCA(Eigen::aligned_vector<Eigen::Vector3d> &points);


        bool FitPCAAndCheckConsistency(Eigen::aligned_vector<Eigen::Vector3d> &points);

        inline double SquaredDistanceToPoint(Eigen::Vector3d const &point) const;


        Eigen::Vector3d Direction;
        Eigen::Vector3d Position;

        double MaxDistance = 0.02;      // [mn]
        double MaxAngle = DEG2RAD(40.); // [rad]
    };

    bool LineFitting::FitPCA(Eigen::aligned_vector<Eigen::Vector3d> &points) {


        Eigen::MatrixXd data(points.size(), 3);

        for (size_t k = 0; k < points.size(); k++) {
            data.row(k) = points[k];
        }


        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig = ComputePCA(data, this->Position);

        this->Direction = eig.eigenvectors().col(2).normalized();

        bool isLineFittingAccurate = true;

        const double squaredMaxDistance = this->MaxDistance * this->MaxDistance;
        for (const Eigen::Vector3d &point : points) {
            if (this->SquaredDistanceToPoint(point) > squaredMaxDistance) {
                isLineFittingAccurate = false;
                break;
            }
        }

        return isLineFittingAccurate;
    }

//-----------------------------------------------------------------------------
    bool LineFitting::FitPCAAndCheckConsistency(Eigen::aligned_vector<Eigen::Vector3d> &points) {
        const double maxSinAngle = std::sin(this->MaxAngle);
        bool isLineFittingAccurate = true;


        Eigen::Vector3d U, V;
        U = (points[1] - points[0]).normalized();
        for (size_t index = 1; index < points.size() - 1; index++) {
            V = (points[index + 1] - points[index]).normalized();
            double sin_angle = (U.cross(V)).norm();
            if (sin_angle > maxSinAngle) {
                isLineFittingAccurate = false;
                break;
            }
        }
        return isLineFittingAccurate && this->FitPCA(points);
    }

//-----------------------------------------------------------------------------
    inline double LineFitting::SquaredDistanceToPoint(Eigen::Vector3d const &point) const {
        return ((point - this->Position).cross(this->Direction)).squaredNorm();
    }
}

//-----------------------------------------------------------------------------

void LidarKeypointExtractor::PrepareDateForNextFrame() {

    this->pclCurrentFrameByScan.resize(this->NLasers);
    for (auto &scanLineCloud : this->pclCurrentFrameByScan) {

        if (scanLineCloud)
            scanLineCloud->clear();
        else
            scanLineCloud.reset(new pcl::PointCloud<Point>());
    }


    this->EdgesPoints.reset(new pcl::PointCloud<Point>());
    this->PlanarsPoints.reset(new pcl::PointCloud<Point>());
    this->BlobsPoints.reset(new pcl::PointCloud<Point>());
    this->RawPoints.reset(new pcl::PointCloud<pcl::PointXYZHSV>());
    CopyPointCloudMetadata(*this->pclCurrentFrame, *this->EdgesPoints);
    CopyPointCloudMetadata(*this->pclCurrentFrame, *this->PlanarsPoints);
    CopyPointCloudMetadata(*this->pclCurrentFrame, *this->BlobsPoints);
    CopyPointCloudMetadata(*this->pclCurrentFrame, *this->RawPoints);

    this->Angles.resize(this->NLasers);
    this->Saliency.resize(this->NLasers);
    this->DepthGap.resize(this->NLasers);
    this->IntensityGap.resize(this->NLasers);
    this->IsPointValid.resize(this->NLasers);
    this->Label.resize(this->NLasers);
}
//-----------------------------------------------------------------------------

void LidarKeypointExtractor::ConvertAndSortScanLines() {
    size_t nbPoints = this->pclCurrentFrame->size();
    double frameStartTime = this->pclCurrentFrame->points[0].time;
    double framsDutation = this->pclCurrentFrame->points[nbPoints - 1].time;

    for (size_t index = 0; index < nbPoints; ++index) {
        const Point &oldPoint = this->pclCurrentFrame->points[index];
        Point newPoint(oldPoint);

        int id = oldPoint.laserId;
#if 0
        id = this->LaserIdMapping[oldPoint.laserId];
        // modity the point so that:
        // -laserId is corrected with the laserIdMapping
        // -time become a relative advancement time (between 0 and 1)
        newPoint.laserId = static_cast<int>(id);
#endif
        newPoint.time = (oldPoint.time - frameStartTime) / framsDutation;

        // add the current point to its correspanding laser scan
        this->pclCurrentFrameByScan[id]->push_back(newPoint);
    }
}

void LidarKeypointExtractor::ComputeKeyPoints(pcl::PointCloud<Point>::Ptr pc, int N_scans, bool dynamic_mask) {
    //  if (this->LaserIdMapping.empty())
    //  {
    //    this->NLasers = laserIdMapping.size();
    //    this->LaserIdMapping = laserIdMapping;
    //  }
    this->NLasers = N_scans;
    this->pclCurrentFrame = pc;
    //  LOG(INFO) << "PrepareDateForNextFrame begin";
    this->PrepareDateForNextFrame();
    //  LOG(INFO) << "PrepareDateForNextFrame end";

    //  LOG(INFO) << "ConvertAndSortScanLines begin";
    this->ConvertAndSortScanLines();
    //  LOG(INFO) << "ConvertAndSortScanLines end";

    //  LOG(INFO) << "Initialize the vectors with the correct length begin";
    // Initialize the vectors with the correct length

    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t scanLine = range.begin(); scanLine != range.end(); ++scanLine) {
            size_t nbPoint = this->pclCurrentFrameByScan[scanLine]->size();
            this->IsPointValid[scanLine].assign(nbPoint, KeypointFlags().set()); // set all flags to 1
            this->Label[scanLine].assign(nbPoint, KeypointFlags().reset());      // set all flags to 0  //TODO:edge point 本来就是０,冲突？
            this->Angles[scanLine].assign(nbPoint, 0.);
            this->Saliency[scanLine].assign(nbPoint, 0.);
            this->DepthGap[scanLine].assign(nbPoint, 0.);
            this->IntensityGap[scanLine].assign(nbPoint, 0.);
        }
    };


    tbb::blocked_range<size_t> range(0, this->NLasers);
    tbb::parallel_for(range, compute_func);

    //  LOG(INFO) << "Initialize the vectors with the correct length end";

    //  LOG(INFO) << "InvalidPointWithBadCriteria begin";
    // Invalid points with bad criteria

    this->InvalidPointWithBadCriteria(dynamic_mask);
    
    //  LOG(INFO) << "InvalidPointWithBadCriteria end";

    //  LOG(INFO) << "ComputeCurvature begin";
    // compute keypoints scores
    this->ComputeCurvature();
    //  LOG(INFO) << "ComputeCurvature end";

    //  LOG(INFO) << "SetKeyPointsLabels begin";
    // labelize keypoints
    this->SetKeyPointsLabels();
    //  LOG(INFO) << "SetKeyPointsLabels end";
}

void LidarKeypointExtractor::ComputeCurvature() {

    const double squaredDistToLineThreshold =
            this->DistToLineThreshold * this->DistToLineThreshold; // [m²]
    const double squaredDepthDistCoeff = 0.25;
    const double minDepthGapDist = 1.5; // [m]


    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t scanLine = range.begin(); scanLine != range.end(); ++scanLine) {

            Eigen::aligned_vector<Eigen::Vector3d> leftNeighbors(this->NeighborWidth);
            Eigen::aligned_vector<Eigen::Vector3d> rightNeighbors(this->NeighborWidth);
            Eigen::aligned_vector<Eigen::Vector3d> farNeighbors;
            farNeighbors.reserve(2 * this->NeighborWidth);

            LineFitting leftLine, rightLine, farNeighborsLine;

            const int Npts = this->pclCurrentFrameByScan[scanLine]->size();

            if (this->IsScanLineAlmostEmpty(Npts)) {
                continue;
            }

            for (int index = this->NeighborWidth; (index + this->NeighborWidth) < Npts; ++index) {

                if (this->IsPointValid[scanLine][index].none()) {
                    continue;
                }


                const Point &currentPoint = this->pclCurrentFrameByScan[scanLine]->points[index];
                const Eigen::Vector3d centralPoint(currentPoint.x, currentPoint.y, currentPoint.z);

                const Point &previousPoint = this->pclCurrentFrameByScan[scanLine]->points[index - 1];
                const Point &nextPoint = this->pclCurrentFrameByScan[scanLine]->points[index + 1];
                this->IntensityGap[scanLine][index] =
                        std::abs(nextPoint.intensity - previousPoint.intensity);


                for (int j = index - 1; j >= index - this->NeighborWidth; --j) {
                    const Point &point = this->pclCurrentFrameByScan[scanLine]->points[j];
                    leftNeighbors[index - 1 - j] << point.x, point.y, point.z;
                }
                for (int j = index + 1; j <= index + this->NeighborWidth; ++j) {
                    const Point &point = this->pclCurrentFrameByScan[scanLine]->points[j];
                    rightNeighbors[j - index - 1] << point.x, point.y, point.z;
                }


                const bool leftFlat = leftLine.FitPCAAndCheckConsistency(leftNeighbors);
                const bool rightFlat = rightLine.FitPCAAndCheckConsistency(rightNeighbors);

                double distLeft = 0., distRight = 0.;


                if (leftFlat && rightFlat) {

                    distLeft = leftLine.SquaredDistanceToPoint(centralPoint);
                    distRight = rightLine.SquaredDistanceToPoint(centralPoint);


                    if ((distLeft < squaredDistToLineThreshold) && (distRight < squaredDistToLineThreshold))
                        this->Angles[scanLine][index] = (leftLine.Direction.cross(rightLine.Direction)).norm();
                }


                else if (!leftFlat && rightFlat) {
                    distLeft = std::numeric_limits<double>::max();
                    for (const Eigen::Vector3d &leftNeighbor : leftNeighbors) {
                        distLeft = std::min(distLeft, rightLine.SquaredDistanceToPoint(leftNeighbor));
                    }
                    distLeft *= squaredDepthDistCoeff;
                } else if (leftFlat && !rightFlat) {
                    distRight = std::numeric_limits<double>::max();
                    for (const Eigen::Vector3d &rightNeighbor : rightNeighbors) {
                        distRight = std::min(distRight, leftLine.SquaredDistanceToPoint(rightNeighbor));
                    }
                    distRight *= squaredDepthDistCoeff;
                }


                else {

                    const double currDepth = centralPoint.norm();
                    bool hasLeftEncounteredDepthGap = false;
                    bool hasRightEncounteredDepthGap = false;
                    farNeighbors.clear();

                    for (const Eigen::Vector3d &leftNeighbor : leftNeighbors) {

                        if (std::abs(leftNeighbor.norm() - currDepth) > minDepthGapDist) {
                            hasLeftEncounteredDepthGap = true;
                            farNeighbors.emplace_back(leftNeighbor);
                        } else if (hasLeftEncounteredDepthGap)
                            break;
                    }
                    for (const Eigen::Vector3d &rightNeighbor : rightNeighbors) {

                        if (std::abs(rightNeighbor.norm() - currDepth) > minDepthGapDist) {
                            hasRightEncounteredDepthGap = true;
                            farNeighbors.emplace_back(rightNeighbor);
                        } else if (hasRightEncounteredDepthGap)
                            break;
                    }

                    if (farNeighbors.size() > static_cast<unsigned int>(this->NeighborWidth)) {
                        farNeighborsLine.FitPCA(farNeighbors);
                        this->Saliency[scanLine][index] = farNeighborsLine.SquaredDistanceToPoint(centralPoint);
                    }
                }


                this->DepthGap[scanLine][index] = std::max(distLeft, distRight);
            }
        }
    };
    tbb::blocked_range<size_t> range(0, this->NLasers);
    tbb::parallel_for(range, compute_func);
} // function LidarKeypointExtractor::ComputeCurvature end

//-----------------------------------------------------------------------------

void LidarKeypointExtractor::InvalidPointWithBadCriteria( bool dynamic_mask) {
    const double expectedCoeff = 10.;

    // loop over scan lines
    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {

        for (size_t scanLine = range.begin(); scanLine != range.end(); ++scanLine) {
            const int Npts = this->pclCurrentFrameByScan[scanLine]->size();

            if (this->IsScanLineAlmostEmpty(Npts)) {
                for (int index = 0; index < Npts; ++index)
                    this->IsPointValid[scanLine][index].reset();
                continue;
            }


            for (int index = 0; index <= this->NeighborWidth; ++index)
                this->IsPointValid[scanLine][index].reset();
            for (int index = Npts - 1 - this->NeighborWidth - 1; index < Npts; ++index)
                this->IsPointValid[scanLine][index].reset();


            for (int index = this->NeighborWidth; index < Npts - this->NeighborWidth - 1; ++index) {
                const Point &previousPoint = this->pclCurrentFrameByScan[scanLine]->points[index - 1];
                const Point &currentPoint = this->pclCurrentFrameByScan[scanLine]->points[index];
                const Point &nextPoint = this->pclCurrentFrameByScan[scanLine]->points[index + 1];


                const Eigen::Vector3f &Xp = previousPoint.getVector3fMap();
                const Eigen::Vector3f &X = currentPoint.getVector3fMap();
                const Eigen::Vector3f &Xn = nextPoint.getVector3fMap();

                const double L = X.norm();
                const double Ln = Xn.norm();
                const double dLn = (Xn - X).norm();
                const double dLp = (X - Xp).norm();

                const double expectedLength = this->AngleResolution * L;
                

                double MAX_Z = 0.4;
                double MIN_Z = -0.8;
                double MAX_ANGLE = M_PI / 12;
                double MAX_X = 10;

                double angle = atan2(currentPoint.y, currentPoint.x);
                
                if (dynamic_mask==true)
               { 
                if (currentPoint.z < MAX_Z && currentPoint.z > MIN_Z && currentPoint.x < MAX_X && std::abs(angle) < MAX_ANGLE)
                {
                    this->IsPointValid[scanLine][index].reset();
                }

                if (currentPoint.z < MAX_Z && currentPoint.z > MIN_Z && currentPoint.x > -MAX_X && (angle < - M_PI + MAX_ANGLE || angle > M_PI - MAX_ANGLE))
                {
                    this->IsPointValid[scanLine][index].reset();
                }
               }
                if (dLn > expectedCoeff * expectedLength) {

                    if (L < Ln) {
                        this->IsPointValid[scanLine][index + 1].reset();
                        for (int i = index + 2; i <= index + this->NeighborWidth; ++i) {
                            const Eigen::Vector3f &Y =
                                    this->pclCurrentFrameByScan[scanLine]->points[i - 1].getVector3fMap();
                            const Eigen::Vector3f &Yn =
                                    this->pclCurrentFrameByScan[scanLine]->points[i].getVector3fMap();

                            if ((Yn - Y).norm() > expectedCoeff * expectedLength) {
                                break;
                            }

                            this->IsPointValid[scanLine][i].reset();
                        }
                    }

                    else {
                        this->IsPointValid[scanLine][index].reset();
                        for (int i = index - this->NeighborWidth; i < index; ++i) {
                            const Eigen::Vector3f &Yp =
                                    this->pclCurrentFrameByScan[scanLine]->points[i].getVector3fMap();
                            const Eigen::Vector3f &Y =
                                    this->pclCurrentFrameByScan[scanLine]->points[i + 1].getVector3fMap();


                            if ((Y - Yp).norm() > expectedCoeff * expectedLength) {
                                break;
                            }

                            this->IsPointValid[scanLine][i].reset();
                        }
                    }
                }


                if (L < this->MinDistanceToSensor) {
                    this->IsPointValid[scanLine][index].reset();
                }

                if (L > this->MaxDistanceToSensor) {
                    this->IsPointValid[scanLine][index].reset();
                }


                else if ((dLp > 0.25 * expectedCoeff * expectedLength) &&
                         (dLn > 0.25 * expectedCoeff * expectedLength)) {
                    this->IsPointValid[scanLine][index].reset();
                }
            }
        }
    }; // function cunpute_func
    tbb::blocked_range<size_t> range(0, this->NLasers);
    tbb::parallel_for(range, compute_func);
} // function InvalidPointWithBadCriteria

//-----------------------------------------------------------------------------
void LidarKeypointExtractor::SetKeyPointsLabels() {

    const double squaredEdgeSaliencythreshold =
            this->EdgeSaliencyThreshold * this->EdgeSaliencyThreshold;
    const double squaredEdgeDepthGapThreshold =
            this->EdgeDepthGapThreshold * this->EdgeDepthGapThreshold;

    // loop over the scan lines
    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t scanLine = range.begin(); scanLine != range.end(); ++scanLine) {
            const int Npts = this->pclCurrentFrameByScan[scanLine]->size();


            if (this->IsScanLineAlmostEmpty(Npts)) {
                continue;
            }


            std::vector<size_t> sortedDepthGapIdx = SortIdx(this->DepthGap[scanLine]);
            std::vector<size_t> sortedAnglesIdx = SortIdx(this->Angles[scanLine]);
            std::vector<size_t> sortedSaliencyIdx = SortIdx(this->Saliency[scanLine]);
            std::vector<size_t> sortedIntensityGap = SortIdx(this->IntensityGap[scanLine]);


            auto addEdgesUsingCriterion = [this, scanLine, Npts](    //TODO:how to use this?
                    const std::vector<size_t> &sortedValuesIdx,
                    const std::vector<std::vector<double> > &values,
                    double threshold, int invalidNeighborhoodSize) {
                for (const auto &index : sortedValuesIdx) {

                    if (values[scanLine][index] < threshold)
                        break;


                    if (!this->IsPointValid[scanLine][index][Keypoint::EDGE]) //TODO:Edge 是０，最后赋值还是０，会不会冲突？
                        continue;


                    this->Label[scanLine][index].set(Keypoint::EDGE);


                    const int indexBegin = std::max(0, static_cast<int>(index - invalidNeighborhoodSize));
                    const int indexEnd =
                            std::min(Npts - 1, static_cast<int>(index + invalidNeighborhoodSize));
                    for (int j = indexBegin; j <= indexEnd; ++j)
                        this->IsPointValid[scanLine][j].reset(Keypoint::EDGE);
                }
            };

            // Edges using depth gap
            addEdgesUsingCriterion(
                    sortedDepthGapIdx, this->DepthGap, squaredEdgeDepthGapThreshold, this->NeighborWidth - 1);
            // Edges using angles
            addEdgesUsingCriterion(
                    sortedAnglesIdx, this->Angles, this->EdgeSinAngleThreshold, this->NeighborWidth);
            // Edges using saliency
            addEdgesUsingCriterion(
                    sortedSaliencyIdx, this->Saliency, squaredEdgeSaliencythreshold, this->NeighborWidth - 1);
            // Edges using intensity
            addEdgesUsingCriterion(
                    sortedIntensityGap, this->IntensityGap, this->EdgeIntensityGapThreshold, 1);

            // Planes (using angles)
            for (int k = Npts - 1; k >= 0; --k) {
                size_t index = sortedAnglesIdx[k];
                const double sinAngle = this->Angles[scanLine][index];

                // thresh
                if (sinAngle > this->PlaneSinAngleThreshold)
                    break;


                if (!this->IsPointValid[scanLine][index][Keypoint::PLANE])
                    continue;


                this->Label[scanLine][index].set(Keypoint::PLANE);


                const int indexBegin = std::max(0, static_cast<int>(index - 4));
                const int indexEnd = std::min(Npts - 1, static_cast<int>(index + 4));
                for (int j = indexBegin; j <= indexEnd; ++j)
                    this->IsPointValid[scanLine][j].reset(Keypoint::PLANE);
            }


            for (int index = 0; index < Npts; index += 3) {
                if (this->IsPointValid[scanLine][index][Keypoint::BLOB])
                    this->Label[scanLine][index].set(Keypoint::BLOB);
            }
        }
    };

    tbb::blocked_range<size_t> range(0, this->NLasers);
    tbb::parallel_for(range, compute_func);

    auto addKeypoints = [this](Keypoint type, pcl::PointCloud<Point>::Ptr &keypoints) {
        for (unsigned int scanLine = 0; scanLine < this->NLasers; ++scanLine) {
            for (unsigned int index = 0; index < this->pclCurrentFrameByScan[scanLine]->size(); ++index) {
                if (this->Label[scanLine][index][type]) {
                    this->IsPointValid[scanLine][index].set(type);
                    const Point &p = this->pclCurrentFrameByScan[scanLine]->points[index];
                    keypoints->push_back(p);
                }
            }
        }
    };
    addKeypoints(Keypoint::EDGE, this->EdgesPoints);
    addKeypoints(Keypoint::PLANE, this->PlanarsPoints);
    addKeypoints(Keypoint::BLOB, this->PlanarsPoints);

    auto addAllPoints = [this](pcl::PointCloud<pcl::PointXYZHSV>::Ptr &raw) {
        pcl::PointXYZHSV point;
        for (unsigned int scanLine = 0; scanLine < this->NLasers; ++scanLine) {
            for (unsigned int index = 0; index < this->pclCurrentFrameByScan[scanLine]->size(); ++index) {
                const Point &p = this->pclCurrentFrameByScan[scanLine]->points[index];
                point.x = p.x;
                point.y = p.y;
                point.z = p.z;
                point.h = p.intensity;
                point.s = this->Saliency[scanLine][index];
                if (this->Label[scanLine][index][Keypoint::EDGE])
                    point.v = Keypoint::EDGE;
                if (this->Label[scanLine][index][Keypoint::PLANE])
                    point.v = Keypoint::PLANE;
                if (this->Label[scanLine][index][Keypoint::BLOB])
                    point.v = Keypoint::BLOB;
                
                raw->push_back(point);
            }
        }
    };

    addAllPoints(this->RawPoints);
}
