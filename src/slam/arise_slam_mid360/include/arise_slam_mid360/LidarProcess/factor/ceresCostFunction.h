//
// Created by ubuntu on 2020/9/25.
//

#ifndef CERESCOSTFUNCTION_H
#define CERESCOSTFUNCTION_H

#include "arise_slam_mid360/LidarProcess/MotionModel.h"

struct MahalanobisDistanceInterpolatedMotionResidual
{
public:
  MahalanobisDistanceInterpolatedMotionResidual(const Eigen::Matrix3d& argA,
    const Eigen::Vector3d& argC, const Eigen::Vector3d& argX, double argTime, double argWeight)
    : A(argA)
    , C(argC)
    , X(argX)
    , Time(argTime)
    , Weight(argWeight)
  {
  }

  template<typename T>
  bool operator()(const T* const w0, const T* const w1, T* residual) const
  {

    using Vector3T = Eigen::Matrix<T, 3, 1>;
    using TransformT = Twist<T>;

    static TransformT T0, T1;
    static LinearTransformInterpolator<T> transformInterpolator;
    static T lastw0[7] = { T(-1.), T(-1.), T(-1.), T(-1.), T(-1.), T(-1.), T(-1.) };
    static T lastw1[7] = { T(-1.), T(-1.), T(-1.), T(-1.), T(-1.), T(-1.), T(-1.) };

    if (!std::equal(w0, w0 + 7, lastw0))
    {
      T0.pos << w0[0], w0[1], w0[2];
      T0.rot.x() = w0[3];
      T0.rot.y() = w0[4];
      T0.rot.z() = w0[5];
      T0.rot.w() = w0[6];

      transformInterpolator.SetT0(T0);
      std::copy(w0, w0 + 7, lastw0);
    }

    if (!std::equal(w1, w1 + 7, lastw0))
    {
      T0.pos << w1[0], w1[1], w1[2];
      T0.rot.x() = w1[3];
      T0.rot.y() = w1[4];
      T0.rot.z() = w1[5];
      T0.rot.w() = w1[6];

      transformInterpolator.SetT1(T1);
      std::copy(w1, w1 + 7, lastw1);
    }

    // Compute the transform to apply to X depending on (R0, T0) and (R1, T1).
    // The applied isometry will be the linear interpolation between them :
    // (R, T) = (R0^(1-t) * R1^t, (1 - t)T0 + tT1)

    const TransformT T_0_cur = transformInterpolator(T(this->Time));

    const Vector3T Y = T_0_cur.rot * this->X + T_0_cur.pos - this->C;

    const T squareResidual = this->Weight * (Y.transpose() * this->A * Y)(0);

    residual[0] = squareResidual < 1e-6 ? T(0) : ceres::sqrt(squareResidual);

    return true;
  }

private:
  const Eigen::Matrix3d A;
  const Eigen::Vector3d C;
  const Eigen::Vector3d X;
  const double Time;
  const double Weight;
};

struct MahalanobisDistanceAffineIsometryResidual
{
public:
    MahalanobisDistanceAffineIsometryResidual(const Eigen::Matrix3d& argA,
                                                   const Eigen::Vector3d& argC, const Eigen::Vector3d& argX, double argWeight)
    : A(argA)
    , C(argC)
    , X(argX)
    , Weight(argWeight)
  {
  }

  template<typename T>
  bool operator()( const T* w, T* residual) const
  {

    //using Matrix3T = Eigen::Matrix<T, 3, 3>;
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    //using QuaternionT = Eigen::Quaternion<T>;

    //Eigen::Map<const Vector3T> trans(w);
  //  Eigen::Map<const QuaternionT> rot(w + 3);

    Eigen::Quaternion<T> rot{w[3], w[0], w[1], w[2]};

    Eigen::Matrix<T, 3, 1> trans{w[4],w[5],w[6]};

    Vector3T Y = rot * X.cast<T>() + trans - C.cast<T>();

    T squaredResidual = T(Weight) * (Y.transpose() * A.cast<T>() * Y)(0);

    residual[0] = squaredResidual < 1e-6 ? T(0) : ceres::sqrt(squaredResidual);

    return true;
  }

private:
  const Eigen::Matrix3d A;
  const Eigen::Vector3d C;
  const Eigen::Vector3d X;
  const double Weight;
};






#endif // CERESCOSTFUNCTION_H
