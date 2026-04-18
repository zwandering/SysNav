//
// Created by ubuntu on 2020/9/21.
//

#pragma once

// Eigen
#include <Eigen/Dense>

// CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// internal
#include "arise_slam_mid360/utils/Twist.h"

/**
 * \struct LinearTransformInterpolator
 * \brief Linear interpolator to estimate an intermediate transform between two isometries.
 *
 * At t=0, the first isometry is returned, at t=1 the second.
 * The translation will be interpolated linearly and the rotation spherical linearly.
 */

template<typename T>
struct LinearTransformInterpolator
{
  // Useful types
  using Vector3T = Eigen::Matrix<T, 3, 1>;
  using QuaternionT = Eigen::Quaternion<T>;
  using Translation3T = Eigen::Translation<T, 3>;
  using TransformT = Twist<T>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LinearTransformInterpolator() = default;

  LinearTransformInterpolator(const TransformT& T0, const TransformT& T1)
    : Rot0(T0.rot)
    , Rot1(T1.rot)
    , Trans0(T0.pos)
    , Trans1(T1.pos)
  {
  }

  // Set transforms
  void SetTransforms(const TransformT& T0, const TransformT& T1)
  {
    SetT0(T0);
    SetT1(T1);
  }
  void SetT0(const TransformT& T0)
  {
    Rot0 = T0.rot;
    Trans0 = T0.pos;
  }
  void SetT1(const TransformT& T1)
  {
    Rot1 = T1.rot;
    Trans1 = T1.pos;
  }

  // Return the affine isometry linearly interpolated at the requested time between H0 (t=0) and H1
  // (t=1).
  TransformT operator()(T t) const
  {
    // Rotation part : spherical interpolation,   Translation part : linear interpolation
    return TransformT(Rot0.slerp(t, Rot1), Trans0 + t * (Trans1 - Trans0));
  }

private:
  QuaternionT Rot0, Rot1;
  Vector3T Trans0, Trans1;
};

/**
 * \brief Interpolate spherical linearly between two isometries.
 *
 * At t=t0, the first isometry is returned, at t=t1 the second.
 * The translation will be interpolated linearly and the rotation spherical linearly.
 * @return
 */
// TODO:
inline Transformd LinearInterpolation(
  const Transformd& T0, const Transformd& T1, double t, double t0 = 0., double t1 = 1.0)
{
  const double time = (t - t0) / (t1 - t0);
  Eigen::Quaterniond rot(T0.rot.slerp(time, T1.rot));
  Eigen::Vector3d trans(T0.pos + time * (T1.pos));

  return Transformd(rot, trans);
}

/**
* \class LinearTransformInterpolation
* \brief Perform the linear interpolation between
*        two isometric affine transforms. The rotational
*        part is interpolated using a SLERP and the translation
*        part is linearly interpolated. The function is templated
*        to be usable with dual number and autodifferentiation
*/
//-----------------------------------------------------------------------------
template <typename T>
inline Eigen::Matrix<T, 4, 4> LinearTransformInterpolation(const Eigen::Matrix<T, 3, 3>& R0, const Eigen::Matrix<T, 3, 1>& T0,
                                                    const Eigen::Matrix<T, 3, 3>& R1, const Eigen::Matrix<T, 3, 1>& T1,
                                                    T s)
{
    // Linearly interpolate the translation part
    Eigen::Matrix<T, 3, 1> Tc = (T(1.0) - s) * T0 + s * T1;

    // SLERP interpolation of the rotational part
    T r0[9], r1[9];
    T angle_axis_r0[3], angle_axis_r1[3];

    // column major
    for (unsigned int j = 0; j < 3; ++j)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
            r0[j + 3 * i] = R0(i, j);
            r1[j + 3 * i] = R1(i, j);
        }
    }

    T q0[4], q1[4], q[4];
    // Rotation matrix to quaternions
    ceres::RotationMatrixToAngleAxis(r0, angle_axis_r0);
    ceres::RotationMatrixToAngleAxis(r1, angle_axis_r1);
    ceres::AngleAxisToQuaternion(angle_axis_r0, q0);
    ceres::AngleAxisToQuaternion(angle_axis_r1, q1);

    // Canonical scalar product on quaternion
    T dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];

    // To prevent the SLERP interpolation to take the long path
    // we first check their relative orientation. If the angle
    // is superior to 90 we take the opposite quaternion which
    // is closer and represents the same rotation
    if(dot < T(0.0))
    {
        dot = -dot;
        for (unsigned int k = 0; k < 4; ++k)
        {
            q1[k] = T(-1.0) * q1[k];
        }
    }

    // To avoid division by zero, perform a linear interpolation (LERP), if our
    // quarternions are nearly in the same direction, otherwise resort
    // to spherical linear interpolation. In the limiting case (for small
    // angles), SLERP is equivalent to LERP.
    T t1, t2;
    if ((T(1.0) - ceres::abs(dot)) < T(1e-6))
    {
        t1 = T(1.0) - s;
        t2 = s;
    }
    else
    {
        // Angle (defined by the canonical scalar product for quaternions)
        // between the two quaternions
        const T theta = ceres::acos(dot);
        t1 = ceres::sin((T(1.0) - s) * theta) / ceres::sin(theta);
        t2 = ceres::sin(s * theta) / ceres::sin(theta);
    }
    for (unsigned int k = 0; k < 4; ++k)
    {
        q[k] = t1 * q0[k] + t2 * q1[k];
    }
    T r[9];
    ceres::QuaternionToRotation(q, r);

    Eigen::Matrix<T, 4, 4> H = Eigen::Matrix<T, 4, 4>::Zero();
    // column major
    for (unsigned int j = 0; j < 3; ++j)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
            H(i, j) = r[i + 3 * j];
        }
        H(j, 3) = Tc(j);
    }
    H(3, 3) = T(1);
    return H;
}