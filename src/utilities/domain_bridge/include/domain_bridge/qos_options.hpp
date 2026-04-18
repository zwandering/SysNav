// Copyright 2021, Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DOMAIN_BRIDGE__QOS_OPTIONS_HPP_
#define DOMAIN_BRIDGE__QOS_OPTIONS_HPP_

#include <optional>

#include "rclcpp/qos.hpp"

#include "domain_bridge/visibility_control.hpp"

namespace domain_bridge
{

/// Options for overriding QoS settings.
class QosOptions
{
public:
  /// Constructor.
  /**
   *  Default values:
   *
   *    - reliability = nullopt_t (detect automatically)
   *    - durability = nullopt_t (detect automatically)
   *    - history = rclcpp::HistoryPolicy::KeepLast
   *    - depth = 10
   *    - deadline = 0 (RMW default)
   *    - lifespan = 0 (RMW default)
   */
  DOMAIN_BRIDGE_PUBLIC
  QosOptions() = default;

  /// Get reliability.
  DOMAIN_BRIDGE_PUBLIC
  std::optional<rclcpp::ReliabilityPolicy>
  reliability() const;

  /// Set reliability.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  reliability(const rclcpp::ReliabilityPolicy & reliability);

  /// Get durability.
  DOMAIN_BRIDGE_PUBLIC
  std::optional<rclcpp::DurabilityPolicy>
  durability() const;

  /// Set durability.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  durability(const rclcpp::DurabilityPolicy & durability);

  /// Get history.
  DOMAIN_BRIDGE_PUBLIC
  rclcpp::HistoryPolicy
  history() const;

  /// Set history.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  history(const rclcpp::HistoryPolicy & history);

  /// Get history depth.
  DOMAIN_BRIDGE_PUBLIC
  size_t
  depth() const;

  /// Set history depth.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  depth(const size_t & depth);

  /// Get deadline in nanoseconds.
  /**
   * If nullopt_t is returned, then use automatic matching.
   */
  DOMAIN_BRIDGE_PUBLIC
  std::optional<int64_t>
  deadline() const;

  /// Set deadline.
  /**
   * \param deadline: number of nanoseconds.
   *   If zero, then use the RMW default policy.
   *   If negative, then it is treated an "infinite" (max number of nanoseconds).
   */
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  deadline(int64_t deadline);

  /// Set deadline policy to automatically match available publishers.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  deadline_auto();

  /// Get lifespan in nanoseconds.
  /**
   * If nullopt_t is returned, then use automatic matching.
   */
  DOMAIN_BRIDGE_PUBLIC
  std::optional<int64_t>
  lifespan() const;

  /// Set lifespan in nanoseconds.
  /**
   * \param lifespan: number of nanoseconds.
   *   If zero, then use the RMW default policy.
   *   If negative, then it is treated an "infinite" (max number of nanoseconds).
   */
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  lifespan(int64_t lifespan);

  /// Set lifespan policy to automatically match available publihsers.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  lifespan_auto();

  /// Get liveliness.
  DOMAIN_BRIDGE_PUBLIC
  std::optional<rclcpp::LivelinessPolicy>
  liveliness() const;

  /// Set history.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  liveliness(const rclcpp::LivelinessPolicy & liveliness);

  /// Set liveliness policy to automatically match available publishers.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  liveliness_auto();

  /// Get liveliness.
  DOMAIN_BRIDGE_PUBLIC
  std::optional<int64_t>
  liveliness_lease_duration() const;

  /// Set history.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions &
  liveliness_lease_duration(int64_t liveliness_lease_duration);

private:
  std::optional<rclcpp::ReliabilityPolicy> reliability_;
  std::optional<rclcpp::DurabilityPolicy> durability_;
  rclcpp::HistoryPolicy history_{rclcpp::HistoryPolicy::KeepLast};
  size_t depth_{10};
  std::optional<int64_t> deadline_{0};
  std::optional<int64_t> lifespan_{0};
  std::optional<rclcpp::LivelinessPolicy> liveliness_;
  std::optional<int64_t> liveliness_lease_duration_{0};
};  // class QosOptions

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__QOS_OPTIONS_HPP_
