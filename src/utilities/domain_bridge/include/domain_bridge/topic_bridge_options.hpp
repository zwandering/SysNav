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

#ifndef DOMAIN_BRIDGE__TOPIC_BRIDGE_OPTIONS_HPP_
#define DOMAIN_BRIDGE__TOPIC_BRIDGE_OPTIONS_HPP_

#include <memory>
#include <string>

#include "rclcpp/callback_group.hpp"

#include "domain_bridge/qos_options.hpp"
#include "domain_bridge/visibility_control.hpp"

namespace domain_bridge
{

/// Options for bridged topics.
class TopicBridgeOptions
{
public:
  /// Constructor.
  /**
   *  Default values:
   *
   *    - callback_group = nullptr (node's default)
   *    - qos_options = default (see QosOptions for more information)
   *    - remap_name = "" (no remap)
   *    - bidirectional = false (one-way)
   *    - reversed = false
   *    - wait_for_subscription = false
   *    - wait_for_publisher = true
   *    - auto_remove = AutoRemove::Disabled
   *    - delay = 0 (no extra delay to wait for publishers before creating bridge)
   */
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions() = default;

  /// Get callback group associated with the topics pub/sub pair.
  DOMAIN_BRIDGE_PUBLIC
  std::shared_ptr<rclcpp::CallbackGroup>
  callback_group() const;

  /// Set the callback group associated with the topics pub/sub pair.
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  callback_group(std::shared_ptr<rclcpp::CallbackGroup> group);

  /// Get QoS options.
  DOMAIN_BRIDGE_PUBLIC
  QosOptions
  qos_options() const;

  /// Set QoS options.
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  qos_options(const QosOptions & qos_options);

  /// Get remap name.
  DOMAIN_BRIDGE_PUBLIC
  const std::string &
  remap_name() const;

  /// Set remap name.
  /**
   * An empty string implies no remap.
   */
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  remap_name(const std::string & remap_name);

  /// Get bidirectional option.
  DOMAIN_BRIDGE_PUBLIC
  const bool &
  bidirectional() const;

  /// Set bidirectional option. If true, this will bridge the topic in both directions.
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  bidirectional(const bool & bidirectional);

  /// Get reversed option.
  DOMAIN_BRIDGE_PUBLIC
  const bool &
  reversed() const;

  /// Set reversed option. If true, this will swap the 'to' and 'from' domain IDs.
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  reversed(const bool & reversed);

  /// Get bridge delay.
  DOMAIN_BRIDGE_PUBLIC
  const std::chrono::milliseconds &
  delay() const;

  /// Set bridge delay (amount of time to wait for publishers before creating bridge).
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  delay(const std::chrono::milliseconds & delay);

  /// Set wait_for_subscription to `value`.
  /**
   * When set to true, the domain bridge will wait for a subscription to be available in the
   * "to domain" before creating the bridge.
   *
   * If wait_for_publisher() is also true, the bridge will first wait for an available publisher and
   * then for an available subscription.
   * QoS matching will be taken from the publisher in that case.
   */
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  wait_for_subscription(bool value);

  /// Get wait_for_subscription option.
  DOMAIN_BRIDGE_PUBLIC
  bool
  wait_for_subscription() const;

  /// Set wait_for_publisher to value.
  /**
   * When set to true, the domain bridge will wait for a publisher to be available in the
   * "from domain" before creating the bridge.
   */
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  wait_for_publisher(bool value);

  /// Get wait_for_publisher option.
  DOMAIN_BRIDGE_PUBLIC
  bool
  wait_for_publisher() const;

  enum class AutoRemove
  {
    OnNoPublisher,
    OnNoSubscription,
    OnNoPublisherOrSubscription,
    Disabled,
  };

  /// Set auto_remove to value.
  /**
   * If set to AutoRemove::OnNoPublisher the domain bridge will stop bridging when no publisher is
   * available in the "from domain".
   * The bridge will automatically be created again when a new publisher is discovered.
   *
   * If set to AutoRemove::OnNoSubscription the domain bridge will stop bridging when no
   * subscription is available in the "to domain".
   * The bridge will automatically be created again when a new subscription is discovered.
   *
   * When set to AutoRemove::Disabled, the bridge will keep running forever once created.
   */
  DOMAIN_BRIDGE_PUBLIC
  TopicBridgeOptions &
  auto_remove(AutoRemove value);

  /// Get remove_if_no_publisher option.
  DOMAIN_BRIDGE_PUBLIC
  AutoRemove
  auto_remove() const;

private:
  std::shared_ptr<rclcpp::CallbackGroup> callback_group_{nullptr};

  QosOptions qos_options_;

  std::string remap_name_;

  bool bidirectional_{false};

  bool reversed_{false};

  bool wait_for_subscription_{false};

  bool wait_for_publisher_{true};

  AutoRemove auto_remove_{AutoRemove::Disabled};

  std::chrono::milliseconds delay_{0};
};  // class TopicBridgeOptions

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__TOPIC_BRIDGE_OPTIONS_HPP_
