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

#ifndef DOMAIN_BRIDGE__SERVICE_BRIDGE_OPTIONS_HPP_
#define DOMAIN_BRIDGE__SERVICE_BRIDGE_OPTIONS_HPP_

#include <memory>
#include <string>

#include "rclcpp/callback_group.hpp"

#include "domain_bridge/visibility_control.hpp"

namespace domain_bridge
{

/// Options for bridged services.
class ServiceBridgeOptions
{
public:
  /// Constructor.
  /**
   *  Default values:
   *
   *    - callback_group = nullptr (node's default)
   *    - remap_name = "" (no remap)
   */
  DOMAIN_BRIDGE_PUBLIC
  ServiceBridgeOptions() = default;

  /// Assignment operator.
  DOMAIN_BRIDGE_PUBLIC
  ServiceBridgeOptions &
  operator=(const ServiceBridgeOptions & other) = default;

  /// Get callback group associated with the services pub/sub pair.
  DOMAIN_BRIDGE_PUBLIC
  std::shared_ptr<rclcpp::CallbackGroup>
  callback_group() const;

  /// Set the callback group associated with the services pub/sub pair.
  DOMAIN_BRIDGE_PUBLIC
  ServiceBridgeOptions &
  callback_group(std::shared_ptr<rclcpp::CallbackGroup> group);

  /// Get remap name.
  DOMAIN_BRIDGE_PUBLIC
  const std::string &
  remap_name() const;

  /// Set remap name.
  /**
   * An empty string implies no remap.
   */
  DOMAIN_BRIDGE_PUBLIC
  ServiceBridgeOptions &
  remap_name(const std::string & remap_name);

private:
  std::shared_ptr<rclcpp::CallbackGroup> callback_group_{nullptr};
  std::string remap_name_;
};  // class ServiceBridgeOptions

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__SERVICE_BRIDGE_OPTIONS_HPP_
