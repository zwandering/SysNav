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

#ifndef DOMAIN_BRIDGE__DOMAIN_BRIDGE_CONFIG_HPP_
#define DOMAIN_BRIDGE__DOMAIN_BRIDGE_CONFIG_HPP_

#include <utility>
#include <vector>

#include "domain_bridge/domain_bridge_options.hpp"
#include "domain_bridge/topic_bridge.hpp"
#include "domain_bridge/topic_bridge_options.hpp"

namespace domain_bridge
{

/// A configuration for a DomainBridge
struct DomainBridgeConfig
{
  /// Options
  DomainBridgeOptions options;

  /// Topic bridges with options
  std::vector<std::pair<TopicBridge, TopicBridgeOptions>> topics;
};

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__DOMAIN_BRIDGE_CONFIG_HPP_
