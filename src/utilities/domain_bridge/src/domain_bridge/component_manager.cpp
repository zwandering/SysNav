// Copyright 2021 Open Source Robotics Foundation, Inc.
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

#include "domain_bridge/component_manager.hpp"

#include <memory>

namespace domain_bridge
{

rclcpp::NodeOptions
ComponentManager::create_node_options(const std::shared_ptr<LoadNode::Request> request)
{
  auto options = rclcpp_components::ComponentManager::create_node_options(request);

  for (const auto & a : request->extra_arguments) {
    const rclcpp::Parameter extra_argument = rclcpp::Parameter::from_parameter_msg(a);
    if (extra_argument.get_name() == "domain_id") {
      if (extra_argument.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER ||
        extra_argument.get_value<int>() < 0)
      {
        throw rclcpp_components::ComponentManagerException(
                "Extra component argument 'domain_id' must be a positive integer");
      }
      std::size_t domain_id = extra_argument.get_value<std::size_t>();

      auto it = contexts_.find(domain_id);
      if (it == contexts_.end()) {
        // if context does not already exist for given domain id, create one
        auto context = std::make_shared<rclcpp::Context>();
        rclcpp::InitOptions init_options;
        init_options.auto_initialize_logging(false).set_domain_id(domain_id);
        context->init(0, nullptr, init_options);
        auto it_emplaced_pair = contexts_.emplace(domain_id, context);
        it = it_emplaced_pair.first;
      }
      options.context(it->second);
    }
  }

  return options;
}

}  // namespace domain_bridge
