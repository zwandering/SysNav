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

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "test_msgs/msg/basic_types.hpp"

using namespace std::chrono_literals;

namespace domain_bridge
{
class TestComponent : public rclcpp::Node
{
public:
  explicit TestComponent(rclcpp::NodeOptions options)
  : rclcpp::Node("test_component", options)
  {
    auto publish_message =
      [this]() -> void
      {
        pub_->publish(test_msgs::msg::BasicTypes{});
      };

    pub_ = this->create_publisher<test_msgs::msg::BasicTypes>(
      "test_domain_bridge_component_manager",
      1);
    timer_ = this->create_wall_timer(100ms, publish_message);
  }

private:
  rclcpp::Publisher<test_msgs::msg::BasicTypes>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace domain_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(domain_bridge::TestComponent)
