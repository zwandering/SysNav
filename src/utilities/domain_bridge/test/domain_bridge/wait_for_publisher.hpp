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

#ifndef DOMAIN_BRIDGE__WAIT_FOR_PUBLISHER_HPP_
#define DOMAIN_BRIDGE__WAIT_FOR_PUBLISHER_HPP_

#include <chrono>
#include <string>
#include <thread>

#include "rclcpp/event.hpp"
#include "rclcpp/node.hpp"

/// Wait for a publisher to be available (or not)
/*
 * \return true if the wait was successful, false if a timeout occured.
 */
inline bool wait_for_publisher(
  rclcpp::Node::SharedPtr node,
  const std::string & topic_name,
  std::chrono::nanoseconds timeout = std::chrono::seconds(3),
  std::chrono::nanoseconds sleep_period = std::chrono::milliseconds(100))
{
  auto start = std::chrono::steady_clock::now();
  std::chrono::microseconds time_slept(0);
  auto predicate = [&node, &topic_name]() -> bool {
      // A publisher is available if the count is greater than 0
      return node->count_publishers(topic_name) > 0;
    };

  do {
    if (predicate()) {
      return true;
    }
    rclcpp::Event::SharedPtr graph_event = node->get_graph_event();
    node->wait_for_graph_change(graph_event, sleep_period);
    time_slept = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start);
  } while (!predicate() &&
  time_slept < std::chrono::duration_cast<std::chrono::microseconds>(timeout));

  return predicate();
}

#endif  // DOMAIN_BRIDGE__WAIT_FOR_PUBLISHER_HPP_
