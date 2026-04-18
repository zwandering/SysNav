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

#ifndef DOMAIN_BRIDGE__TOPIC_BRIDGE_HPP_
#define DOMAIN_BRIDGE__TOPIC_BRIDGE_HPP_

#include <cstddef>
#include <string>

namespace domain_bridge
{

/// Info and ROS entities related to a topic bridge
struct TopicBridge
{
  /// Name of the bridged topic
  std::string topic_name;

  /// Name of the message type sent on the topic
  std::string type_name;

  /// Domain ID that the subscription uses
  std::size_t from_domain_id;

  /// Domain ID that the publisher uses
  std::size_t to_domain_id;

  /// Less-than operator.
  /**
   * Sort by 'from_domain_id',
   *   then by 'to_domain_id',
   *   then by 'topic_name',
   *   then by 'type_name'
   */
  bool operator<(const TopicBridge & other) const
  {
    if (from_domain_id < other.from_domain_id) {
      return true;
    }
    if (from_domain_id > other.from_domain_id) {
      return false;
    }
    if (to_domain_id < other.to_domain_id) {
      return true;
    }
    if (to_domain_id > other.to_domain_id) {
      return false;
    }
    int name_compare = topic_name.compare(other.topic_name);
    if (name_compare < 0) {
      return true;
    }
    if (name_compare > 0) {
      return false;
    }
    int type_compare = type_name.compare(other.type_name);
    if (type_compare < 0) {
      return true;
    }
    return false;
  }
};

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__TOPIC_BRIDGE_HPP_
