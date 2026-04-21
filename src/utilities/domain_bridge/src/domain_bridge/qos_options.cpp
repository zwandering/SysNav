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

#include <optional>

#include <cstdint>

#include "rclcpp/qos.hpp"

#include "domain_bridge/qos_options.hpp"

namespace domain_bridge
{

std::optional<rclcpp::ReliabilityPolicy>
QosOptions::reliability() const
{
  return reliability_;
}

QosOptions &
QosOptions::reliability(const rclcpp::ReliabilityPolicy & reliability)
{
  reliability_.emplace(reliability);
  return *this;
}

std::optional<rclcpp::DurabilityPolicy>
QosOptions::durability() const
{
  return durability_;
}

QosOptions &
QosOptions::durability(const rclcpp::DurabilityPolicy & durability)
{
  durability_.emplace(durability);
  return *this;
}

rclcpp::HistoryPolicy
QosOptions::history() const
{
  return history_;
}

QosOptions &
QosOptions::history(const rclcpp::HistoryPolicy & history)
{
  history_ = history;
  return *this;
}

size_t
QosOptions::depth() const
{
  return depth_;
}

QosOptions &
QosOptions::depth(const size_t & depth)
{
  depth_ = depth;
  return *this;
}

std::optional<int64_t>
QosOptions::deadline() const
{
  return deadline_;
}

QosOptions &
QosOptions::deadline(int64_t deadline)
{
  deadline_.emplace(deadline);
  return *this;
}

QosOptions &
QosOptions::deadline_auto()
{
  deadline_.reset();
  return *this;
}

std::optional<int64_t>
QosOptions::lifespan() const
{
  return lifespan_;
}

QosOptions &
QosOptions::lifespan(int64_t lifespan)
{
  lifespan_.emplace(lifespan);
  return *this;
}

QosOptions &
QosOptions::lifespan_auto()
{
  lifespan_.reset();
  return *this;
}

std::optional<rclcpp::LivelinessPolicy>
QosOptions::liveliness() const
{
  return liveliness_;
}

QosOptions &
QosOptions::liveliness(const rclcpp::LivelinessPolicy & liveliness)
{
  liveliness_ = liveliness;
  return *this;
}

QosOptions &
QosOptions::liveliness_auto()
{
  liveliness_ = std::nullopt;
  return *this;
}

std::optional<int64_t>
QosOptions::liveliness_lease_duration() const
{
  return liveliness_lease_duration_;
}

QosOptions &
QosOptions::liveliness_lease_duration(int64_t liveliness_lease_duration)
{
  liveliness_lease_duration_ = liveliness_lease_duration;
  return *this;
}

}  // namespace domain_bridge
