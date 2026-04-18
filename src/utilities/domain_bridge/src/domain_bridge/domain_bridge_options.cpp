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

#include "domain_bridge/domain_bridge_options.hpp"

#include <functional>
#include <string>
#include <utility>

namespace domain_bridge
{

const std::string &
DomainBridgeOptions::name() const
{
  return this->name_;
}

DomainBridgeOptions &
DomainBridgeOptions::name(std::string name)
{
  this->name_ = name;
  return *this;
}

DomainBridgeOptions::Mode
DomainBridgeOptions::mode() const
{
  return mode_;
}

DomainBridgeOptions &
DomainBridgeOptions::mode(DomainBridgeOptions::Mode mode)
{
  mode_ = mode;
  return *this;
}

DomainBridgeOptions &
DomainBridgeOptions::on_new_domain_callback(std::function<void(size_t)> callback)
{
  this->on_new_domain_callback_ = std::move(callback);
  return *this;
}

}  // namespace domain_bridge
