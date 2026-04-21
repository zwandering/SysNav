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

#ifndef DOMAIN_BRIDGE__COMPRESS_MESSAGES_HPP_
#define DOMAIN_BRIDGE__COMPRESS_MESSAGES_HPP_

#include <zstd.h>

#include <vector>

#include "rclcpp/serialized_message.hpp"

#include "domain_bridge/visibility_control.hpp"

namespace domain_bridge
{

DOMAIN_BRIDGE_PUBLIC
std::vector<uint8_t>
compress_message(ZSTD_CCtx * ctx, rclcpp::SerializedMessage msg);

DOMAIN_BRIDGE_PUBLIC
rclcpp::SerializedMessage
decompress_message(ZSTD_DCtx * ctx, std::vector<uint8_t> compressed_msg);

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__COMPRESS_MESSAGES_HPP_
