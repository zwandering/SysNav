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

#include "domain_bridge/compress_messages.hpp"

#include <zstd.h>

#include <stdexcept>
#include <sstream>
#include <vector>

namespace domain_bridge
{

// Increasing the compression level will:
//   - Increase the time taken to compress
//   - Decrease the size of the compressed data
// Setting to zero uses Zstd's default value of 3.
constexpr const int kDefaultZstdCompressionLevel = 1;

using ZstdDecompressReturnType = decltype(ZSTD_decompress(
    nullptr, 0,
    nullptr, 0));

void throw_on_zstd_error(const ZstdDecompressReturnType compression_result)
{
  if (ZSTD_isError(compression_result)) {
    std::stringstream error;
    error << "ZSTD decompression error: " << ZSTD_getErrorName(compression_result);

    throw std::runtime_error{error.str()};
  }
}


using ZstdGetFrameContentSizeReturnType = decltype(ZSTD_getFrameContentSize(nullptr, 0));

void throw_on_invalid_frame_content(const ZstdGetFrameContentSizeReturnType frame_content)
{
  if (frame_content == ZSTD_CONTENTSIZE_ERROR) {
    throw std::runtime_error{"Unable to determine file size due to error."};
  } else if (frame_content == ZSTD_CONTENTSIZE_UNKNOWN) {
    throw std::runtime_error{"Unable to determine file size."};
  }
}

std::vector<uint8_t>
compress_message(ZSTD_CCtx * ctx, rclcpp::SerializedMessage msg)
{
  // Allocate based on compression bound and compress
  const auto maximum_compressed_length =
    ZSTD_compressBound(msg.size());
  std::vector<uint8_t> compressed_buffer(maximum_compressed_length);

  // Perform compression and check.
  // compression_result is either the actual compressed size or an error code.
  const auto compression_result = ZSTD_compressCCtx(
    ctx,
    compressed_buffer.data(), maximum_compressed_length,
    msg.get_rcl_serialized_message().buffer, msg.get_rcl_serialized_message().buffer_length,
    kDefaultZstdCompressionLevel);
  throw_on_zstd_error(compression_result);

  // Compression_buffer_length might be larger than the actual compression size
  // Resize compressed_buffer so its size is the actual compression size.
  compressed_buffer.resize(compression_result);
  return compressed_buffer;
}

rclcpp::SerializedMessage
decompress_message(ZSTD_DCtx * ctx, std::vector<uint8_t> compressed_msg)
// void ZstdDecompressor::decompress_serialized_bag_message(
//   rosbag2_storage::SerializedBagMessage * message)
{
  const auto compressed_buffer_length = compressed_msg.size();

  const auto decompressed_buffer_length =
    ZSTD_getFrameContentSize(compressed_msg.data(), compressed_buffer_length);

  throw_on_invalid_frame_content(decompressed_buffer_length);

  rclcpp::SerializedMessage msg;
  msg.reserve(decompressed_buffer_length);

  const auto decompression_result = ZSTD_decompressDCtx(
    ctx,
    msg.get_rcl_serialized_message().buffer, decompressed_buffer_length,
    compressed_msg.data(), compressed_buffer_length);
  msg.get_rcl_serialized_message().buffer_length = decompressed_buffer_length;

  throw_on_zstd_error(decompression_result);
  return msg;
}

}  // namespace domain_bridge
