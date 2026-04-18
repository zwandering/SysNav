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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/context.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"

#include "domain_bridge/domain_bridge.hpp"

class TestDomainBridge : public ::testing::Test
{
protected:
  static void SetUpTestCase()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestCase()
  {
    rclcpp::shutdown();
  }
};

TEST_F(TestDomainBridge, construction_destruction)
{
  // Default
  {
    domain_bridge::DomainBridge bridge;
  }
  // With options
  {
    domain_bridge::DomainBridge bridge{domain_bridge::DomainBridgeOptions()};
  }
}

TEST_F(TestDomainBridge, get_options)
{
  domain_bridge::DomainBridgeOptions input_options;
  input_options.name("domain_bridge_test_name");
  domain_bridge::DomainBridge bridge(input_options);
  domain_bridge::DomainBridgeOptions output_options = bridge.get_domain_bridge_options();
  EXPECT_EQ(input_options.name(), output_options.name());
}

TEST_F(TestDomainBridge, bridge_topic_valid)
{
  // Individual parameters
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
  }
  // Struct parameter
  {
    domain_bridge::DomainBridge bridge;
    domain_bridge::TopicBridge topic_bridge = {
      "foo",
      "test_msgs/msg/BasicTypes",
      1u,
      2u
    };
    bridge.bridge_topic(topic_bridge);
  }
  // Topic with namespace
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo/bar/baz", "test_msgs/msg/BasicTypes", 1u, 2u);
  }
  // Multiple topics
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    bridge.bridge_topic("bar", "test_msgs/msg/BasicTypes", 1u, 2u);
    bridge.bridge_topic("baz", "test_msgs/msg/BasicTypes", 1u, 2u);
  }
  // Same topic, different domains
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 3u);
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 2u, 1u);
  }
  // Same topic, different types
  // This use-case isn't clearly supported by ROS 2, see https://github.com/ros2/ros2/issues/1095
  // {
  //   domain_bridge::DomainBridge bridge;
  //   bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
  //   bridge.bridge_topic("foo", "test_msgs/msg/Nested", 1u, 2u);
  //   bridge.bridge_topic("foo", "test_msgs/msg/Constants", 1u, 2u);
  // }
}

TEST_F(TestDomainBridge, bridge_topic_invalid)
{
  // Invalid topic name
  {
    domain_bridge::DomainBridge bridge;
    EXPECT_THROW(
      bridge.bridge_topic("Not a v@lid topic name!", "test_msgs/msg/BasicTypes", 1u, 2u),
      rclcpp::exceptions::InvalidTopicNameError);
  }
  // Invalid type name
  {
    domain_bridge::DomainBridge bridge;
    EXPECT_THROW(
      bridge.bridge_topic("foo", "not a valid message type", 1u, 2u), std::runtime_error);
  }
  // Same domain ID
  {
    testing::internal::CaptureStderr();
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 1u);
    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(
      stderr_output,
      ::testing::HasSubstr(
        "Cannot bridge topic '/foo' from domain 1 to domain 1. Domain IDs must be different.\n"));
  }
  // Same bridge twice
  {
    testing::internal::CaptureStderr();
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(
      stderr_output,
      ::testing::HasSubstr(
        "Topic '/foo' with type 'test_msgs/msg/BasicTypes' already bridged from "
        "domain 1 to domain 2, ignoring\n"));
  }
  // Same bridge twice, following another bridge
  // Regression test for https://github.com/ros2/domain_bridge/pull/3
  {
    testing::internal::CaptureStderr();
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic("bar", "test_msgs/msg/Strings", 1u, 2u);
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    bridge.bridge_topic("foo", "test_msgs/msg/BasicTypes", 1u, 2u);
    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(
      stderr_output,
      ::testing::HasSubstr(
        "Topic '/foo' with type 'test_msgs/msg/BasicTypes' already bridged from "
        "domain 1 to domain 2, ignoring\n"));
  }
}

TEST_F(TestDomainBridge, add_to_executor_valid)
{
  // No nodes
  {
    domain_bridge::DomainBridge bridge;
    rclcpp::executors::SingleThreadedExecutor executor;
    bridge.add_to_executor(executor);
  }
  // With nodes
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic(
      "foo",
      "test_msgs/msg/BasicTypes",
      1u,
      2u);
    rclcpp::executors::SingleThreadedExecutor executor;
    bridge.add_to_executor(executor);
  }
}

TEST_F(TestDomainBridge, add_to_executor_invalid)
{
  // Add twice
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic(
      "foo",
      "test_msgs/msg/BasicTypes",
      1u,
      2u);
    rclcpp::executors::SingleThreadedExecutor executor;
    bridge.add_to_executor(executor);
    EXPECT_THROW(bridge.add_to_executor(executor), std::runtime_error);
  }
  // Add to different executors
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic(
      "foo",
      "test_msgs/msg/BasicTypes",
      1u,
      2u);
    rclcpp::executors::SingleThreadedExecutor executor1;
    rclcpp::executors::SingleThreadedExecutor executor2;
    bridge.add_to_executor(executor1);
    EXPECT_THROW(bridge.add_to_executor(executor2), std::runtime_error);
  }
}

TEST_F(TestDomainBridge, get_bridged_topics)
{
  // Empty
  {
    domain_bridge::DomainBridge bridge;
    auto result = bridge.get_bridged_topics();
    EXPECT_EQ(result.size(), 0u);
  }
  // One topic
  {
    domain_bridge::DomainBridge bridge;
    bridge.bridge_topic({"foo", "test_msgs/msg/BasicTypes", 1u, 2u});
    auto result = bridge.get_bridged_topics();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].topic_name, "foo");
    EXPECT_EQ(result[0].type_name, "test_msgs/msg/BasicTypes");
    EXPECT_EQ(result[0].from_domain_id, 1u);
    EXPECT_EQ(result[0].to_domain_id, 2u);
  }
  // Several topics
  {
    domain_bridge::DomainBridge bridge;
    std::vector<domain_bridge::TopicBridge> input = {
      {"foo", "test_msgs/msg/BasicTypes", 3u, 2u},
      {"foo", "test_msgs/msg/BasicTypes", 1u, 2u},
      {"foo", "test_msgs/msg/BasicTypes", 2u, 1u},
      {"baz", "test_msgs/msg/Strings", 1u, 2u},
      {"bar", "test_msgs/msg/BasicTypes", 1u, 2u},
      {"foo", "test_msgs/msg/BasicTypes", 2u, 2u},  // same domain ID
      {"foo", "test_msgs/msg/BasicTypes", 1u, 2u}  // duplicate
    };
    for (const auto & topic_bridge : input) {
      bridge.bridge_topic(topic_bridge);
    }
    auto result = bridge.get_bridged_topics();
    ASSERT_EQ(result.size(), 5u);  // duplicate and same domain ID entries should not appear
    // Expect result to be in sorted order
    std::sort(input.begin(), input.end() - 2);  // - 2 to ignore duplicate and same domain ID
    for (std::size_t i = 0u; i < result.size(); ++i) {
      EXPECT_EQ(result[i].topic_name, input[i].topic_name);
      EXPECT_EQ(result[i].type_name, input[i].type_name);
      EXPECT_EQ(result[i].from_domain_id, input[i].from_domain_id);
      EXPECT_EQ(result[i].to_domain_id, input[i].to_domain_id);
    }
  }
}

TEST_F(TestDomainBridge, remap_topic_name_invalid)
{
  const std::string topic_name("test_remap_topic_invalid_before");
  const std::string remap_name("test_remap_topic_1nv@lid_after");

  domain_bridge::DomainBridge bridge;
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.remap_name(remap_name);
  EXPECT_THROW(
    bridge.bridge_topic(
      topic_name, "test_msgs/msg/BasicTypes", 0, 2, topic_bridge_options),
    rclcpp::exceptions::InvalidTopicNameError);
}
