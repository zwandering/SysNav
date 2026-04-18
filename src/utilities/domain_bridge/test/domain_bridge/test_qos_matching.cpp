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

#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/context.hpp"
#include "rclcpp/node.hpp"
#include "test_msgs/msg/basic_types.hpp"

#include "domain_bridge/domain_bridge.hpp"

#include "wait_for_publisher.hpp"

class TestDomainBridgeQosMatching : public ::testing::Test
{
protected:
  static void SetUpTestCase()
  {
    // Initialize contexts in different domains
    context_1_ = std::make_shared<rclcpp::Context>();
    rclcpp::InitOptions context_options_1;
    context_options_1.auto_initialize_logging(false).set_domain_id(domain_1_);
    context_1_->init(0, nullptr, context_options_1);

    context_2_ = std::make_shared<rclcpp::Context>();
    rclcpp::InitOptions context_options_2;
    context_options_2.auto_initialize_logging(false).set_domain_id(domain_2_);
    context_2_->init(0, nullptr, context_options_2);

    node_options_1_.context(context_1_);
    node_options_2_.context(context_2_);
  }

  static const std::size_t domain_1_{1u};
  static const std::size_t domain_2_{2u};
  static std::shared_ptr<rclcpp::Context> context_1_;
  static std::shared_ptr<rclcpp::Context> context_2_;
  static rclcpp::NodeOptions node_options_1_;
  static rclcpp::NodeOptions node_options_2_;
};

const std::size_t TestDomainBridgeQosMatching::domain_1_;
const std::size_t TestDomainBridgeQosMatching::domain_2_;
std::shared_ptr<rclcpp::Context> TestDomainBridgeQosMatching::context_1_;
std::shared_ptr<rclcpp::Context> TestDomainBridgeQosMatching::context_2_;
rclcpp::NodeOptions TestDomainBridgeQosMatching::node_options_1_;
rclcpp::NodeOptions TestDomainBridgeQosMatching::node_options_2_;

TEST_F(TestDomainBridgeQosMatching, qos_matches_topic_exists_before_bridge)
{
  const std::string topic_name("test_topic_exists_before_bridge");

  // Create a publisher on domain 1
  auto node_1 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_before_bridge_node_1", node_options_1_);
  rclcpp::QoS qos(1);
  qos.best_effort()
  .transient_local()
  .deadline(rclcpp::Duration(123, 456u))
  .lifespan(rclcpp::Duration(554, 321u))
  .liveliness(rclcpp::LivelinessPolicy::Automatic);
  auto pub = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos);

  // Bridge the publisher topic to domain 2
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_);

  // Wait for bridge publisher to appear on domain 2
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_before_bridge_node_2", node_options_2_);
  ASSERT_TRUE(wait_for_publisher(node_2, topic_name));

  // Assert the QoS of the bridged publisher matches
  std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec =
    node_2->get_publishers_info_by_topic(topic_name);
  ASSERT_EQ(endpoint_info_vec.size(), 1u);
  const rclcpp::QoS & bridged_qos = endpoint_info_vec[0].qos_profile();
  EXPECT_EQ(bridged_qos.reliability(), qos.reliability());
  EXPECT_EQ(bridged_qos.durability(), qos.durability());
  EXPECT_EQ(bridged_qos.liveliness(), qos.liveliness());
  // Deadline and lifespan default to max
  auto max_duration = rclcpp::Duration::from_nanoseconds(std::numeric_limits<int64_t>::max());
  EXPECT_EQ(bridged_qos.deadline(), max_duration);
  EXPECT_EQ(bridged_qos.lifespan(), max_duration);
}

TEST_F(TestDomainBridgeQosMatching, qos_matches_topic_exists_after_bridge)
{
  const std::string topic_name("test_topic_exists_after_bridge");

  auto node_1 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_after_bridge_node_1", node_options_1_);

  // Bridge the publisher topic to domain 2
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_);

  // Wait for bridge publisher to appear on domain 2
  // It shouldn't be available yet
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_after_bridge_node_2", node_options_2_);
  ASSERT_FALSE(wait_for_publisher(node_2, topic_name, std::chrono::milliseconds(300)));

  // Create a publisher on domain 1
  rclcpp::QoS qos(1);
  qos.best_effort()
  .transient_local()
  .deadline(rclcpp::Duration(123, 456u))
  .lifespan(rclcpp::Duration(554, 321u))
  .liveliness(rclcpp::LivelinessPolicy::Automatic);
  auto pub = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos);

  // Wait for bridge publihser to appear on domain 2
  // It should be there now
  ASSERT_TRUE(wait_for_publisher(node_2, topic_name));

  // Assert the QoS of the bridged publisher matches
  std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec =
    node_2->get_publishers_info_by_topic(topic_name);
  ASSERT_EQ(endpoint_info_vec.size(), 1u);
  const rclcpp::QoS & bridged_qos = endpoint_info_vec[0].qos_profile();
  EXPECT_EQ(bridged_qos.reliability(), qos.reliability());
  EXPECT_EQ(bridged_qos.durability(), qos.durability());
  EXPECT_EQ(bridged_qos.liveliness(), qos.liveliness());
  // Deadline and lifespan default to max
  auto max_duration = rclcpp::Duration::from_nanoseconds(std::numeric_limits<int64_t>::max());
  EXPECT_EQ(bridged_qos.deadline(), max_duration);
  EXPECT_EQ(bridged_qos.lifespan(), max_duration);
}

TEST_F(TestDomainBridgeQosMatching, qos_matches_topic_exists_multiple_publishers)
{
  const std::string topic_name("test_topic_exists_multiple_publishers");

  // Create two publishers on domain 1
  auto node_1 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_multiple_publishers_node_1", node_options_1_);
  rclcpp::QoS qos(1);
  qos.reliable()
  .durability_volatile()
  .deadline(rclcpp::Duration(123, 456u))
  .lifespan(rclcpp::Duration(554, 321u))
  .liveliness(rclcpp::LivelinessPolicy::Automatic);
  auto pub_1 = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos);
  // Second publisher has different QoS
  qos.best_effort().transient_local();
  auto pub_2 = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos);
  // Delay creation of bridge to allow both publishers to become available
  domain_bridge::TopicBridgeOptions bridge_options;
  bridge_options.delay(std::chrono::milliseconds(1000));

  // Bridge the publisher topic to domain 2
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_, bridge_options);

  // Wait for bridge publisher to appear on domain 2
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_topic_exists_multiple_publishers_node_2", node_options_2_);
  ASSERT_TRUE(wait_for_publisher(node_2, topic_name));

  // Assert the QoS of the bridged publishers matches both publishers
  // I.e. it should have best effort reliability and volatile durability
  std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec =
    node_2->get_publishers_info_by_topic(topic_name);
  ASSERT_EQ(endpoint_info_vec.size(), 1u);
  const rclcpp::QoS & bridged_qos = endpoint_info_vec[0].qos_profile();
  EXPECT_EQ(bridged_qos.reliability(), rclcpp::ReliabilityPolicy::BestEffort);
  EXPECT_EQ(bridged_qos.durability(), rclcpp::DurabilityPolicy::Volatile);
  EXPECT_EQ(bridged_qos.liveliness(), qos.liveliness());
  // Deadline and lifespan default to max
  auto max_duration = rclcpp::Duration::from_nanoseconds(std::numeric_limits<int64_t>::max());
  EXPECT_EQ(bridged_qos.deadline(), max_duration);
  EXPECT_EQ(bridged_qos.lifespan(), max_duration);
}

TEST_F(TestDomainBridgeQosMatching, qos_matches_topic_does_not_exist)
{
  const std::string topic_name("test_topic_does_not_exist");

  // Bridge a non-existent publisher topic to domain 2
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_);

  // We do not expect a bridge publisher to appear because there is no input publisher
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_topic_does_not_exist_node_2", node_options_2_);
  ASSERT_FALSE(wait_for_publisher(node_2, topic_name, std::chrono::seconds(1)));
}

TEST_F(TestDomainBridgeQosMatching, qos_matches_always_automatic_liveliness)
{
  const std::string topic_name("test_always_automatic_liveliness");

  // Create a publisher on domain 1 with liveliness set to "manual by topic"
  auto node_1 = std::make_shared<rclcpp::Node>(
    "test_always_automatic_liveliness_node_1", node_options_1_);
  rclcpp::QoS qos(1);
  qos.liveliness(rclcpp::LivelinessPolicy::ManualByTopic);
  auto pub = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos);

  // Bridge the publisher topic to domain 2
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_);

  // Wait for bridge publisher to appear on domain 2
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_always_automatic_liveliness_node_2", node_options_2_);
  ASSERT_TRUE(wait_for_publisher(node_2, topic_name));

  // Assert the liveliness policy is "automatic", not "manual by topic"
  std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec =
    node_2->get_publishers_info_by_topic(topic_name);
  ASSERT_EQ(endpoint_info_vec.size(), 1u);
  const rclcpp::QoS & bridged_qos = endpoint_info_vec[0].qos_profile();
  EXPECT_EQ(bridged_qos.liveliness(), rclcpp::LivelinessPolicy::Automatic);
}

TEST_F(TestDomainBridgeQosMatching, qos_matches_max_of_duration_policy)
{
  const std::string topic_name("test_max_of_duration_policy");

  // Create two publishers on domain 1
  // The first deadline will be greater than the second deadline
  // The second lifespan will be greater than the first lifespan
  auto node_1 = std::make_shared<rclcpp::Node>(
    "test_max_of_duration_policy_node_1", node_options_1_);
  rclcpp::QoS qos_1(1);
  qos_1.deadline(rclcpp::Duration(554, 321u))
  .lifespan(rclcpp::Duration(123, 456u));
  rclcpp::QoS qos_2(1);
  qos_2.deadline(rclcpp::Duration(123, 456u))
  .lifespan(rclcpp::Duration(554, 321u));
  auto pub_1 = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos_1);
  auto pub_2 = node_1->create_publisher<test_msgs::msg::BasicTypes>(topic_name, qos_2);

  // Bridge the publisher topic to domain 2
  // Enable auto-matching for deadline and lifespan
  domain_bridge::TopicBridgeOptions bridge_options;
  domain_bridge::QosOptions qos_options;
  qos_options.deadline_auto().lifespan_auto();
  bridge_options.qos_options(qos_options);
  // Delay creation of bridge to allow both publishers to become available
  bridge_options.delay(std::chrono::milliseconds(1000));
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", domain_1_, domain_2_, bridge_options);

  // Wait for bridge publisher to appear on domain 2
  auto node_2 = std::make_shared<rclcpp::Node>(
    "test_max_of_duration_policy_node_2", node_options_2_);
  ASSERT_TRUE(wait_for_publisher(node_2, topic_name));

  // Assert max of the two deadline and lifespan policies are used for the bridge QoS
  std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec =
    node_2->get_publishers_info_by_topic(topic_name);
  ASSERT_EQ(endpoint_info_vec.size(), 1u);
  const rclcpp::QoS & bridged_qos = endpoint_info_vec[0].qos_profile();
  EXPECT_EQ(bridged_qos.deadline(), rclcpp::Duration(554, 321u));
  EXPECT_EQ(bridged_qos.lifespan(), rclcpp::Duration(554, 321u));
}
