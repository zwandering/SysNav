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

#include <memory>
#include <string>
#include <utility>

#include "rclcpp/context.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/serialization.hpp"
#include "test_msgs/msg/basic_types.hpp"

#include "domain_bridge/component_manager.hpp"
#include "domain_bridge/compress_messages.hpp"
#include "domain_bridge/domain_bridge.hpp"
#include "domain_bridge/msg/compressed_msg.hpp"

#include "wait_for_publisher.hpp"

static constexpr std::size_t kDomain1{1u};
static constexpr std::size_t kDomain2{2u};

using namespace std::chrono_literals;

// TODO(ivanpauno): We could use parameterized testing here ...
class TestDomainBridgeEndToEnd : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Initialize contexts in different domains
    rclcpp::InitOptions context_options;

    context_1_ = std::make_shared<rclcpp::Context>();
    context_options.auto_initialize_logging(true).set_domain_id(kDomain1);
    context_1_->init(0, nullptr, context_options);

    context_2_ = std::make_shared<rclcpp::Context>();
    context_options.auto_initialize_logging(false).set_domain_id(kDomain2);
    context_2_->init(0, nullptr, context_options);

    // Initialize one node in each domain
    rclcpp::NodeOptions node_options;

    node_options.context(context_1_);
    node_1_ = std::make_shared<rclcpp::Node>("node_1", node_options);

    node_options.context(context_2_);
    node_2_ = std::make_shared<rclcpp::Node>("node_2", node_options);
  }
  std::shared_ptr<rclcpp::Context> context_1_;
  std::shared_ptr<rclcpp::Context> context_2_;
  std::shared_ptr<rclcpp::Node> node_1_;
  std::shared_ptr<rclcpp::Node> node_2_;
  // use a qos profile with transient local volatility to make the tests simple.
  // it's hard to guarantee that all pub/sub have matched and so the first few messages
  // can be lost if using a profile with volatile durability.
  rclcpp::QoS pub_sub_qos_ = rclcpp::QoS(1).transient_local();
};

static
bool
poll_condition(std::function<bool()> condition, std::chrono::seconds timeout)
{
  auto start = std::chrono::steady_clock::now();
  while (
    !condition() &&
    (start + timeout > std::chrono::steady_clock::now()))
  {
    std::this_thread::sleep_for(50ms);
  }
  return condition();
}

class ScopedAsyncSpinner
{
public:
  explicit ScopedAsyncSpinner(std::shared_ptr<rclcpp::Context> context)
  : executor_{std::make_shared<rclcpp::executors::SingleThreadedExecutor>(
        get_executor_options_with_context(std::move(context)))},
    thread_{[this, stop_token = promise_.get_future()] {
        executor_->spin_until_future_complete(stop_token);
      }}
  {}

  ~ScopedAsyncSpinner()
  {
    promise_.set_value();
    // TODO(ivanpauno): Report bug in rclcpp.
    // This shouldn't be needed if spin_until_future_complete() worked
    // correctly.
    executor_->cancel();
    thread_.join();
  }

  rclcpp::Executor &
  get_executor()
  {
    return *executor_;
  }

  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> &
  get_executor_ptr()
  {
    return executor_;
  }

private:
  static
  rclcpp::ExecutorOptions
  get_executor_options_with_context(std::shared_ptr<rclcpp::Context> context)
  {
    rclcpp::ExecutorOptions ret;
    ret.context = std::move(context);
    return ret;
  }

  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::promise<void> promise_;
  std::thread thread_;
};

TEST_F(TestDomainBridgeEndToEnd, remap_topic_name)
{
  const std::string topic_name("test_remap_topic_before");
  const std::string remap_name("test_remap_topic_after");

  std::atomic<bool> got_message = false;

  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, pub_sub_qos_);
  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    remap_name,
    pub_sub_qos_,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridge bridge;
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.remap_name(remap_name);
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2, topic_bridge_options);

  pub->publish(test_msgs::msg::BasicTypes{});
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, remap_topic_name_with_substitution)
{
  const std::string topic_name("test_remap_topic_with_sub");
  const std::string remap_name("~/test_remap_topic_with_sub");
  const std::string domain_bridge_name("my_test_name");
  const std::string expected_name("/my_test_name/test_remap_topic_with_sub");

  std::atomic<bool> got_message = false;

  // Create a publisher on domain 1
  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, pub_sub_qos_);
  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    expected_name,
    pub_sub_qos_,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridgeOptions domain_bridge_options;
  domain_bridge_options.name(domain_bridge_name);
  domain_bridge::DomainBridge bridge{domain_bridge_options};
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.remap_name(remap_name);
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2, topic_bridge_options);

  pub->publish(test_msgs::msg::BasicTypes{});
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, wait_for_subscription)
{
  const std::string topic_name("test_wait_for_subscription");
  const std::string domain_bridge_name("test_wait_for_subscription_domain_bridge");

  std::atomic<bool> got_message = false;

  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, pub_sub_qos_);

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridgeOptions domain_bridge_options;
  domain_bridge_options.name(domain_bridge_name);
  domain_bridge::DomainBridge bridge{domain_bridge_options};
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.wait_for_subscription(true);
  ASSERT_TRUE(topic_bridge_options.wait_for_subscription());
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2, topic_bridge_options);

  // bridge shouldn't be created until the subscription is up
  EXPECT_FALSE(poll_condition([pub]() {return pub->get_subscription_count() > 0;}, 5s));

  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    topic_name,
    pub_sub_qos_,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  pub->publish(test_msgs::msg::BasicTypes{});
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, auto_remove_no_subscription)
{
  const std::string topic_name("test_auto_remove_no_subscription");
  const std::string domain_bridge_name("test_auto_remove_no_subscription_domain_bridge");

  std::atomic<bool> got_message = false;

  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, pub_sub_qos_);

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridgeOptions domain_bridge_options;
  domain_bridge_options.name(domain_bridge_name);
  domain_bridge::DomainBridge bridge{domain_bridge_options};
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.wait_for_subscription(true);
  topic_bridge_options.auto_remove(domain_bridge::TopicBridgeOptions::AutoRemove::OnNoSubscription);
  ASSERT_TRUE(topic_bridge_options.wait_for_subscription());
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2, topic_bridge_options);

  // bridge shouldn't be created until the subscription is up
  EXPECT_FALSE(poll_condition([pub]() {return pub->get_subscription_count() > 0;}, 5s));

  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    topic_name,
    pub_sub_qos_,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  pub->publish(test_msgs::msg::BasicTypes{});
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
  sub.reset();
  EXPECT_FALSE(poll_condition([pub]() {return pub->get_subscription_count() == 0;}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, compress_mode)
{
  const std::string topic_name("test_compress");
  const std::string domain_bridge_name("my_test_name");

  std::atomic<bool> got_message = false;

  // Create a publisher on domain 1
  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, pub_sub_qos_);
  auto sub = node_2_->create_subscription<domain_bridge::msg::CompressedMsg>(
    topic_name,
    pub_sub_qos_,
    [&got_message](const domain_bridge::msg::CompressedMsg &) {got_message = true;});

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridgeOptions domain_bridge_options;
  domain_bridge_options.name(domain_bridge_name);
  domain_bridge_options.mode(domain_bridge::DomainBridgeOptions::Mode::Compress);
  domain_bridge::DomainBridge bridge{domain_bridge_options};
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2);

  pub->publish(test_msgs::msg::BasicTypes{});
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, decompress_mode)
{
  const std::string topic_name("test_decompress");
  const std::string domain_bridge_name("my_test_name");

  std::atomic<bool> got_message = false;

  // Create a publisher on domain 1
  auto pub = node_1_->create_publisher<domain_bridge::msg::CompressedMsg>(topic_name, pub_sub_qos_);
  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    topic_name,
    pub_sub_qos_,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridgeOptions domain_bridge_options;
  domain_bridge_options.name(domain_bridge_name);
  domain_bridge_options.mode(domain_bridge::DomainBridgeOptions::Mode::Decompress);
  domain_bridge::DomainBridge bridge{domain_bridge_options};
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2);

  rclcpp::Serialization<test_msgs::msg::BasicTypes> serializer;
  test_msgs::msg::BasicTypes msg;
  rclcpp::SerializedMessage serialized_msg;
  serializer.serialize_message(&msg, &serialized_msg);
  domain_bridge::msg::CompressedMsg compressed_msg;
  std::unique_ptr<ZSTD_CCtx, size_t (*)(ZSTD_CCtx *)> cctx{ZSTD_createCCtx(), &ZSTD_freeCCtx};
  compressed_msg.data = domain_bridge::compress_message(cctx.get(), std::move(serialized_msg));
  pub->publish(compressed_msg);
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 10s));
}

TEST_F(TestDomainBridgeEndToEnd, create_bidirectional_bridge)
{
  const std::string topic_name("test_bidirectional");

  // Create a publisher on domain 1
  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, 1);

  // Bridge the publisher topic bidirectionally
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.bidirectional(true);
  domain_bridge::DomainBridgeConfig config;
  config.topics.push_back(
  {
    {topic_name, "test_msgs/msg/BasicTypes", kDomain2, kDomain1},
    topic_bridge_options
  });
  domain_bridge::DomainBridge bridge(config);

  // 'to' domain is 1, but since we are bridging both ways, publisher should also appear on domain 2
  ASSERT_TRUE(wait_for_publisher(node_1_, topic_name));
  ASSERT_TRUE(wait_for_publisher(node_2_, topic_name));
}

TEST_F(TestDomainBridgeEndToEnd, create_reversed_bridge)
{
  const std::string topic_name("test_reversed");

  // Create a publisher on domain 1
  auto pub = node_1_->create_publisher<test_msgs::msg::BasicTypes>(topic_name, 1);

  // Bridge the publisher topic in the reversed direction
  domain_bridge::DomainBridge bridge;
  domain_bridge::TopicBridgeOptions topic_bridge_options;
  topic_bridge_options.reversed(true);
  bridge.bridge_topic(
    {topic_name, "test_msgs/msg/BasicTypes", kDomain2, kDomain1},
    topic_bridge_options
  );

  // 'to' domain is 1, but since the bridge is reversed, publisher should only appear on domain 2
  ASSERT_TRUE(wait_for_publisher(node_2_, topic_name));
}

TEST_F(TestDomainBridgeEndToEnd, domain_bridge_component_manager)
{
  rclcpp::init(0, nullptr);
  auto client_node = std::make_shared<rclcpp::Node>("client_node");
  std::shared_ptr<domain_bridge::ComponentManager> manager = nullptr;

  // Bridge topic
  const std::string topic_name("test_domain_bridge_component_manager");
  domain_bridge::DomainBridge bridge;
  bridge.bridge_topic(
    topic_name, "test_msgs/msg/BasicTypes", kDomain1, kDomain2);

  // Add component manager, client, and node_2 (to receive the message)
  ScopedAsyncSpinner spinner{context_1_};
  manager = std::make_shared<domain_bridge::ComponentManager>(spinner.get_executor_ptr());
  spinner.get_executor().add_node(manager);
  spinner.get_executor().add_node(client_node);
  spinner.get_executor().add_node(node_2_);

  // Wait for LoadNode service
  auto client = client_node->create_client<composition_interfaces::srv::LoadNode>(
    "/ComponentManager/_container/load_node");
  if (!client->wait_for_service(5s)) {
    ASSERT_TRUE(false) << "service not available after waiting";
  }

  // Load publisher component on domain 1
  auto resources = manager->get_component_resources("domain_bridge");
  EXPECT_EQ("domain_bridge::TestComponent", resources[0].first);
  auto request = std::make_shared<composition_interfaces::srv::LoadNode::Request>();
  request->package_name = "domain_bridge";
  request->plugin_name = "domain_bridge::TestComponent";
  rclcpp::Parameter domain_id_1("domain_id", rclcpp::ParameterValue(1));
  request->extra_arguments.push_back(domain_id_1.to_parameter_msg());
  auto result = client->async_send_request(request);
  EXPECT_TRUE(result.get()->success);

  // Create subscriber on domain 2
  std::atomic<bool> got_message = false;
  auto sub = node_2_->create_subscription<test_msgs::msg::BasicTypes>(
    topic_name,
    1,
    [&got_message](const test_msgs::msg::BasicTypes &) {got_message = true;});

  // Add bridge and verify message from component is received
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_message]() {return got_message.load();}, 1s));
}
