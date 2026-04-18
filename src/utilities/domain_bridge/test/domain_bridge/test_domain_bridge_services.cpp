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

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <utility>

#include "rclcpp/context.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp/node.hpp"
#include "test_msgs/srv/empty.hpp"

#include "domain_bridge/domain_bridge.hpp"


static constexpr std::size_t kDomain1{1u};
static constexpr std::size_t kDomain2{2u};

using namespace std::chrono_literals;

class TestDomainBridgeServices : public ::testing::Test
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
};

static
bool
poll_condition(std::function<bool()> condition, std::chrono::seconds timeout)
{
  auto start = std::chrono::system_clock::now();
  while (
    !condition() &&
    (start + timeout > std::chrono::system_clock::now()))
  {
    std::this_thread::sleep_for(50ms);
  }
  return condition();
}

class ScopedAsyncSpinner
{
public:
  explicit ScopedAsyncSpinner(std::shared_ptr<rclcpp::Context> context)
  : executor_{get_executor_options_with_context(std::move(context))},
    thread_{[this, stop_token = promise_.get_future()] {
        executor_.spin_until_future_complete(stop_token);
      }}
  {}

  ~ScopedAsyncSpinner()
  {
    promise_.set_value();
    // TODO(ivanpauno): Report bug in rclcpp.
    // This shouldn't be needed if spin_until_future_complete() worked
    // correctly.
    executor_.cancel();
    thread_.join();
  }

  rclcpp::Executor &
  get_executor()
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

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::promise<void> promise_;
  std::thread thread_;
};

TEST_F(TestDomainBridgeServices, bridge_service)
{
  std::atomic<bool> got_request = false;

  auto srv = node_1_->create_service<test_msgs::srv::Empty>(
    "my_service",
    [&got_request](
      const std::shared_ptr<test_msgs::srv::Empty::Request>,
      std::shared_ptr<test_msgs::srv::Empty::Response>)
    {got_request = true;});
  auto cli = node_2_->create_client<test_msgs::srv::Empty>("my_service");

  // Bridge the publisher topic to domain 2 with a remap option
  domain_bridge::DomainBridge bridge;
  bridge.bridge_service<test_msgs::srv::Empty>("my_service", kDomain1, kDomain2);

  std::atomic<bool> got_response = false;
  EXPECT_TRUE(poll_condition([cli]() {return cli->service_is_ready();}, 3s));
  auto future = cli->async_send_request(std::make_shared<test_msgs::srv::Empty::Request>());
  ScopedAsyncSpinner spinner{context_1_};
  spinner.get_executor().add_node(node_1_);
  spinner.get_executor().add_node(node_2_);
  bridge.add_to_executor(spinner.get_executor());
  EXPECT_TRUE(poll_condition([&got_request]() {return got_request.load();}, 3s));
  EXPECT_EQ(future.wait_for(3s), std::future_status::ready);
}
