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

#include <filesystem>

#include <string>
#include <vector>

#include "domain_bridge/domain_bridge_config.hpp"
#include "domain_bridge/parse_domain_bridge_yaml_config.hpp"
#include "domain_bridge/topic_bridge.hpp"

class TestParseDomainBridgeYamlConfig : public ::testing::Test
{
protected:
  void SetUp()
  {
    test_yaml_dir_ = std::filesystem::current_path() / std::filesystem::path{"config"};
  }

  std::filesystem::path test_yaml_dir_;
};

TEST_F(TestParseDomainBridgeYamlConfig, empty)
{
  // Empty YAML file
  const std::string yaml_path = (test_yaml_dir_ / std::filesystem::path{"empty.yaml"}).string();
  auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
  // Default name
  EXPECT_EQ(config.options.name(), "domain_bridge");
  // No topics
  EXPECT_EQ(config.topics.size(), 0u);
}

TEST_F(TestParseDomainBridgeYamlConfig, name)
{
  const std::string yaml_path = (test_yaml_dir_ / std::filesystem::path{"name.yaml"}).string();
  auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
  EXPECT_EQ(config.options.name(), "domain_bridge_test_name");
  EXPECT_EQ(config.topics.size(), 0u);
}

TEST_F(TestParseDomainBridgeYamlConfig, mode)
{
  {
    const std::string yaml_path = (
      test_yaml_dir_ / std::filesystem::path{"compress_mode.yaml"}).string();
    auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
    EXPECT_EQ(config.options.mode(), domain_bridge::DomainBridgeOptions::Mode::Compress);
    EXPECT_EQ(config.topics.size(), 0u);
  }
  {
    const std::string yaml_path = (
      test_yaml_dir_ / std::filesystem::path{"decompress_mode.yaml"}).string();
    auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
    EXPECT_EQ(config.options.mode(), domain_bridge::DomainBridgeOptions::Mode::Decompress);
    EXPECT_EQ(config.topics.size(), 0u);
  }
}

TEST_F(TestParseDomainBridgeYamlConfig, topics)
{
  std::vector<domain_bridge::TopicBridge> expected = {
    {"baz", "test_msgs/msg/Empty", 10, 7},
    {"/foo/bar", "test_msgs/msg/BasicTypes", 6, 10},
    {"foo", "test_msgs/msg/Strings", 3, 4},
    {"foo", "test_msgs/msg/Strings", 3, 4}
  };
  const std::string yaml_path = (test_yaml_dir_ / std::filesystem::path{"topics.yaml"}).string();
  auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
  ASSERT_EQ(config.topics.size(), expected.size());
  for (std::size_t i = 0u; i < config.topics.size(); ++i) {
    EXPECT_EQ(config.topics[i].first.topic_name, expected[i].topic_name);
    EXPECT_EQ(config.topics[i].first.type_name, expected[i].type_name);
    EXPECT_EQ(config.topics[i].first.from_domain_id, expected[i].from_domain_id);
    EXPECT_EQ(config.topics[i].first.to_domain_id, expected[i].to_domain_id);
  }
}

TEST_F(TestParseDomainBridgeYamlConfig, multiple_files)
{
  std::vector<domain_bridge::TopicBridge> expected = {
    {"baz", "test_msgs/msg/Empty", 10, 7},
    {"/foo/bar", "test_msgs/msg/BasicTypes", 6, 10},
    {"foo", "test_msgs/msg/Strings", 3, 4},
    {"foo", "test_msgs/msg/Strings", 3, 4},
    {"baz", "test_msgs/msg/Empty", 7, 11},
    {"/foo/bar", "test_msgs/msg/BasicTypes", 6, 11},
    {"foo", "test_msgs/msg/Strings", 7, 4},
    {"foobar", "test_msgs/msg/Strings", 3, 4}
  };
  std::vector<std::filesystem::path> file_paths;
  file_paths.push_back((test_yaml_dir_ / std::filesystem::path{"topics.yaml"}));
  file_paths.push_back((test_yaml_dir_ / std::filesystem::path{"default_domain_ids.yaml"}));
  auto config = domain_bridge::parse_domain_bridge_yaml_configs(file_paths);
  ASSERT_EQ(config.topics.size(), expected.size());
  for (std::size_t i = 0u; i < config.topics.size(); ++i) {
    EXPECT_EQ(config.topics[i].first.topic_name, expected[i].topic_name);
    EXPECT_EQ(config.topics[i].first.type_name, expected[i].type_name);
    EXPECT_EQ(config.topics[i].first.from_domain_id, expected[i].from_domain_id);
    EXPECT_EQ(config.topics[i].first.to_domain_id, expected[i].to_domain_id);
  }
}

TEST_F(TestParseDomainBridgeYamlConfig, default_domain_ids)
{
  std::vector<domain_bridge::TopicBridge> expected = {
    {"baz", "test_msgs/msg/Empty", 7, 11},
    {"/foo/bar", "test_msgs/msg/BasicTypes", 6, 11},
    {"foo", "test_msgs/msg/Strings", 7, 4},
    {"foobar", "test_msgs/msg/Strings", 3, 4}
  };
  const std::string yaml_path =
    (test_yaml_dir_ / std::filesystem::path{"default_domain_ids.yaml"}).string();
  auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
  ASSERT_EQ(config.topics.size(), expected.size());
  for (std::size_t i = 0u; i < config.topics.size(); ++i) {
    EXPECT_EQ(config.topics[i].first.topic_name, expected[i].topic_name);
    EXPECT_EQ(config.topics[i].first.type_name, expected[i].type_name);
    EXPECT_EQ(config.topics[i].first.from_domain_id, expected[i].from_domain_id);
    EXPECT_EQ(config.topics[i].first.to_domain_id, expected[i].to_domain_id);
  }
}

TEST_F(TestParseDomainBridgeYamlConfig, topic_options)
{
  std::vector<domain_bridge::TopicBridgeOptions> expected = {
    domain_bridge::TopicBridgeOptions(),
    domain_bridge::TopicBridgeOptions().qos_options(
      domain_bridge::QosOptions()
      .reliability(rclcpp::ReliabilityPolicy::BestEffort)
      .durability(rclcpp::DurabilityPolicy::TransientLocal)
      .history(rclcpp::HistoryPolicy::KeepAll)
      .depth(42u)
      .deadline(123456)
      .lifespan_auto()
      .liveliness(rclcpp::LivelinessPolicy::ManualByTopic)
      .liveliness_lease_duration(1000))
    .remap_name("bar/remapped")
    .bidirectional(true),
    domain_bridge::TopicBridgeOptions().qos_options(
      domain_bridge::QosOptions()
      .deadline_auto()
      .lifespan(-5)
      .liveliness_lease_duration(-1000))
    .remap_name("")
    .reversed(true)
  };
  const std::string yaml_path =
    (test_yaml_dir_ / std::filesystem::path{"topic_options.yaml"}).string();
  auto config = domain_bridge::parse_domain_bridge_yaml_config(yaml_path);
  ASSERT_EQ(config.topics.size(), expected.size());
  for (std::size_t i = 0u; i < config.topics.size(); ++i) {
    EXPECT_EQ(config.topics[i].second.callback_group(), expected[i].callback_group());
    EXPECT_EQ(
      config.topics[i].second.qos_options().reliability(),
      expected[i].qos_options().reliability());
    EXPECT_EQ(
      config.topics[i].second.qos_options().durability(),
      expected[i].qos_options().durability());
    EXPECT_EQ(
      config.topics[i].second.qos_options().history(), expected[i].qos_options().history());
    EXPECT_EQ(
      config.topics[i].second.qos_options().depth(), expected[i].qos_options().depth());
    EXPECT_EQ(config.topics[i].second.remap_name(), expected[i].remap_name());
    EXPECT_EQ(config.topics[i].second.bidirectional(), expected[i].bidirectional());
    EXPECT_EQ(config.topics[i].second.reversed(), expected[i].reversed());
  }
}

TEST_F(TestParseDomainBridgeYamlConfig, invalid)
{
  // Non-existent file
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"not_a_file.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Missing 'type' key
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"missing_type.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Missing 'from_domain' key
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"missing_from_domain.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Missing 'to_domain' key
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"missing_to_domain.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // 'topics' has wrong type
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"topics_as_list.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // 'qos' has wrong type
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"qos_as_list.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Invalid reliability value
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"invalid_reliability.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Invalid durability value
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"invalid_durability.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Invalid history value
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"invalid_history.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Invalid depth value
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"invalid_depth.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
  // Invalid mode
  {
    const std::string yaml_path =
      (test_yaml_dir_ / std::filesystem::path{"invalid_mode.yaml"}).string();
    EXPECT_THROW(
      domain_bridge::parse_domain_bridge_yaml_config(yaml_path), domain_bridge::YamlParsingError);
  }
}
