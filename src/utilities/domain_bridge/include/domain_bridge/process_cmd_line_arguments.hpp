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

#ifndef DOMAIN_BRIDGE__PROCESS_CMD_LINE_ARGUMENTS_HPP_
#define DOMAIN_BRIDGE__PROCESS_CMD_LINE_ARGUMENTS_HPP_

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "domain_bridge/domain_bridge_config.hpp"

namespace domain_bridge
{
namespace detail
{
static constexpr char kCompressModeStr[] = "compress";
static constexpr char kDecompressModeStr[] = "decompress";
static constexpr char kNormalModeStr[] = "normal";

inline
void
print_help()
{
  std::cerr << "Usage: domain_bridge "
    "[--from FROM_DOMAIN_ID] [--to TO_DOMAIN_ID] [-h] YAML_CONFIG ..." << std::endl <<
    std::endl <<
    "Arguments:" << std::endl <<
    "    YAML_CONFIG    path to a YAML configuration file." << std::endl <<
    std::endl <<
    "Options:" << std::endl <<
    "    --from FROM_DOMAIN_ID    All data will be bridged from this domain ID.  " << std::endl <<
    "                             This overrides any domain IDs set in the YAML file." <<
    std::endl <<
    "    --to TO_DOMAIN_ID        All data will be bridged to this domain ID.  " << std::endl <<
    "                             This overrides any domain IDs set in the YAML file." <<
    std::endl <<
    "    --mode MODE_STR          Specify the bridge mode, valid values: '" << kCompressModeStr <<
    "', '" << kDecompressModeStr << "' or '" << kNormalModeStr << "'.  " << std::endl <<
    "                             This overrides any mode set in the YAML file." << std::endl <<
    "                             Defaults to '" << kNormalModeStr << "'." <<
    std::endl <<
    "    --wait-for-subscription true|false  Will wait for an available subscription before"
    " bridging a topic. This overrides any value set in the YAML file. Defaults to false." <<
    std::endl <<
    "    --wait-for-publisher true|false  Will wait for an available subscription before"
    " bridging a topic. This overrides any value set in the YAML file. Defaults to true." <<
    std::endl <<
    "    --auto-remove true|false  If true, the bridge will be removed if the endpoint that was"
    " waited on is removed. I.e. if both --wait-for-subscription and --wait-for-publisher were"
    " passed, then when either the subscription or publisher is removed, the bridge will also be"
    " removed."
    " If only --wait-for-subscription was passed, the bridge will only be removed if the"
    " subscription is."
    " The bridge will be recreated when the original \"wait for\" condition is satisfied." <<
    std::endl <<
    "    --help, -h               Print this help message." << std::endl;
}

/// Parse argument string and return as a size_t
/**
 * \param arg argument string
 * \param error_str argument name to output if parsing fails
 * \return size_t value
*/
inline
std::optional<size_t>
parse_size_t_arg(const std::string & arg, const char * error_str)
{
  size_t value;
  std::istringstream iss(arg);
  iss >> value;
  if (iss.fail() || !iss.eof()) {
    std::cerr << "error: Failed to parse " << error_str << " '" <<
      arg << "'" << std::endl;
    return std::nullopt;
  }
  return value;
}

inline
std::optional<bool>
parse_bool_arg(const std::string & arg, const char * error_str)
{
  if (arg.empty()) {
    std::cerr << "error: Failed to parse " << error_str << " argument. " <<
      "Must be followed by true|false" << std::endl;
    return std::nullopt;
  }
  bool value;
  std::istringstream iss(arg);
  iss >> std::boolalpha >> value;
  if (iss.fail()) {
    std::cerr << "error: Failed to parse " << error_str << " argument '" <<
      arg << "'" << std::endl;
    return std::nullopt;
  }
  return value;
}
}  // namespace detail

/// Parse command line arguments
/**
 * \param args vector of argument strings
 * \return pair containing domain bridge config and status
*/
inline
std::pair<std::optional<DomainBridgeConfig>, int>
process_cmd_line_arguments(const std::vector<std::string> & args)
{
  if (args.size() < 2) {
    std::cerr << "error: Expected YAML config file" << std::endl;
    detail::print_help();
    return std::make_pair(std::nullopt, 1);
  }
  std::optional<size_t> from_domain_id;
  std::optional<size_t> to_domain_id;
  std::optional<bool> wait_for_subscription;
  std::optional<domain_bridge::DomainBridgeOptions::Mode> mode;
  std::vector<std::filesystem::path> yaml_configs;
  std::optional<bool> wait_for_publisher;
  std::optional<bool> auto_remove;

  for (auto it = ++args.cbegin() /*skip executable name*/; it != args.cend(); ++it) {
    const auto & arg = *it;
    if (arg == "-h" || arg == "--help") {
      detail::print_help();
      return std::make_pair(std::nullopt, 0);
    }
    if (arg == "--from") {
      if (from_domain_id) {
        std::cerr << "error: --from option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--from must be followed by a domain id" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      from_domain_id = detail::parse_size_t_arg(*it, "FROM_DOMAIN_ID");
      if (!from_domain_id) {
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    if (arg == "--to") {
      if (to_domain_id) {
        std::cerr << "error: --to option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--to must be followed by a domain id" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      to_domain_id = detail::parse_size_t_arg(*it, "TO_DOMAIN_ID");
      if (!to_domain_id) {
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    if (arg == "--wait-for-subscription") {
      if (wait_for_subscription) {
        std::cerr << "error: --wait-for-subscription option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--wait-for-subscription must be followed by true|false" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      wait_for_subscription = detail::parse_bool_arg(*it, "--wait-for-subscription");
      if (!wait_for_subscription) {
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    if (arg == "--wait-for-publisher") {
      if (wait_for_subscription) {
        std::cerr << "error: --wait-for-publisher option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--wait-for-publisher must be followed by true|false" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      wait_for_publisher = detail::parse_bool_arg(*it, "--wait-for-publisher");
      if (!wait_for_publisher) {
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    if (arg == "--mode") {
      if (mode) {
        std::cerr << "error: --mode option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--mode must be followed by compress|decompress|normal" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      const auto & mode_str = *it;
      if (mode_str == detail::kCompressModeStr) {
        mode = domain_bridge::DomainBridgeOptions::Mode::Compress;
      } else if (mode_str == detail::kDecompressModeStr) {
        mode = domain_bridge::DomainBridgeOptions::Mode::Decompress;
      } else if (mode_str == detail::kNormalModeStr) {
        mode = domain_bridge::DomainBridgeOptions::Mode::Normal;
      } else {
        std::cerr << "error: Invalid '--mode' option '" <<
          mode_str << "'" << std::endl;
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    if (arg == "--auto-remove") {
      if (auto_remove) {
        std::cerr << "error: --auto-remove option passed more than once" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      ++it;
      if (it == args.cend()) {
        std::cerr << "--auto-remove must be followed by true|false" << std::endl;
        detail::print_help();
        return std::make_pair(std::nullopt, 1);
      }
      auto_remove = detail::parse_bool_arg(*it, "--auto-remove");
      if (!auto_remove) {
        return std::make_pair(std::nullopt, 1);
      }
      continue;
    }
    yaml_configs.push_back(std::filesystem::path{*it});
  }
  if (yaml_configs.empty()) {
    std::cerr << "error: Must specify at least one yaml configuration file" << std::endl;
    return std::make_pair(std::nullopt, 1);
  }
  domain_bridge::DomainBridgeConfig domain_bridge_config =
    domain_bridge::parse_domain_bridge_yaml_configs(yaml_configs);

  // Override domain bridge configuration options
  if (
    from_domain_id || to_domain_id || wait_for_subscription || wait_for_publisher ||
    auto_remove)
  {
    for (auto & topic_option_pair : domain_bridge_config.topics) {
      if (from_domain_id) {
        topic_option_pair.first.from_domain_id = *from_domain_id;
      }
      if (to_domain_id) {
        topic_option_pair.first.to_domain_id = *to_domain_id;
      }
      if (wait_for_subscription) {
        topic_option_pair.second.wait_for_subscription(*wait_for_subscription);
      }
      if (wait_for_publisher) {
        topic_option_pair.second.wait_for_publisher(*wait_for_publisher);
      }
      if (auto_remove) {
        auto auto_remove_enum = TopicBridgeOptions::AutoRemove::Disabled;
        if (*auto_remove) {
          bool actual_wait_for_publisher = topic_option_pair.second.wait_for_publisher();
          bool actual_wait_for_subscription = topic_option_pair.second.wait_for_subscription();
          if (actual_wait_for_publisher && actual_wait_for_subscription) {
            auto_remove_enum = TopicBridgeOptions::AutoRemove::OnNoPublisherOrSubscription;
          } else if (actual_wait_for_publisher) {
            auto_remove_enum = TopicBridgeOptions::AutoRemove::OnNoPublisher;
          } else if (actual_wait_for_subscription) {
            auto_remove_enum = TopicBridgeOptions::AutoRemove::OnNoSubscription;
          }
        }
        topic_option_pair.second.auto_remove(auto_remove_enum);
      }
    }
  }
  if (mode) {
    domain_bridge_config.options.mode(*mode);
  }
  return std::make_pair(domain_bridge_config, 0);
}
}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__PROCESS_CMD_LINE_ARGUMENTS_HPP_
