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

#ifndef DOMAIN_BRIDGE__PARSE_DOMAIN_BRIDGE_YAML_CONFIG_HPP_
#define DOMAIN_BRIDGE__PARSE_DOMAIN_BRIDGE_YAML_CONFIG_HPP_

#include <filesystem>
#include <string>
#include <vector>

#include "domain_bridge/domain_bridge_config.hpp"
#include "domain_bridge/visibility_control.hpp"

namespace domain_bridge
{

class YamlParsingError : public std::runtime_error
{
public:
  DOMAIN_BRIDGE_PUBLIC
  YamlParsingError(const std::filesystem::path file_path, const std::string message)
  : std::runtime_error("error parsing the file '" + file_path.string() + "': " + message)
  {}
};

/// Create a DomainBridgeConfig object from a YAML file.
/**
 * The YAML file may contain the following optional keys:
 *
 * - name: Name of the domain bridge
 * - from_domain: The default 'from_domain' used for bridged topics.
 *     If omitted, then each topic bridge must specify its own value.
 * - to_domain: The default 'to_domain' used for bridged topics
 *     If omitted, then each topic bridge must specify it's own value.
 * - topics: A map of topic names to a map of topic bridge information
 *
 * For each topic bridge, the following keys are valid:
 *
 * - type: Message type to bridge
 * - from_domain: Subscribe to the topic on this domain ID
 * - to_domain: Publish to the topic on this domain ID
 * - qos: QoS options.
 * - remap: Remap topic name to this name in 'to_domain'.
 *
 * QoS options are a map of policy kinds to policy values.
 * If an option for a policy is not provided then a default value is used.
 * For each QoS option provided, it will override the default policy.
 *
 * Supported QoS options:
 *
 * - reliability: must be 'reliable' or 'best_effort'
 * - durability: must be 'volatile' or 'transient_local'
 * - history: must be 'keep_last' or 'keep_all'
 * - depth: a positive integer
 * - deadline: an integer representing number of nanoseconds (negative for "infinite"),
 *             or the string 'auto' to automatically match the value of a publisher.
 * - lifespan: an integer representing number of nanoseconds (negative for "infinite"),
 *             or the string 'auto' to automatically match the value of a publisher.
 *
 * \see QosOptions
 *
 * For example,
 *
 * ```yaml
 * name: my_bridge
 * from_domain: 2
 * to_domain: 3
 * topics:
 *   chatter:
 *     type: example_interfaces/msg/String
 *   clock:
 *     type: rosgraph_msgs/msg/Clock
 *     # Override the default 'to_domain' above
 *     to_domain: 4
 *   foo/bar:
 *     type: example_interfaces/msg/Empty
 *     # Override reliability, history, and depth
 *     # Let durability policy be automatically determined
 *     qos:
 *       reliability: best_effort
 *       history: keep_last
 *       depth: 42
 *   chitter:
 *     type: example_interfaces/msg/String
 *     # Remap 'chitter' topic to 'chatter'
 *     remap: chatter
 * ```
 *
 * \param file_path: Path to the YAML file.
 * \return A DomainBridgeConfig object populated according to the input YAML file.
 * \throws domain_bridge::YamlParsingError if there was an error parsing the YAML file,
 *   for example, if a required key is missing.
 */
DOMAIN_BRIDGE_PUBLIC
DomainBridgeConfig parse_domain_bridge_yaml_config(std::filesystem::path file_path);

/// Create a DomainBridgeConfig object from some YAML files.
/**
 * If the passed files have overlapping settings, the one specified in a later file will apply.
 *
 * See \ref parse_domain_bridge_yaml_config() for more details.
 */
DOMAIN_BRIDGE_PUBLIC
DomainBridgeConfig
parse_domain_bridge_yaml_configs(const std::vector<std::filesystem::path> & file_paths);

/// Updates domain bridge config with a yaml file.
/**
 * Updates the contents of a `DomainBridgeConfig` object with the settings specified in the yaml file.
 * Settings not specified in the yaml file are not modified.
 *
 * See \ref parse_domain_bridge_yaml_config() for more details.
 */
DOMAIN_BRIDGE_PUBLIC
void
update_domain_bridge_config_from_yaml(
  std::filesystem::path file_path,
  DomainBridgeConfig & domain_bridge_config);

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__PARSE_DOMAIN_BRIDGE_YAML_CONFIG_HPP_
