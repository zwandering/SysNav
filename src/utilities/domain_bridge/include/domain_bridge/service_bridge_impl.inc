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

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/client.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/service.hpp"

// This is a non-header file meant to be included in `domain_bridge.hpp`.
// It contains the implementation of the DomainBridge::bridge_service() template method.

// Conditionally include domain_bridge.hpp to help intellisense.
#ifndef DOMAIN_BRIDGE__DOMAIN_BRIDGE_HPP_
# include "domain_bridge/domain_bridge.hpp"
#endif

namespace domain_bridge
{

namespace detail
{
// service_name, from_domain_id, to_domain_id
using ServiceBridge = std::tuple<std::string, size_t, size_t>;

bool
is_bridging_service(
  const DomainBridgeImpl & impl, const ServiceBridge & service_bridge);

void
add_service_bridge(
  DomainBridgeImpl & impl,
  const rclcpp::Node::SharedPtr & node,
  ServiceBridge service_bridge,
  std::function<std::shared_ptr<rclcpp::ServiceBase>()> create_service,
  std::shared_ptr<rclcpp::ClientBase> client);

rclcpp::Node::SharedPtr
get_node_for_domain(DomainBridgeImpl & impl, std::size_t domain_id);

const std::string &
get_node_name(const DomainBridgeImpl & impl);
}  // namespace detail

/// Bridge a service from one domain to another.
/**
 * \param service_bridge: Struct containing info about the service to bridge.
 * \param options: Options for bridging the topic.
 */
template<typename ServiceT>
void
DomainBridge::bridge_service(
  const std::string & service_name,
  size_t from_domain_id,
  size_t to_domain_id,
  const ServiceBridgeOptions & options)
{
  const auto & node_name = detail::get_node_name(*impl_);
  // Validate service name
  const std::string & resolved_service_name = rclcpp::expand_topic_or_service_name(
    service_name, node_name, "/", true);

  // If a remap name is provided, then validate it
  // otherwise "remap" to the same name
  std::string service_remapped = resolved_service_name;
  if (!options.remap_name().empty()) {
    service_remapped = rclcpp::expand_topic_or_service_name(
      options.remap_name(), node_name, "/", true);
  }

  std::tuple<std::string, size_t, size_t> service_bridge = {
    resolved_service_name,
    from_domain_id,
    to_domain_id
  };

  // Check if already bridged
  if (detail::is_bridging_service(*impl_, service_bridge))
  {
    std::cerr << "Service '" << resolved_service_name << "'" << " already bridged from domain " <<
      std::to_string(from_domain_id) << " to domain " << std::to_string(to_domain_id) <<
      ", ignoring" << std::endl;
    return;
  }

  rclcpp::Node::SharedPtr from_domain_node = detail::get_node_for_domain(*impl_, from_domain_id);
  rclcpp::Node::SharedPtr to_domain_node = detail::get_node_for_domain(*impl_, to_domain_id);

  // Create a client for the 'from_domain'
  auto client = from_domain_node->create_client<ServiceT>(
    resolved_service_name,
    rclcpp::ServicesQoS{},
    options.callback_group());

  auto handle_request =
    [client](
      std::shared_ptr<rclcpp::Service<ServiceT>> me,
      std::shared_ptr<rmw_request_id_t> request_header,
      std::shared_ptr<typename ServiceT::Request> request) -> void
    {
      // TODO(ivanpauno): What do we do if the original server is down?
      // This is a general problem in ROS 2, we don't have a clean way of communicating a service error ....
      auto result = client->async_send_request(
        request, [me = std::move(me), request_header](typename rclcpp::Client<ServiceT>::SharedFuture future_response)
      {
        // TODO(ivanpauno): Why do we get a future instead of the response directly here?
        // That doesn't seem to make any sense.
        // See if we can improve rclcpp service API.

        auto response = future_response.get();
        me->send_response(*request_header, *response);
      });
    };

  // A service for the 'to_domain' will be created when we detect there is a service running in
  // the 'from_domain'.
  auto create_service_cb = [
    to_domain_node = std::move(to_domain_node),
    service_remapped = std::move(service_remapped),
    handle_request = std::move(handle_request),
    options = std::move(options)]()
  {
    return to_domain_node->create_service<ServiceT>(
      service_remapped,
      handle_request,
      rclcpp::ServicesQoS{},
      options.callback_group());
  };

  detail::add_service_bridge(
    *impl_, from_domain_node, service_bridge, create_service_cb, client);
}

}  // namespace domain_bridge
