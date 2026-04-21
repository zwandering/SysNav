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

#ifndef DOMAIN_BRIDGE__WAIT_FOR_GRAPH_EVENTS_HPP_
#define DOMAIN_BRIDGE__WAIT_FOR_GRAPH_EVENTS_HPP_

// Silly cpplint thinks these are C system headers
#include <optional>
#include <variant>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/client.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/qos.hpp"

namespace domain_bridge
{

/// QoS match information.
/**
 * Used as a return value for QoS matching functions.
 */
struct QosMatchInfo
{
  /// A matching QoS
  rclcpp::QoS qos{10};  // the depth value should be overridden later

  /// Any warning messages about the matching QoS
  std::vector<std::string> warnings;
};

/// Listens for publishers on the ROS graph and provides best matching QoS for them.
/**
 * To start listening on a particular topic, register a callback with
 * `register_on_publisher_qos_ready_callback()`.
 * The registered callback will be called when one or more publishers are available.
 * The callback is automatically unregistered after it is called once, or when this class
 * object is destroyed.
 *
 * This is a RAII class.
 */
class WaitForGraphEvents
{
public:
  WaitForGraphEvents() = default;

  ~WaitForGraphEvents()
  {
    for (auto & t : waiting_threads_) {
      // Notify and join
      {
        std::lock_guard guard{t.second.mutex};
        t.second.shutting_down = true;
      }
      t.second.cv.notify_all();
      t.second.thread.join();
    }
  }

  /// Deleted copy constructor.
  explicit WaitForGraphEvents(const WaitForGraphEvents & other) = delete;

  /// Deleted assignment operator.
  WaitForGraphEvents &
  operator=(const WaitForGraphEvents & other) = delete;

  /// Register a callback that is called when a service server is ready.
  /**
   * \param client: The client waiting for a matching server.
   * \param node: The node to use to monitor the topic.
   * \param callback: User callback that is triggered when a matching server is found.
   */
  void register_on_server_ready_callback(
    rclcpp::ClientBase::SharedPtr client,
    const rclcpp::Node::SharedPtr & node,
    std::function<void()> callback)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it_emplaced_pair = waiting_threads_.try_emplace(node);
    auto & t = it_emplaced_pair.first->second;
    {
      std::lock_guard<std::mutex> lock(t.mutex);
      t.clients_callback_vec.push_back({client, callback});
    }
    // If we already have a thread for this node, then notify that there is a new callback
    if (!it_emplaced_pair.second) {
      t.cv.notify_all();
      return;
    }
    // If we made it this far, there doesn't exist a thread for waiting so we'll create one
    t.thread = this->launch_thread(node, t);
  }

  /// Register a callback that is called when QoS is ready for one or more publishers.
  /**
   * \param topic: The name of the topic to monitor.
   * \param node: The node to use to monitor the topic.
   * \param callback: User callback that is triggered when QoS settings can be determined
   *   for the provided topic name.
   *   If QoS settings are already available, then the callback is called immediately.
   */
  void register_on_publisher_qos_ready_callback(
    const std::string & topic,
    const rclcpp::Node::SharedPtr & node,
    std::function<void(const QosMatchInfo &)> callback)
  {
    return this->register_on_pubsub_callback(topic, node, callback, true);
  }

  /// Register a callback that is called when a subscription is ready.
  /**
   * \param topic: The name of the topic to monitor.
   * \param node: The node to use to monitor the topic.
   * \param callback: User callback that is triggered when a subscription is found.
   */
  void register_on_subscription_qos_ready_callback(
    const std::string & topic,
    const rclcpp::Node::SharedPtr & node,
    std::function<void(const QosMatchInfo &)> callback)
  {
    return this->register_on_pubsub_callback(topic, node, callback, false);
  }

  /// Register a callback that is called when QoS is ready for one or more publishers.
  /**
   * \param topic: The name of the topic to monitor.
   * \param node: The node to use to monitor the topic.
   * \param callback: User callback that is triggered when there's no more publishers
   *  available in that topic.
   */
  void register_on_no_publisher_callback(
    const std::string & topic,
    const rclcpp::Node::SharedPtr & node,
    std::function<void()> callback)
  {
    return this->register_on_pubsub_callback(topic, node, callback, true);
  }

  /// Register a callback that is called when a subscription is ready.
  /**
   * \param topic: The name of the topic to monitor.
   * \param node: The node to use to monitor the topic.
   * \param callback: User callback that is triggered when there's no more subscriptions
   *  available in that topic.
   */
  void register_on_no_subscription_callback(
    const std::string & topic,
    const rclcpp::Node::SharedPtr & node,
    std::function<void()> callback)
  {
    return this->register_on_pubsub_callback(topic, node, callback, false);
  }

  void set_delay(const std::chrono::milliseconds & delay)
  {
    delay_ = delay;
  }

private:
  struct TopicAndCallback
  {
    std::string topic;
    std::variant<
      std::function<void(const QosMatchInfo &)>,  // used to wait for discovered entity
      std::function<void()>>  // used to wait for no entity available in topic
    cb;
    bool is_publisher;
  };
  struct ClientAndCallback
  {
    rclcpp::ClientBase::SharedPtr client;
    std::function<void()> cb;
  };
  struct ThreadMapValue
  {
    std::thread thread;
    std::condition_variable cv;
    std::mutex mutex;
    std::vector<TopicAndCallback> topics_callback_vec;
    std::vector<ClientAndCallback> clients_callback_vec;
    bool shutting_down = false;
  };
  using ThreadMap = std::unordered_map<
    std::shared_ptr<rclcpp::Node>,
    ThreadMapValue>;

  /// Get QoS settings that best match all available publishers.
  /**
   * Queries QoS of existing publishers and returns QoS that is compatibile
   * with the majority of publishers.
   * If possible, the returned QoS will exactly match those of the publishers.
   *
   * If the existing publishers do not have matching QoS settings, then a warning is set in the
   * return struct.
   *
   * If there are no publishers, then no QoS is returned (i.e. the optional object is not set).
   */
  std::optional<QosMatchInfo> get_topic_qos(
    const std::string & topic, rclcpp::Node & node, bool is_publisher = true) const
  {
    // TODO(jacobperron): replace this with common implementation when it is available.
    //       See: https://github.com/ros2/rosbag2/issues/601

    // This implementation is inspired by rosbag2_transport:
    // https://github.com/ros2/rosbag2/blob/master/rosbag2_transport/src/rosbag2_transport/qos.cpp

    // Wait to allow publishers to become available
    std::this_thread::sleep_for(delay_);

    // Query QoS info for publishers
    std::vector<rclcpp::TopicEndpointInfo> endpoint_info_vec;
    if (is_publisher) {
      endpoint_info_vec = node.get_publishers_info_by_topic(topic);
    } else {
      endpoint_info_vec = node.get_subscriptions_info_by_topic(topic);
    }
    std::size_t num_endpoints = endpoint_info_vec.size();

    // If there are no publishers, return an empty optional
    if (num_endpoints < 1u) {
      return {};
    }

    // Initialize QoS
    QosMatchInfo result_qos;
    // Default reliability and durability to value of first endpoint
    result_qos.qos.reliability(endpoint_info_vec[0].qos_profile().reliability());
    result_qos.qos.durability(endpoint_info_vec[0].qos_profile().durability());
    // Always use automatic liveliness
    result_qos.qos.liveliness(rclcpp::LivelinessPolicy::Automatic);

    // Reliability and durability policies can cause trouble with enpoint matching
    // Count number of "reliable" publishers and number of "transient local" publishers
    std::size_t reliable_count = 0u;
    std::size_t transient_local_count = 0u;
    // For duration-based policies, note the largest value to ensure matching all publishers
    rclcpp::Duration max_deadline(0, 0u);
    rclcpp::Duration max_lifespan(0, 0u);
    for (const auto & info : endpoint_info_vec) {
      const auto & profile = info.qos_profile();
      if (profile.reliability() == rclcpp::ReliabilityPolicy::Reliable) {
        reliable_count++;
      }
      if (profile.durability() == rclcpp::DurabilityPolicy::TransientLocal) {
        transient_local_count++;
      }
      if (profile.deadline() > max_deadline) {
        max_deadline = profile.deadline();
      }
      if (profile.lifespan() > max_lifespan) {
        max_lifespan = profile.lifespan();
      }
    }

    // If not all publishers have a "reliable" policy, then use a "best effort" policy
    // and print a warning
    if (reliable_count > 0u && reliable_count != num_endpoints) {
      result_qos.qos.best_effort();
      std::string warning = "Some, but not all, publishers on topic '" + topic +
        "' on domain ID " + std::to_string(node.get_node_options().context()->get_domain_id()) +
        " offer 'reliable' reliability. Falling back to 'best effort' reliability in order "
        "to connect to all publishers.";
      result_qos.warnings.push_back(warning);
    }

    // If not all publishers have a "transient local" policy, then use a "volatile" policy
    // and print a warning
    if (transient_local_count > 0u && transient_local_count != num_endpoints) {
      result_qos.qos.durability_volatile();
      std::string warning = "Some, but not all, publishers on topic '" + topic +
        "' on domain ID " + std::to_string(node.get_node_options().context()->get_domain_id()) +
        " offer 'transient local' durability. Falling back to 'volatile' durability in order "
        "to connect to all publishers.";
      result_qos.warnings.push_back(warning);
    }

    result_qos.qos.deadline(max_deadline);
    result_qos.qos.lifespan(max_lifespan);

    return result_qos;
  }

  std::thread
  launch_thread(rclcpp::Node::SharedPtr node, ThreadMapValue & t)
  {
    auto invoke_callback_when_qos_ready = [
      this,
      node = std::move(node),
      &t]()
      {
        auto event = node->get_graph_event();
        while (true) {
          // Wait for graph event
          // Note, in case new publishers don't trigger a graph event we add a
          // timeout so that we can still poll periodically for new publishers
          node->wait_for_graph_change(event, std::chrono::seconds(1));
          event->check_and_clear();

          {
            std::unique_lock<std::mutex> lock(t.mutex);
            // If we're shutting down, exit the thread
            if (t.shutting_down) {
              return;
            }
            {
              // Check if a matching service server was found
              auto it = t.clients_callback_vec.begin();
              while (it != t.clients_callback_vec.end()) {
                if (it->client->service_is_ready()) {
                  it->cb();
                  it = t.clients_callback_vec.erase(it);
                } else {
                  ++it;
                }
              }
            }
            {
              // Check if QoS is ready for any of the topics
              auto it = t.topics_callback_vec.begin();
              while (it != t.topics_callback_vec.end()) {
                const std::string & topic = it->topic;
                const auto & callback_variant = it->cb;
                bool is_publisher = it->is_publisher;
                std::optional<QosMatchInfo> opt_qos;
                try {
                  opt_qos = this->get_topic_qos(topic, *node, is_publisher);
                } catch (const rclcpp::exceptions::RCLError & ex) {
                  // If the context was shutdown, then exit cleanly
                  // This can happen if we get a SIGINT
                  const auto context = node->get_node_options().context();
                  if (!context->is_valid()) {
                    return;
                  }
                  auto entity_type = is_publisher ? "publisher" : "subscription";
                  // Otherwise, don't crash if there was a hiccup querying the topic endpoint
                  // Log an error instead
                  std::cerr << "Failed to query " << entity_type << " info for topic '" <<
                    topic << "': " << ex.what() << std::endl;
                }
                if (std::holds_alternative<std::function<void()>>(callback_variant)) {
                  // waiting until no entity is available
                  if (!opt_qos) {
                    auto callback = std::get<std::function<void()>>(callback_variant);
                    callback();
                  }
                  ++it;
                } else {
                  // waiting for discovered entity
                  if (opt_qos) {
                    auto callback = std::get<std::function<void(const QosMatchInfo &)>>(
                      callback_variant);
                    const QosMatchInfo & qos = opt_qos.value();
                    callback(qos);
                  }
                  ++it;
                }
              }
            }

            // It's only worth continuing if we have callbacks to handle
            // or we're shutting down
            t.cv.wait(
              lock,
              [&t]
              {
                return (t.topics_callback_vec.size() > 0u) ||
                (t.clients_callback_vec.size() > 0u) ||
                t.shutting_down;
              });
          }
        }
      };
    return std::thread(invoke_callback_when_qos_ready);
  }

  // helper
  template<class>
  static inline constexpr bool always_false_v = false;

  template<typename CallbackT>
  void register_on_pubsub_callback(
    const std::string & topic,
    const rclcpp::Node::SharedPtr & node,
    CallbackT callback,
    bool is_publisher)
  {
    // If the QoS is already available, trigger the callback immediately
    auto opt_qos = get_topic_qos(topic, *node, is_publisher);
    using T = std::decay_t<CallbackT>;

    if constexpr (std::is_same_v<T, std::function<void(const QosMatchInfo &)>>) {
      // notify when entity discovered
      if (opt_qos) {
        const QosMatchInfo & qos = opt_qos.value();
        callback(qos);
      }
    } else if constexpr (std::is_same_v<T, std::function<void()>>) {  // NOLINT, cpplint false positive
      // notify when no entity is available
      if (!opt_qos) {
        callback();
      }
    } else {
      static_assert(
        always_false_v<T>,
        "callback should either be std::function<void(const QosMatchInfo)>"
        " or std::function<void()>");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it_emplaced_pair = waiting_threads_.try_emplace(node);
    auto & t = it_emplaced_pair.first->second;
    {
      std::lock_guard<std::mutex> lock(t.mutex);
      t.topics_callback_vec.push_back({topic, callback, is_publisher});
    }
    // If we already have a thread for this node, then notify that there is a new callback
    if (!it_emplaced_pair.second) {
      t.cv.notify_all();
      return;
    }
    // If we made it this far, there doesn't exist a thread for waiting so we'll create one
    t.thread = this->launch_thread(node, t);
  }

  /// Threads used for waiting on publishers to become available
  ThreadMap waiting_threads_;

  /// Mutex for waiting_threads_
  std::mutex mutex_;

  /// Amount of time to wait for publishers before getting QoS
  std::chrono::milliseconds delay_;
};  // class WaitForGraphEvents

}  // namespace domain_bridge

#endif  // DOMAIN_BRIDGE__WAIT_FOR_GRAPH_EVENTS_HPP_
