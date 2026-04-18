# domain_bridge

A ROS 2 domain bridge.
Bridges ROS communication between different ROS domain IDs.

See the [design document](doc/design.md) for more details about how the bridge works.

## Prerequisites

- [ROS 2](https://index.ros.org/doc/ros2/Installation) (Galactic or newer)

## Installation

### Ubuntu 20.04

Replace `$ROS_DISTRO` with the ROS codename (e.g. `galactic`):

```
sudo apt install ros-$ROS_DISTRO-domain-bridge
```

### From source

**Prerequisites:**

- Git
- [colcon](https://colcon.readthedocs.io/) (colcon_cmake)

```
mkdir -p domain_bridge_ws/src
cd domain_bridge_ws
git clone https://github.com/ros2/domain_bridge.git src/
colcon build
```

## Usage

### CLI

There is a standalone executable that can be configured via a YAML file.

You must provide the path to a YAML configuration file as the last argument to the executable.
For example,

```sh
ros2 run domain_bridge domain_bridge examples/example_bridge_config.yaml
```

There are also options to override the domain IDs set for all entities in the YAML config,
for example,

```sh
ros2 run domain_bridge domain_bridge --from 7 --to 8 examples/example_bridge_config.yaml
```

Use the `--help` option for more usage information:

```sh
ros2 run domain_bridge domain_bridge --help
```

You can also load composable nodes alongside the domain bridge to allow for intra-process communication.
Use the extra argument `domain_id` to set the domain ID for each loaded component:

```sh
ros2 component load /ComponentManager composition composition::Talker -e domain_id:=1
ros2 component load /ComponentManager composition composition::Listener -e domain_id:=2
```

### Launch

You can also use the example launch script, which takes a required launch argument `config`:

```xml
ros2 launch domain_bridge domain_bridge.launch.xml config:=examples/example_bridge_config.yaml
```

You can also override domain IDs with optional launch arguments `from_domain` and `to_domain`.

To reverse the `from` and `to` domain IDs for a topic, set the `reversed` argument to `true`.

To enable bidirectional bridging of a topic, set the `bidirectional` argument to `true`.

Here is an example of including the domain bridge launch script into your own:

```xml
<launch>
  <include file="$(find-pkg-share domain_bridge)/launch/domain_bridge.launch.xml">
    <arg name="config" value="$(find-pkg-share domain_bridge)/examples/example_bridge_config.yaml" />
    <!-- Optionally override domain IDs -->
    <arg name="from_domain" value="42" />
    <arg name="to_domain" value="43" />
  </include>
</launch>
```

### C++ library

There is a C++ API that can be integrated into your own process, you can find the [API docs here](https://docs.ros.org/en/ros2_packages/rolling/api/domain_bridge/index.html).
