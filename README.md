<div align="center">

# SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation

[Haokun Zhu](https://github.com/igzat1no)\*,
[Zongtai Li]()\*,
[Zihan Liu](),
[Kevin Guo](),
[Zhengzhi Lin](),
[Yuxin Cai](),
[Guofei Chen](),
[Chen Lv](),
[Wenshan Wang](),
[Jean Oh](),
[Ji Zhang](https://frc.ri.cmu.edu/~zhangji)

Carnegie Mellon University, New York University, Nanyang Technological University

[[Project Page](https://cmu-vln.github.io/)] [[arXiv](https://arxiv.org/abs/2603.06914)]

<img src="img/teaser.jpg" width="100%"/>

</div>

## News

- **[2026-03]** Paper released on [arXiv](https://arxiv.org/abs/2603.06914).
- **[2026-03]** [Project page](https://cmu-vln.github.io/) is online.
- **[2026-04]** Code released for Unity simulation, wheeled robot, Unitree Go2, and Unitree G1 platforms.

## Abstract

Object navigation in real-world environments remains a significant challenge in embodied AI. We present **SysNav**, a three-level object navigation system that decouples semantic reasoning, navigation planning, and motion control. The framework employs Vision-Language Models for high-level semantic guidance and implements a hierarchical room-based navigation strategy that treats rooms as minimal decision-making units, combined with classical exploration for in-room navigation. Through 190 real-world experiments across three robot embodiments (wheeled, quadruped, humanoid), we demonstrate 4-5x improvement in navigation efficiency over existing baselines. The system also achieves state-of-the-art results on HM3D-v1, HM3D-v2, MP3D, and HM3D-OVON simulation benchmarks.

## Demo

### Long-range Object Navigation

<table>
<tr>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=FpF6IATXWds">
<img src="https://img.youtube.com/vi/FpF6IATXWds/maxresdefault.jpg" alt="Find Refrigerator in Lounge" width="100%"/>
</a>
<br><b>Find Refrigerator<br>in Lounge.</b>
<br><a href="https://www.youtube.com/watch?v=FpF6IATXWds">&#9654; Watch on YouTube</a>
</td>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=GqRUvwAEqc8">
<img src="https://img.youtube.com/vi/GqRUvwAEqc8/maxresdefault.jpg" alt="Find Blue Trash Can in Classroom" width="100%"/>
</a>
<br><b>Find Blue Trash Can<br>in Classroom.</b>
<br><a href="https://www.youtube.com/watch?v=GqRUvwAEqc8">&#9654; Watch on YouTube</a>
</td>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=A78TSwI78iM">
<img src="https://img.youtube.com/vi/A78TSwI78iM/maxresdefault.jpg" alt="Find Microwave Oven near Refrigerator" width="100%"/>
</a>
<br><b>Find Microwave Oven<br>near Refrigerator.</b>
<br><a href="https://www.youtube.com/watch?v=A78TSwI78iM">&#9654; Watch on YouTube</a>
</td>
</tr>
</table>

### Cross-Embodiment Object Navigation

<table>
<tr>
<th></th>
<th align="center">System View</th>
<th align="center">Third-person View</th>
</tr>
<tr>
<td rowspan="2" align="center" width="12%"><b>Wheeled<br>Robot</b></td>
<td width="44%">

[.webm](https://github.com/user-attachments/assets/bd7cec26-9198-401a-8ff6-d7f3f8f6f093)

</td>
<td width="44%">

[.webm](https://github.com/user-attachments/assets/8821366c-b439-4661-8802-200c9259f933)

</td>
</tr>
<tr>
<td colspan="2" align="center"><em>Find the microwave_oven.</em></td>
</tr>
<tr>
<td rowspan="2" align="center"><b>Quadruped<br>(Go2)</b></td>
<td>

[.webm](https://github.com/user-attachments/assets/428ad7a1-82f8-4f3b-88bd-6ab620c707ea)

</td>
<td>

[.webm](https://github.com/user-attachments/assets/09abb740-46ce-405a-8922-697fc074fcaf)

</td>
</tr>
<tr>
<td colspan="2" align="center"><em>Find the blue trash_can.</em></td>
</tr>
<tr>
<td rowspan="2" align="center"><b>Humanoid<br>(G1)</b></td>
<td>

[.webm](https://github.com/user-attachments/assets/b235e61d-f2b7-4300-b982-33567c5fa880)

</td>
<td>

[.webm](https://github.com/user-attachments/assets/ad1b03a0-b41c-493c-a549-cff6a232cac1)

</td>
</tr>
<tr>
<td colspan="2" align="center"><em>Find the tv_monitor on the black desk.</em></td>
</tr>
</table>

<p align="center"><em>More demos on our <a href="https://cmu-vln.github.io/">project page</a>.</em></p>

## Contents

- [Demo](#demo)
- [Installation](#installation)
  - [Dependencies](#1-dependencies)
  - [Submodules and Python Packages](#2-submodules-and-python-packages)
  - [SLAM Dependencies](#3-slam-dependencies)
  - [Mid-360 Lidar Driver](#4-mid-360-lidar-driver)
  - [Compile](#5-compile)
- [Simulation Setup](#simulation-setup)
  - [Base Autonomy](#base-autonomy)
  - [Exploration Planner](#exploration-planner)
- [Real-robot Setup](#real-robot-setup)
  - [Hardware](#hardware)
  - [System Setup](#system-setup)
  - [360 Camera Driver](#360-camera-driver)
  - [System Usage](#system-usage)
- [Bagfile Setup](#bagfile-setup)
- [Credits](#credits)
- [Citation](#citation)
- [License](#license)

## Installation

The system has been tested on **Ubuntu 24.04** with **ROS2 Jazzy**.

### 1) Dependencies

Install [ROS2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html), then:
```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Install system dependencies:
```bash
sudo apt update
sudo apt install ros-jazzy-desktop-full ros-jazzy-pcl-ros libpcl-dev git
sudo apt install -y nlohmann-json3-dev
sudo apt install ros-jazzy-backward-ros
```

### 2) Submodules and Python Packages

```bash
git submodule update --init --recursive

pip install -r requirement.txt --break-system-package

# detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --break-system-package

# pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation --break-system-package

# sam2
cd src/semantic_mapping/semantic_mapping/external/sam2
pip install -e . --break-system-package
cd checkpoints && ./download_ckpts.sh && cd ../..

# spacy
python -m spacy download en_core_web_sm --break-system-package

# CLIP
pip install git+https://github.com/ultralytics/CLIP.git --break-system-package

# YOLO models
python set_yolo_e.py
python set_yolo_world.py
```

### 3) SLAM Dependencies

Install **Sophus** (from `src/slam/dependency/Sophus`):
```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF
make && sudo make install
```

Install **Ceres Solver** (from `src/slam/dependency/ceres-solver`):
```bash
mkdir build && cd build
cmake ..
make -j6 && sudo make install
```

Install **GTSAM** (from `src/slam/dependency/gtsam`):
```bash
mkdir build && cd build
cmake .. -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
make -j6 && sudo make install
sudo /sbin/ldconfig -v
```

### 4) Mid-360 Lidar Driver

Install **Livox-SDK2** (from `src/utilities/livox_ros_driver2/Livox-SDK2`):
```bash
mkdir build && cd build
cmake ..
make && sudo make install
```

Configure the lidar IP in `src/utilities/livox_ros_driver2/config/MID360_config.json` — set the IP to `192.168.1.1xx` where `xx` are the last two digits of the lidar serial number.

Compile the driver:
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select livox_ros_driver2
```

### 5) Compile

**For simulation** (skips SLAM and lidar driver):
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2
```

**For real robot** (full build, requires steps 3-4):
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Gemini API Key

Go to [Google AI Studio](https://aistudio.google.com/app/api-keys) and generate an API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
Optionally add the line above to your `~/.bashrc` so it persists across terminal sessions.

## Simulation Setup

### Base Autonomy

The system is integrated with [Unity](https://unity.com) environment models for simulation. Download a [Unity environment model](https://drive.google.com/drive/folders/1GNz386h6wiiFuQQdaY2_HbNRyd7nKA1N?usp=sharing) (recommend home_building_1.zip) and unzip the files to the `src/base_autonomy/vehicle_simulator/mesh/unity` folder. For computers without a powerful GPU, please try the `without_360_camera` version for a higher rendering rate.

The environment model files should look like:
```
mesh/
  unity/
    environment/
      Model_Data/
      Model.x86_64
      UnityPlayer.so
      Dimensions.csv
      Categories.csv
    map.ply
    object_list.txt
    traversable_area.ply
    map.jpg
    render.jpg
```

Launch the system:
```bash
./system_simulation.sh
```

After seeing data showing up in RVIZ, users can use the 'Waypoint' button to set waypoints and navigate the vehicle around. The system supports three operating modes:

<p align="center">
  <img src="img/rviz_full.jpg" alt="RVIZ" width="80%"/><br>
  <em>Base autonomy (smart joystick, waypoint, and manual modes)</em>
</p>

- **Smart joystick mode** (default): The vehicle follows joystick commands while avoiding collisions. Use the control panel in RVIZ or the right joystick on the controller.

- **Waypoint mode**: The vehicle follows waypoints while avoiding collisions. Use the 'Waypoint' button in RVIZ, or click 'Resume Navigation to Goal' to switch to this mode.

- **Manual mode**: The vehicle follows joystick commands without collision avoidance. Press the 'manual-mode' button on the controller.

<p align="center">
  <img src="img/rviz_control_panel.jpg" alt="RVIZ Control Panel" width="30%"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="img/ps3_controller.jpg" alt="PS3 Controller" width="45%"/>
</p>

Alternatively, users can run a ROS node to send a series of waypoints:
```bash
source install/setup.sh
ros2 launch waypoint_example waypoint_example.launch
```
Click the 'Resume Navigation to Goal' button in RVIZ, and the vehicle will navigate inside the boundary following the waypoints. More information about the base autonomy system is available on the [Autonomous Exploration Development Environment](https://www.cmu-exploration.com) website.

### Exploration Planner

Launch the system with the exploration planner:
```bash
./system_simulation_with_exploration_planner.sh
```
Click the 'Resume Navigation to Goal' button in RVIZ to start the exploration. Users can adjust the navigation boundary by updating the boundary polygon in `src/exploration_planner/tare_planner/data/boundary.ply`.

> **Note:** On ARM computers, download the corresponding [OR-Tools binary release](https://github.com/google/or-tools/releases) and replace the `include` and `lib` folders under `src/exploration_planner/tare_planner/or-tools`.

<p align="center">
  <img src="img/rviz_full_with_exploration_planner.jpg" alt="RVIZ with Exploration Planner" width="80%"/><br>
  <em>Base autonomy with exploration planner</em>
</p>

## Real-robot Setup

### Hardware

The vehicle hardware is designed to support advanced AI. Space is left for users to install a Jetson AGX Orin computer or a gaming laptop. The vehicle is equipped with a 19V and a 110V inverter (both 400W) to power sensors and computers. A wireless HDMI module transmits signals to a control station.

We supply two types of wheels: Mecanum wheels for indoor carpet, and standard wheels for hard floor and outdoors.

<p align="center">
  <img src="img/all_items.jpg" alt="All Items" width="48%"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="img/computer_space.jpg" alt="Computer Space" width="34%"/>
</p>

<p align="center">
  <img src="img/control_station.jpg" alt="Control Station" width="70%"/>
</p>

<p align="center">
  <img src="img/wheel_types.jpg" alt="Wheel Types" width="85%"/>
</p>

### System Setup

Install [Ubuntu 24.04](https://releases.ubuntu.com/noble) and [ROS2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html) on the processing computer. Add user to the dialout group:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo adduser 'username' dialout
sudo reboot now
```

Follow the [Installation](#installation) section to install all dependencies and compile the full repository. For the motor controller, connect it via USB and update the serial device path in `src/base_autonomy/local_planner/launch/local_planner.launch` and `src/utilities/teleop_joy_controller/launch/teleop_joy_controller.launch` if needed (default: `/dev/ttyACM0`).

Test the teleoperation:
```bash
source install/setup.sh
ros2 launch teleop_joy_controller teleop_joy_controller.launch
```

### 360 Camera Driver

The system uses a Ricoh Theta Z1 360-degree camera. The camera driver and lidar-to-camera calibration tools are maintained in a separate repository — clone it alongside this repo and follow its README to build and configure:

[https://github.com/jizhang-cmu/360_camera/tree/jazzy](https://github.com/jizhang-cmu/360_camera/tree/jazzy)

### System Usage

Launch the full system:
```bash
./system_real_robot.sh
```

Launch with the exploration planner:
```bash
./system_real_robot_with_exploration_planner.sh
```

<p align="center">
  <img src="img/exploration.jpg" alt="Exploration" width="80%"/><br>
  <em>Exploration</em>
</p>

## Bagfile Setup

To run the system with a recorded bagfile, open **three terminals**:

**Terminal 1** - Launch the system:
```bash
./system_bagfile.sh
# or with exploration planner:
./system_bagfile_with_exploration_planner.sh
```

**Terminal 2** - Republish camera images:
```bash
ros2 run image_transport republish \
  --ros-args \
  -p in_transport:=compressed \
  -p out_transport:=raw \
  --remap in/compressed:=/camera/image/compressed \
  --remap out:=/camera/image
```

**Terminal 3** - Play the bagfile:
```bash
source install/setup.bash
ros2 bag play bagfolder_path/bagfile_name.mcap
```

Example bagfiles are available [here](https://drive.google.com/drive/folders/1hXBf_A4AS-P2nHnOAXfbH9ezKXbsD6qk?usp=drive_link).

> **Note:** Before processing bagfiles, ensure the repository has been fully compiled following the [Installation](#installation) section.

## Credits

The project is led by [Ji Zhang's](https://frc.ri.cmu.edu/~zhangji) group at Carnegie Mellon University.

The base autonomy system is based on [Autonomous Exploration Development Environment](https://www.cmu-exploration.com). The SLAM module is an upgraded implementation of [LOAM](https://github.com/cuitaixiang/LOAM_NOTED).

## Citation

If you find this work useful, please consider citing:
```bibtex
@article{zhu2026sysnav,
  title={SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation},
  author={Zhu, Haokun and Li, Zongtai and Liu, Zihan and Guo, Kevin and Lin, Zhengzhi and Cai, Yuxin and Chen, Guofei and Lv, Chen and Wang, Wenshan and Oh, Jean and Zhang, Ji},
  journal={arXiv preprint arXiv:2603.06914},
  year={2026}
}
```

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify, and distribute the software for any **noncommercial** purpose (research, education, personal use, government, charitable organizations). Commercial use is not permitted under this license.

Some third-party packages retain their original open-source licenses (BSD, MIT, Apache 2.0, GPLv3). See individual `package.xml` files for per-package license declarations.
