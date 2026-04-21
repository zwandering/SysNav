#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR
source ./install/setup.bash
ros2 launch vehicle_simulator system_real_robot_with_exploration_planner.launch &
sleep 1
ros2 run rviz2 rviz2 -d src/exploration_planner/tare_planner/rviz/tare_planner_ground.rviz
