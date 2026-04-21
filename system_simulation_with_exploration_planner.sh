#!/bin/bash
# export __NV_PRIME_RENDER_OFFLOAD=0
# export __GLX_VENDOR_LIBRARY_NAME=mesa


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR
source ./install/setup.bash
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3 
ros2 launch vehicle_simulator system_simulation_with_exploration_planner.launch &
sleep 1
ros2 run rviz2 rviz2 -d src/exploration_planner/tare_planner/rviz/tare_planner_ground.rviz