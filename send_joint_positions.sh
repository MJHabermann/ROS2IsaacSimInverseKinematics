#!/bin/bash

# Send joint positions to ROS2 topic for Forward Kinematics testing
# Usage: ./send_joint_positions.sh [j1] [j2] [j3] [j4] [j5] [j6] [j7]
# Or use defaults by running: ./send_joint_positions.sh

# Default joint positions if no args provided
J1=${1:-0.0}
J2=${2:-0.0}
J3=${3:-0.0}
J4=${4:-0.0}
J5=${5:-0.0}
J6=${6:-0.0}
J7=${7:-0.0}

echo "Publishing joint positions to /PaperRollBot/Joint_Positions"
echo "Joint positions: [$J1, $J2, $J3, $J4, $J5, $J6, $J7]"

ros2 topic pub /PaperRollBot/Joint_Positions example_interfaces/msg/Float64MultiArray \
  "data: [$J1, $J2, $J3, $J4, $J5, $J6, $J7]" --once
