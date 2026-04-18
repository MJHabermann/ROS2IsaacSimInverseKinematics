#!/bin/bash

# Send target pose coordinates to ROS2 topic
# Usage: ./send_target_pose.sh [x] [y] [z] [qw] [qx] [qy] [qz]
# Or use defaults by running: ./send_target_pose.sh

# Default target pose if no args provided (x, y, z, qw, qx, qy, qz)
X=${1:-0.5}
Y=${2:-0.0}
Z=${3:-0.5}
QW=${4:-1.0}
QX=${5:-0.0}
QY=${6:-0.0}
QZ=${7:-0.0}

echo "Publishing target pose to /PaperRollBot/Joint_Pose"
echo "Position: [$X, $Y, $Z]"
echo "Quaternion: [$QW, $QX, $QY, $QZ]"

ros2 topic pub /PaperRollBot/Joint_Pose example_interfaces/msg/Float64MultiArray \
  "data: [$X, $Y, $Z, $QW, $QX, $QY, $QZ]" --once
