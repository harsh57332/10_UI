#!/bin/bash
source /opt/ros/noetic/setup.bash
source ~/lio_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
pgrep -f rosmaster >/dev/null || (roscore > ~/lio_ws/roscore.log 2>&1 &)
sleep 4
roslaunch livox_ros_driver2 msg_MID360.launch > ~/lio_ws/livox.log 2>&1 &
sleep 4
roslaunch fast_lio mapping_mid360.launch rviz:=false > ~/lio_ws/fastlio.log 2>&1 &
sleep 4
python3 ~/lio_ws/pose_server.py > ~/lio_ws/pose_server.log 2>&1 &
wait
