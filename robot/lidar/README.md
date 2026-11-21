# Head LiDAR odometry (Livox MID-360 + FAST-LIO2) on the robot

How far the robot has really walked, used by `server/lidar_odom.py` and `skills._walk_measured` (the last steps to a
place target). Tape-checked 2026-11-21: within 1.1 cm over 0.3-0.5 m walks (left toe tip).

The MID-360 is in the head, **mounted upside down** (URDF `mid360_joint`, roll 180 deg, pitch 2.9 deg), so it looks
down and around and also sees the robot's own legs, arms and a held box: `fast_lio_g1.patch` drops every point within
`preprocess/self_radius` (0.6 m) of the LiDAR's axis.

## Install (robot, ROS Noetic, once)

```bash
mkdir -p ~/lio_ws/src && cd ~/lio_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git && (cd livox_ros_driver2 && git checkout 1.2.4)
#   1.2.4: the newest version that builds against the Livox-SDK2 installed in /usr/local (newer needs Mid360s/DoubleEcho)
cp livox_ros_driver2/package_ROS1.xml livox_ros_driver2/package.xml
git clone https://github.com/hku-mars/FAST_LIO.git && (cd FAST_LIO && git submodule update --init && git apply <repo>/robot/lidar/fast_lio_g1.patch)
cp <repo>/robot/lidar/MID360_config.json livox_ros_driver2/config/        # lidar 192.168.123.120, robot 192.168.123.164
cp <repo>/robot/lidar/fast_lio_mid360.yaml FAST_LIO/config/mid360.yaml
cp <repo>/robot/lidar/{run_lio.sh,pose_server.py,odom_pose.py} ~/lio_ws/
source /opt/ros/noetic/setup.bash && cd ~/lio_ws && catkin_make -DROS_EDITION=ROS1 -DCMAKE_BUILD_TYPE=Release -j6
```

## Run

`~/lio_ws/run_lio.sh` (roscore, the Livox driver, FAST-LIO, `pose_server.py` serving `GET :8210/pose`). Started by hand
so far (`setsid nohup ~/lio_ws/run_lio.sh &`): **not yet a boot service** - after a reboot start it again. Without it the
walks fall back to timed steps.
