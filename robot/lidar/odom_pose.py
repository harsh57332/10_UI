#!/usr/bin/env python3
"""Print the latest FAST-LIO pose (/Odometry) as JSON: position (m) and orientation quaternion in its start frame."""
import json, rospy
from nav_msgs.msg import Odometry
rospy.init_node("odom_pose_once", anonymous=True, disable_signals=True)
m = rospy.wait_for_message("/Odometry", Odometry, timeout=3.0)
p, q = m.pose.pose.position, m.pose.pose.orientation
print(json.dumps({"t": m.header.stamp.to_sec(), "p": [p.x, p.y, p.z], "q": [q.x, q.y, q.z, q.w]}))
