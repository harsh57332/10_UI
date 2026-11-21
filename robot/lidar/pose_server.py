#!/usr/bin/env python3
"""The latest FAST-LIO pose for the PC: GET http://<robot>:8210/pose -> {"t", "p": [x,y,z], "q": [x,y,z,w], "age_s"}."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rospy
from nav_msgs.msg import Odometry
latest = {}
lock = threading.Lock()
def cb(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    with lock:
        latest.update(t=m.header.stamp.to_sec(), p=[p.x, p.y, p.z], q=[q.x, q.y, q.z, q.w], rx=time.time())
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        with lock:
            d = dict(latest)
        if d:
            d["age_s"] = round(time.time() - d.pop("rx"), 3)
        b = json.dumps(d).encode()
        self.send_response(200 if d else 503); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a):
        pass
rospy.init_node("lio_pose_server", disable_signals=True)
rospy.Subscriber("/Odometry", Odometry, cb, queue_size=1)
ThreadingHTTPServer(("0.0.0.0", 8210), H).serve_forever()
