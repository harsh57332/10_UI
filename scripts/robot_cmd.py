"""Send one command to the Agent, wait for it, print the result and a short state line.
Usage: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.robot_cmd <action> '<json params>'"""
import json
import sys

from server.robot_link import AgentLink


def main():
    link = AgentLink()
    link.connect(link.cfg["robot"]["host"])
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(link.send(sys.argv[1], params, wait=True, wait_timeout_s=120)))
    s = link.state()
    print("now:", s["mode"], s["fsm_id"], "| pelvis rpy", s["imu"]["pelvis_rpy"], "| torso rpy", s["imu"]["torso_rpy"],
          "| arms", {k: s["arms"][k] for k in ("engaged", "moving", "loop_hz")})


if __name__ == "__main__":
    main()
