"""List self-colliding link pairs (not ignored) of the planner model for key arm poses.
Run: ~/miniconda3/envs/g1ui_curobo/bin/python -m scripts.check_self_collision"""
import math

import torch
import yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

from common import config


def collisions(k, ignored, q, buffer=None):
    kc = k.config.kinematics_config
    idx2link = {v: n for n, v in kc.link_name_to_idx_map.items()}
    st = k.compute_kinematics(JointState.from_position(torch.tensor([q], device="cuda", dtype=torch.float32),
                                                       joint_names=k.joint_names))
    s = st.robot_spheres.view(-1, 4)
    links = [idx2link.get(int(i), str(int(i))) for i in kc.link_sphere_idx_map.view(-1).tolist()]
    hits, buffer = {}, buffer or {}
    for i in range(s.shape[0]):
        if s[i, 3] <= 0:
            continue
        for j in range(i + 1, s.shape[0]):
            if s[j, 3] <= 0 or links[i] == links[j] or (links[i], links[j]) in ignored:
                continue
            pen = float(s[i, 3] + s[j, 3] + buffer.get(links[i], 0.0) + buffer.get(links[j], 0.0)
                        - torch.linalg.norm(s[i, :3] - s[j, :3]))
            if pen > 0:
                key = tuple(sorted((links[i], links[j])))
                hits[key] = max(hits.get(key, 0), pen)
    return hits


def main():
    cfg = config.load()
    yml = config.path(cfg["services"]["planner"]["robot_yml"])
    data = yaml.safe_load(open(yml))["kinematics"]
    data["urdf_path"] = str(config.path(cfg["services"]["planner"]["urdf"]))
    data["asset_root_path"] = str(config.path(cfg["services"]["planner"]["urdf"]).parent)
    k = Kinematics(KinematicsCfg.from_robot_yaml_file({"kinematics": data}) if False else KinematicsCfg.from_robot_yaml_file(str(yml)))
    ign = data.get("self_collision_ignore") or {}
    ignored = {(a, b) for a, bs in ign.items() for b in bs} | {(b, a) for a, bs in ign.items() for b in bs}
    poses = {"zero": [0.0] * 14, "ready": cfg["manipulation"]["ready_deg"],
             "standing (501 policy)": [16.6, 13.0, -1.2, 56.2, 11.4, 1.5, -2.0, 16.8, -12.6, 1.5, 56.2, -5.5, 1.3, -0.1]}
    bad = 0
    for name, deg in poses.items():
        h = collisions(k, ignored, [math.radians(v) for v in deg] if name != "zero" else deg,
                       data.get("self_collision_buffer"))
        bad += len(h)
        print(f"{name:22s}", {f"{a}~{b}": round(p * 100, 1) for (a, b), p in h.items()} or "ok", "(cm)")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
