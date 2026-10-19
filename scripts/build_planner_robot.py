"""Build the arms-only planning model for cuRobo.

1. Cut the G1 URDF down to the torso subtree (torso, head, both arms with rubber hands), so the
   planner works in the torso frame with the waist fixed.
2. Add the palm contact frames (left_tcp / right_tcp) from config.planner_model.tcp.
3. Copy the meshes it needs, fit collision spheres and the self-collision matrix with cuRobo's
   RobotBuilder, and save the robot yml.

Run (once, or after changing the TCP offsets):
  ~/miniconda3/envs/g1ui_curobo/bin/python -m scripts.build_planner_robot
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from common import config

BASE = "torso_link"


def cut_urdf(src, dst, tcp):
    tree = ET.parse(src)
    root = tree.getroot()
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    keep, stack = {BASE}, [BASE]
    while stack:
        for j in children.get(stack.pop(), []):
            child = j.find("child").get("link")
            keep.add(child)
            stack.append(child)
    for el in list(root):
        if el.tag == "link" and el.get("name") not in keep:
            root.remove(el)
        elif el.tag == "joint" and (el.find("child").get("link") not in keep or el.find("parent").get("link") not in keep):
            root.remove(el)
    for name, t in tcp.items():
        ET.SubElement(root, "link", name=name)
        j = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(j, "parent", link=t["parent"])
        ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "origin", xyz=" ".join(str(v) for v in t["xyz"]), rpy="0 0 0")
    root.set("name", "g1_arms")
    tree.write(dst)
    meshes = {m.get("filename") for m in root.iter("mesh")}
    return sorted(keep), sorted(meshes)


def main():
    cfg = config.load()
    pm, pl = cfg["planner_model"], cfg["services"]["planner"]
    out_urdf = config.path(pl["urdf"])
    out_dir = out_urdf.parent
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    links, meshes = cut_urdf(config.path(pm["source_urdf"]), out_urdf, pm["tcp"])
    for m in meshes:
        shutil.copy(config.path(pm["source_meshes"]) / Path(m).name, out_dir / "meshes" / Path(m).name)
    print(f"URDF: {out_urdf} ({len(links)} links, {len(meshes)} meshes)")

    from curobo.robot_builder import RobotBuilder
    builder = RobotBuilder(urdf_path=str(out_urdf), asset_path=str(out_dir), tool_frames=list(pm["tcp"]))
    builder.fit_collision_spheres(sphere_density=1.0)
    builder.compute_collision_matrix(prune_collisions=True, num_samples=1000)
    robot_cfg = builder.build()
    yml = config.path(pl["robot_yml"])
    builder.save(robot_cfg, str(yml))
    import yaml
    data = yaml.safe_load(open(yml))
    ign = data["kinematics"].get("self_collision_ignore") or {}
    for link, others in pm.get("self_collision_ignore", {}).items():
        ign[link] = sorted(set(ign.get(link, [])) | set(others))
    data["kinematics"]["self_collision_ignore"] = ign
    yaml.safe_dump(data, open(yml, "w"), sort_keys=False)
    print(f"robot yml: {yml} ({builder.num_spheres} spheres, {len(builder.collision_matrix)} ignore pairs)")


if __name__ == "__main__":
    main()
