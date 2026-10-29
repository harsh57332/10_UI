"""Robot URDF with the Inspire RH56DFTP hands (fitted 2026-10-29), each OPEN hand merged into one rigid link.

Source: Unitree's g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf (unitree_ros, robots/g1_description). Every finger
joint at 0 is the hand fully open (Inspire angle 1000), the state the arms work in; the fingers move only from
the UI (section 5), so for kinematics, gravity and planning each hand is one link:
  <side>_inspire_hand  in place of <side>_base_link and its fingers/force sensors, same frame (z along the fingers,
                       +y out of the palm, thumb side +x left / -x right), mesh = the union of each part's convex
                       hull, mass = the parts' total at their combined centre of mass.
Run (after updating the unitree_ros checkout): ~/miniconda3/envs/g1ui_curobo/bin/python -m scripts.make_inspire_urdf
Writes assets/robot/g1_29dof_rev_1_0_inspire_ftp.urdf and assets/robot/meshes/<side>_inspire_hand.STL."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from common import config
from common.fk import FK, _rpy

SRC_DIR = config.path("third_party/unitree_ros/robots/g1_description")
SRC = SRC_DIR / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
OUT = config.path("assets/robot/g1_29dof_rev_1_0_inspire_ftp.urdf")


def _origin(el):
    """4x4 of an <origin> element (identity if missing)."""
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in (el.get("rpy") or "0 0 0").split()])
        T[:3, 3] = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    return T


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    fk = FK(SRC)
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    (OUT.parent / "meshes").mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        base = f"{side}_base_link"
        hand, stack = [], [base]                        # the base and everything below it
        while stack:
            n = stack.pop()
            hand.append(n)
            stack += children.get(n, [])
        parts, mass, moment = [], 0.0, np.zeros(3)
        for n in hand:
            T = fk.pose(n, {}, base=base)                 # every finger joint at 0: open
            geo = links[n].find("collision") or links[n].find("visual")
            if geo is not None and geo.find("geometry/mesh") is not None:
                m = trimesh.load(SRC_DIR / geo.find("geometry/mesh").get("filename"), force="mesh")
                hull = m.convex_hull
                hull.apply_transform(T @ _origin(geo.find("origin")))
                parts.append(hull)
            i = links[n].find("inertial")
            if i is not None:
                m_ = float(i.find("mass").get("value"))
                c = (T @ _origin(i.find("origin")))[:3, 3]
                mass, moment = mass + m_, moment + m_ * c
        com = moment / mass
        mesh = trimesh.util.concatenate(parts)
        name = f"{side}_inspire_hand"
        mesh.export(OUT.parent / "meshes" / f"{name}.STL")
        # replace the hand subtree by one link, attached where the base was
        for n in hand:
            root.remove(links[n])
        for j in joints:
            if j.find("child").get("link") in hand and j.find("child").get("link") != base:
                root.remove(j)
            elif j.find("child").get("link") == base:
                j.set("name", f"{name}_joint")
                j.find("child").set("link", name)
        ext = mesh.extents
        ixx, iyy, izz = (mass / 12 * (ext[1] ** 2 + ext[2] ** 2), mass / 12 * (ext[0] ** 2 + ext[2] ** 2),
                         mass / 12 * (ext[0] ** 2 + ext[1] ** 2))
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.6f}" for v in com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.5f}")
        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.7f}", ixy="0", ixz="0", iyy=f"{iyy:.7f}", iyz="0", izz=f"{izz:.7f}")
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(el, "geometry"), "mesh", filename=f"meshes/{name}.STL")
        print(f"{name}: {len(hand)} links merged, {mass:.3f} kg, com {np.round(com, 3)}, "
              f"size {np.round(ext, 3)} m, {len(mesh.faces)} faces")
    tree.write(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
