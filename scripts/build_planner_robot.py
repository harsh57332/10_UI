"""Build the arms-only planning model for cuRobo.

1. Cut the G1 URDF down to the torso subtree (torso, head, both arms with their hands), so the
   planner works in the torso frame with the waist fixed.
2. Add the palm contact frames (left_tcp / right_tcp) on the glove surface (planner_model.tcp + glove).
3. Copy the meshes it needs, fit collision spheres and the self-collision matrix with cuRobo's
   RobotBuilder (saved as <robot_yml>_fit.yml), then grow the hand spheres to the padded gloves, add
   the held-box link and the extra ignores, and save the robot yml.

The sphere fit is random (GPU) even with a seed, so it runs only with --refit (or when the fit file is
missing); changing the TCP, the glove sizes or the ignores keeps the arm model that was tested.
Run (after changing the TCP offsets or the glove sizes; --refit after changing the URDF; --refit-hands PREV_FIT
after changing only the hands: the fit of every other link and its self-collision pairs come from PREV_FIT, the
arm model that was tested):
  ~/miniconda3/envs/g1ui_curobo/bin/python -m scripts.build_planner_robot [--hands NAME] [--refit | --refit-hands PREV_FIT]
Each hand profile (config hands: dummy, inspire) has its own model files (hands.<name>.planner_urdf / planner_yml).
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from common import config

BASE = "torso_link"
MIN_R = 0.008                                       # smallest glove sphere kept


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
        ET.SubElement(j, "origin", xyz=" ".join(str(v) for v in t["xyz"]), rpy=" ".join(str(v) for v in t["rpy"]))
    root.set("name", "g1_arms")
    tree.write(dst)
    meshes = {m.get("filename") for m in root.iter("mesh")}
    return sorted(keep), sorted(meshes)


def glove(spheres, tcp, normals, g):
    """Padded gloves: every hand sphere grows by the padding, the fingertip spheres (outer third of the hand
    along x, the finger direction) get copies shifted out by the extra glove finger length, and nothing
    reaches past the glove's flat palm surface (the contact frame) toward the box: the rubber fingertips
    curl in further than the gloved palm, so those spheres are pushed back to it. Nothing reaches further
    back than the measured glove thickness (palm surface to back) either."""
    pad, extra, thick = g.get("pad_m", 0.0), g.get("finger_extra_m", 0.0), g.get("thickness_m")
    for name, t in tcp.items():
        a = t["rpy"][2]                         # the palm surface is twisted about z (see config.tcp_frames)
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        link, n, p0 = t["parent"], rz @ np.array(normals[name], float), np.array(t["xyz"], float)
        sp = spheres[link]
        xs = [c["center"][0] for c in sp]
        tip = max(xs) - (max(xs) - min(xs)) / 3
        tips = [{"center": [c["center"][0] + extra, *c["center"][1:]], "radius": c["radius"]}
                for c in sp if c["center"][0] >= tip] if extra > 0 else []
        out = []
        for c in sp + tips:
            ctr, r = np.array(c["center"], float), c["radius"] + pad
            excess = (ctr - p0) @ n + r             # how far it reaches past the glove's palm surface
            if excess > 0:                          # trim its palm side to the surface: shrink by half the
                ctr -= excess / 2 * n               # excess and move back by half, so its back edge (the
                r -= excess / 2                     # glove's back padding) stays where it was
                if r < MIN_R:
                    ctr -= (MIN_R - r) * n
                    r = MIN_R
            back = (p0 - ctr) @ n + r - thick if thick else 0.0   # how far it reaches past the glove's back
            if back > 0:                            # same on the back: the measured palm-to-back thickness
                ctr += back / 2 * n
                r -= back / 2
                if r < MIN_R:
                    ctr += (MIN_R - r) * n
                    r = MIN_R
            out.append({"center": [round(float(v), 5) for v in ctr], "radius": round(float(r), 5)})
        spheres[link] = out


def keep_arm_fit(fit, prev, hands):
    """The new fit keeps its hand links; every other link's spheres and the self-collision pairs between other
    links come from the previous (tested) fit."""
    import yaml
    new, old = yaml.safe_load(open(fit)), yaml.safe_load(open(prev))
    kn, ko = new["kinematics"], old["kinematics"]
    for link in kn["collision_spheres"]:
        if link not in hands and link in ko["collision_spheres"]:
            kn["collision_spheres"][link] = ko["collision_spheres"][link]
    ign = {}
    for link, others in (kn.get("self_collision_ignore") or {}).items():   # pairs with a hand: the new fit's
        ign[link] = [o for o in others if link in hands or o in hands]
    for link, others in (ko.get("self_collision_ignore") or {}).items():   # pairs between other links: the old fit's
        if link in kn["collision_spheres"]:
            ign[link] = sorted(set(ign.get(link, [])) | {o for o in others if o in kn["collision_spheres"]})
    kn["self_collision_ignore"] = {k: v for k, v in ign.items() if v}
    yaml.safe_dump(new, open(fit, "w"), sort_keys=False)
    print(f"kept the arm fit of {prev} (new spheres only for {sorted(hands)})")


def main():
    if "--hands" in sys.argv:                   # the hand profile to build (config hands); default: the configured one
        config.use_hands(sys.argv[sys.argv.index("--hands") + 1])
    cfg = config.load()
    pm, pl = cfg["planner_model"], cfg["services"]["planner"]
    out_urdf = config.path(pl["urdf"])
    out_dir = out_urdf.parent
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    tcp = config.tcp_frames()
    links, meshes = cut_urdf(config.path(pm["source_urdf"]), out_urdf, tcp)
    dirs = pm["source_meshes"] if isinstance(pm["source_meshes"], list) else [pm["source_meshes"]]
    for m in meshes:                            # the first mesh folder that has it (unitree_ros, then our own)
        src = next((config.path(d) / Path(m).name for d in dirs if (config.path(d) / Path(m).name).exists()), None)
        if src is None:
            raise SystemExit(f"mesh {m} not found in {dirs}")
        shutil.copy(src, out_dir / "meshes" / Path(m).name)
    print(f"URDF: {out_urdf} ({len(links)} links, {len(meshes)} meshes)")

    yml = config.path(pl["robot_yml"])
    fit = yml.with_name(yml.stem + "_fit.yml")
    hands_only = "--refit-hands" in sys.argv
    if "--refit" in sys.argv or hands_only or not fit.exists():
        import torch
        from curobo.robot_builder import RobotBuilder
        np.random.seed(pm["fit_seed"])
        torch.manual_seed(pm["fit_seed"])
        builder = RobotBuilder(urdf_path=str(out_urdf), asset_path=str(out_dir), tool_frames=list(tcp))
        builder.fit_collision_spheres(sphere_density=1.0)
        builder.compute_collision_matrix(prune_collisions=True, num_samples=1000)
        builder.save(builder.build(), str(fit))
        print(f"sphere fit: {fit}")
        if hands_only:
            keep_arm_fit(fit, Path(sys.argv[sys.argv.index("--refit-hands") + 1]), {t["parent"] for t in tcp.values()})
    import yaml
    data = yaml.safe_load(open(fit))
    k = data["kinematics"]
    if pm.get("glove"):                         # the old rubber hands wore padded gloves; bare hands keep their fit
        glove(k["collision_spheres"], tcp, {n: t["normal"] for n, t in pm["tcp"].items()}, pm["glove"])
    # a link for a held box (cuRobo fits spheres to it when attached), riding on the left palm frame
    for key in ("extra_links", "extra_collision_spheres", "self_collision_buffer"):
        k[key] = k.get(key) or {}
    k["extra_links"]["attached_object"] = {
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint", "joint_type": "FIXED",
        "link_name": "attached_object", "parent_link_name": "left_tcp"}
    k["extra_collision_spheres"]["attached_object"] = pm["held_object_spheres"]
    if "attached_object" not in k["collision_link_names"]:
        k["collision_link_names"].append("attached_object")
    k["self_collision_buffer"]["attached_object"] = 0.0
    ign = data["kinematics"].get("self_collision_ignore") or {}
    ign["attached_object"] = sorted(set(ign.get("attached_object", [])) | set(pm["held_object_ignore"]))
    for link, others in pm.get("self_collision_ignore", {}).items():
        ign[link] = sorted(set(ign.get(link, [])) | set(others))
    for link, others in pm.get("self_collision_check", {}).items():   # checked even where the fit ignores them
        for a, b in [(link, o) for o in others] + [(o, link) for o in others]:
            if b in ign.get(a, []):
                ign[a].remove(b)
    k["self_collision_buffer"].update(pm.get("self_collision_buffer", {}))
    data["kinematics"]["self_collision_ignore"] = ign
    if pm.get("cspace"):                        # the posture cuRobo's IK leans toward (natural arms), per-joint weights
        import math
        cs, names = pm["cspace"], k["cspace"]["joint_names"]
        side = lambda n: (n.split("_", 1)[1], 1.0 if n.startswith("left") or "pitch" in n or "elbow" in n else -1.0)
        k["cspace"]["default_joint_position"] = [math.radians(cs["default_deg"][side(n)[0].replace("_joint", "")] * side(n)[1])
                                                 for n in names]
        k["cspace"]["null_space_weight"] = [float(cs["null_space_weight"][side(n)[0].replace("_joint", "")]) for n in names]
    k["urdf_path"], k["asset_root_path"] = str(out_urdf), str(out_dir)    # this profile's files (not the fit's)
    yaml.safe_dump(data, open(yml, "w"), sort_keys=False)
    n = sum(len(v) for v in k["collision_spheres"].values())
    print(f"robot yml: {yml} ({n} spheres, {sum(len(v) for v in ign.values())} ignore pairs)")
    from scripts.check_self_collision import main as check
    if check():
        raise SystemExit("the model self-collides in a key pose: add the pair to planner_model.self_collision_ignore")


if __name__ == "__main__":
    main()
