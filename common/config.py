"""Loads config/config.yaml once. Paths in the config are relative to the project root. The hand-specific parts
come from the active hand profile (config `hands`, switched by use_hands when the robot reports other hands)."""
import copy
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load() -> dict:
    path = Path(os.environ.get("G1UI_CONFIG", ROOT / "config" / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    pm = cfg["planner_model"]
    cfg["_hand_lists"] = {k: copy.deepcopy(pm[k]) for k in ("self_collision_ignore", "held_object_ignore")}
    cfg["_manipulation"] = copy.deepcopy(cfg["manipulation"])     # before any hand profile's overrides
    ee = cfg["end_effector"]
    _apply_hands(cfg, ee["detect"] if ee["detect"] in cfg["hands"] else ee["default"])
    return cfg


def hands() -> str:
    """The active hand profile (config `hands`): what the robot detected on its wrists."""
    return load()["_hands"]


def use_hands(name: str) -> bool:
    """Switch the hand profile process-wide (the robot reported other hands). Returns whether it changed; the
    caller rebuilds what it derived from the old profile (kinematics, gravity, cached hand sizes)."""
    cfg = load()
    if name == cfg["_hands"]:
        return False
    if name not in cfg["hands"]:
        raise ValueError(f"unknown hand profile '{name}' (config hands: {sorted(cfg['hands'])})")
    _apply_hands(cfg, name)
    return True


def _apply_hands(cfg, name):
    """Put one hand profile's values where the code reads them."""
    h, pm = cfg["hands"][name], cfg["planner_model"]
    cfg["robot"]["urdf"] = h["urdf"]
    cfg["services"]["planner"]["urdf"], cfg["services"]["planner"]["robot_yml"] = h["planner_urdf"], h["planner_yml"]
    pm.update(source_urdf=h["urdf"], tcp=h["tcp"], glove=h.get("glove") or {}, contact_ahead_m=h["contact_ahead_m"])
    cfg["manipulation"].clear()                 # the shared values, then this hand's own (profile `manipulation`)
    cfg["manipulation"].update(copy.deepcopy(cfg["_manipulation"]), **copy.deepcopy(h.get("manipulation") or {}))
    cfg["manipulation"]["palm_below_m"] = h["palm_below_m"]
    names = {"$left_hand": h["links"][0], "$right_hand": h["links"][1]}
    sub = lambda v: names.get(v, v)
    lists = cfg["_hand_lists"]
    pm["self_collision_ignore"] = {sub(k): [sub(x) for x in v] for k, v in lists["self_collision_ignore"].items()}
    pm["held_object_ignore"] = [sub(x) for x in lists["held_object_ignore"]]
    cfg["_hands"] = name


def path(rel: str) -> Path:
    """Resolve a config path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def mode_id(name: str) -> int:
    return load()["modes"]["ids"][name]


def mode_name(fsm_id: int) -> str:
    for name, i in load()["modes"]["ids"].items():
        if i == fsm_id:
            return name
    return f"unknown_{fsm_id}"


def tcp_frames() -> dict:
    """Palm contact frames {name: {parent, xyz, rpy}}: x along the fingers, z toward the thumb, the palm facing -y
    (left hand) / +y (right hand), so a squeeze goal with both frames level puts the palms facing each other.
    A tcp entry gives its parent link, the contact point xyz, the palm normal (out of the palm) and, for a hand
    whose link frame is not x-forward, the finger axis (both in the parent frame).
    Without an axis (the old rubber hands with gloves): the rubber fingers curl in, so the glove surface is innermost
    at the fingertips (contact_m beyond the bare TCP, at fingertip_contact_x_m) and turns away toward the wrist by
    palm_twist_deg about z; the palm normals are +-y of the hand frames."""
    pm = load()["planner_model"]
    g = pm.get("glove") or {}
    tw = math.radians(g.get("palm_twist_deg", 0.0))
    out = {}
    for n, t in pm["tcp"].items():
        if "axis" in t:
            x = _unit(t["axis"])
            y = _unit(t["normal"]) * (-1.0 if n.startswith("left") else 1.0)
            out[n] = {"parent": t["parent"], "xyz": list(map(float, t["xyz"])), "rpy": _mat_rpy(np.column_stack([x, y, np.cross(x, y)]))}
            continue
        x = t["xyz"][0]
        off = g.get("contact_m", 0.0) - (g.get("fingertip_contact_x_m", x) - x) * math.tan(tw)
        out[n] = {"parent": t["parent"], "xyz": [v + off * k for v, k in zip(t["xyz"], t["normal"])],
                  "rpy": [0.0, 0.0, tw * t["normal"][1]]}
    return out


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _mat_rpy(R):
    """URDF roll-pitch-yaw of a rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        return [math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])]
    return [math.atan2(-R[1, 2], R[1, 1]), p, 0.0]
