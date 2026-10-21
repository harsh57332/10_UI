"""Loads config/config.yaml once. Paths in the config are relative to the project root."""
import math
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load() -> dict:
    path = Path(os.environ.get("G1UI_CONFIG", ROOT / "config" / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


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
    """Palm contact frames {name: {parent, xyz, rpy}} on the glove's palm surface. The rubber fingers curl
    in, so the glove surface is innermost at the fingertips (contact_m beyond the bare TCP, at
    fingertip_contact_x_m) and turns away toward the wrist by palm_twist_deg. The frame lies in that
    surface: a grasp that puts it flat on an object twists the hand so the whole palm touches.
    The palm normals are +-y of the hand frames, so the twist is a rotation about z."""
    pm = load()["planner_model"]
    g = pm.get("glove", {})
    tw = math.radians(g.get("palm_twist_deg", 0.0))
    out = {}
    for n, t in pm["tcp"].items():
        x = t["xyz"][0]
        off = g.get("contact_m", 0.0) - (g.get("fingertip_contact_x_m", x) - x) * math.tan(tw)
        out[n] = {"parent": t["parent"], "xyz": [v + off * k for v, k in zip(t["xyz"], t["normal"])],
                  "rpy": [0.0, 0.0, tw * t["normal"][1]]}
    return out
