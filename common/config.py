"""Loads config/config.yaml once. Paths in the config are relative to the project root."""
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
    """Palm contact frames {name: {parent, xyz}} on the glove surface (bare-hand TCP + glove contact offset)."""
    pm = load()["planner_model"]
    off = pm.get("glove", {}).get("contact_m", 0.0)
    return {n: {"parent": t["parent"], "xyz": [x + off * k for x, k in zip(t["xyz"], t["normal"])]}
            for n, t in pm["tcp"].items()}
