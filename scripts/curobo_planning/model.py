"""Standalone cuRobo robot-model access for the Zerith H1 PRO.

The committed YAML (assets/zerith/curobo/zerith.yml) stores repository-relative
paths; :func:`load_curobo_config` resolves them at runtime and returns the
inner ``robot_cfg`` mapping accepted by cuRobo's planner/config factories.
``build_single_arm_planning_config`` reduces the 17-DoF whole-body model to a
seven-DoF single-arm planner with every other joint locked at a live snapshot.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .constants import (
    REPO_ROOT,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    ZERITH_ARM_TOOL_FRAME,
    ZERITH_CUROBO_YAML,
    ZERITH_LOCKED_JOINTS,
)


def _resolve_repo_path(value: str, repo_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def load_curobo_config(
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a portable Zerith config ready to pass to cuRobo.

    Relative paths inside the YAML are interpreted relative to this repository,
    not the caller's current directory.  Returns a fresh copy of the inner
    ``robot_cfg`` mapping with absolute ``urdf_path`` and ``asset_root_path``.
    """

    config_path = Path(yaml_path) if yaml_path is not None else ZERITH_CUROBO_YAML
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    if not isinstance(document, dict) or not isinstance(document.get("robot_cfg"), dict):
        raise ValueError(f"Invalid cuRobo robot config (missing robot_cfg): {config_path}")

    robot_cfg = deepcopy(document["robot_cfg"])
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError(f"Invalid cuRobo robot config (missing kinematics): {config_path}")

    for key in ("urdf_path", "asset_root_path"):
        value = kinematics.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid cuRobo robot config ({key} is missing): {config_path}")
        kinematics[key] = _resolve_repo_path(value, REPO_ROOT)

    return robot_cfg


def build_single_arm_planning_config(
    arm: Literal["left", "right"],
    full_joint_position: Mapping[str, float],
    locked_joint_position: Mapping[str, float] | None = None,
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a seven-DOF planning config with every other joint locked.

    Locked values come from the synchronized robot snapshot.  Collision links,
    collision spheres, and self-collision configuration are intentionally left
    intact so the stationary waist and opposite arm remain collision geometry.
    """

    if arm not in ZERITH_ARM_JOINTS:
        raise ValueError(f"Unsupported target arm: {arm!r}")
    missing = set(ZERITH_ACTIVE_JOINTS) - set(full_joint_position)
    extra = set(full_joint_position) - set(ZERITH_ACTIVE_JOINTS)
    if missing or extra:
        raise ValueError(
            f"full_joint_position mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name, value in full_joint_position.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Joint {name} has a non-numeric lock value")

    robot_cfg = load_curobo_config(yaml_path)
    kinematics = robot_cfg["kinematics"]
    cspace = kinematics.get("cspace")
    if not isinstance(cspace, dict):
        raise ValueError("Zerith config has no cspace mapping")
    source_names = list(cspace.get("joint_names", ()))
    if source_names != list(ZERITH_ACTIVE_JOINTS):
        raise ValueError("Zerith cspace joint order differs from the public 17-joint order")

    active_names = list(ZERITH_ARM_JOINTS[arm])
    active_indices = [source_names.index(name) for name in active_names]
    for key in (
        "default_joint_position",
        "cspace_distance_weight",
        "null_space_weight",
    ):
        values = cspace.get(key)
        if not isinstance(values, list) or len(values) != len(source_names):
            raise ValueError(f"Zerith cspace {key} must have 17 values")
        if key == "default_joint_position":
            cspace[key] = [float(full_joint_position[name]) for name in active_names]
        else:
            cspace[key] = [values[index] for index in active_indices]
    cspace["joint_names"] = active_names

    lock_joints = deepcopy(kinematics.get("lock_joints", {}))
    if not isinstance(lock_joints, dict):
        raise ValueError("Zerith kinematics lock_joints must be a mapping")
    if locked_joint_position is not None:
        supplied_locked = {
            str(name): float(value) for name, value in locked_joint_position.items()
        }
        if set(supplied_locked) != set(ZERITH_LOCKED_JOINTS):
            raise ValueError("locked_joint_position keys differ from the Zerith model")
        if not all(math.isfinite(value) for value in supplied_locked.values()):
            raise ValueError("locked_joint_position contains NaN or Inf")
        lock_joints.update(supplied_locked)
    for name in ZERITH_ACTIVE_JOINTS:
        if name not in active_names:
            lock_joints[name] = float(full_joint_position[name])
    kinematics["lock_joints"] = lock_joints
    kinematics["tool_frames"] = [ZERITH_ARM_TOOL_FRAME[arm]]
    return robot_cfg