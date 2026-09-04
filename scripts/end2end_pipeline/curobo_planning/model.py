"""Standalone cuRobo robot-model access for the Zerith H1 PRO.

Abstracted (unmodified in behavior) from the reviewed planning stack in
/home/robot/tanzhen/GraspGenX/scripts/end2end_pipeline/zerith_curobo.py so this
repository can plan with cuRobo without depending on that checkout.

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


ZERITH_ACTIVE_JOINTS = (
    "daogui_joint",
    "body_pitch_joint",
    "body_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
)

# Zerith H1 PRO SDK V4.0 section 2.2.3 software control limits.  These are
# deliberately narrower than the mechanical/hard limits in the vendor URDF.
# Both the generated cuRobo planning URDF and LOW_LEVEL execution validation
# consume this mapping so a plan accepted by cuRobo is commandable by the SDK.
ZERITH_SOFTWARE_POSITION_LIMITS = {
    "daogui_joint": (0.0, 0.8),
    "body_pitch_joint": (0.0, 1.3),
    "body_yaw_joint": (-0.7, 0.7),
    "left_shoulder_pitch_joint": (-2.7, 1.5),
    "left_shoulder_roll_joint": (-0.3, 2.0),
    "left_shoulder_yaw_joint": (-2.9, 2.9),
    "left_elbow_joint": (-1.3, 1.5),
    "left_wrist_roll_joint": (-2.9, 2.9),
    "left_wrist_yaw_joint": (-1.0, 1.0),
    "left_wrist_pitch_joint": (-1.0, 1.0),
    "right_shoulder_pitch_joint": (-2.7, 1.5),
    "right_shoulder_roll_joint": (-2.0, 0.3),
    "right_shoulder_yaw_joint": (-2.9, 2.9),
    "right_elbow_joint": (-1.3, 1.5),
    "right_wrist_roll_joint": (-2.9, 2.9),
    "right_wrist_yaw_joint": (-1.0, 1.0),
    "right_wrist_pitch_joint": (-1.0, 1.0),
}

ZERITH_LOCKED_JOINTS = {
    "left_jaw_left_finger_joint": 0.0,
    "left_jaw_right_finger_joint": 0.0,
    "right_jaw_left_finger_joint": 0.0,
    "right_jaw_right_finger_joint": 0.0,
    "neck_yaw_joint": 0.0,
    "neck_pitch_joint": 0.0,
    "left_middle_wheel_joint": 0.0,
    "right_middle_wheel_joint": 0.0,
}

ZERITH_ARM_JOINTS = {
    "left": ZERITH_ACTIVE_JOINTS[3:10],
    "right": ZERITH_ACTIVE_JOINTS[10:17],
}

ZERITH_ARM_TOOL_FRAME = {
    "left": "left_end_effector_link",
    "right": "right_end_effector_link",
}

ZERITH_CONTACT_LINKS = {
    "left": ("left_jaw_left_finger_link", "left_jaw_right_finger_link"),
    "right": ("right_jaw_left_finger_link", "right_jaw_right_finger_link"),
}

_MODEL_DIR = Path(__file__).resolve().parents[3] / "assets" / "zerith" / "curobo"
ZERITH_CUROBO_YAML = _MODEL_DIR / "zerith.yml"
ZERITH_PLANNING_URDF = _MODEL_DIR / "zerith_planning.urdf"


def get_repo_root() -> Path:
    """Return the checkout root without consulting the working directory."""

    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: str, repo_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def load_curobo_config(
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a portable Zerith config ready to pass to cuRobo.

    Args:
        yaml_path: Optional alternate generated YAML. Relative paths inside it
            are still interpreted relative to this repository, not the
            caller's current directory.

    Returns:
        A fresh copy of the YAML's inner ``robot_cfg`` dictionary with
        absolute ``urdf_path`` and ``asset_root_path`` entries.
    """

    config_path = Path(yaml_path) if yaml_path is not None else ZERITH_CUROBO_YAML
    if not config_path.is_absolute():
        config_path = get_repo_root() / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    if not isinstance(document, dict) or not isinstance(document.get("robot_cfg"), dict):
        raise ValueError(f"Invalid cuRobo robot config (missing robot_cfg): {config_path}")

    robot_cfg = deepcopy(document["robot_cfg"])
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError(f"Invalid cuRobo robot config (missing kinematics): {config_path}")

    repo_root = get_repo_root()
    for key in ("urdf_path", "asset_root_path"):
        value = kinematics.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid cuRobo robot config ({key} is missing): {config_path}")
        kinematics[key] = _resolve_repo_path(value, repo_root)

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


__all__ = [
    "ZERITH_ACTIVE_JOINTS",
    "ZERITH_ARM_JOINTS",
    "ZERITH_ARM_TOOL_FRAME",
    "ZERITH_CUROBO_YAML",
    "ZERITH_CONTACT_LINKS",
    "ZERITH_LOCKED_JOINTS",
    "ZERITH_PLANNING_URDF",
    "ZERITH_SOFTWARE_POSITION_LIMITS",
    "get_repo_root",
    "build_single_arm_planning_config",
    "load_curobo_config",
]
