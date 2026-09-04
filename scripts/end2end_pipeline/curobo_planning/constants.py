"""Shared constants for the cuRobo planning stack.

Single source of truth for the Zerith model constants, model file paths, the
wrist->end-effector offset, and the reviewed cuRobo commit pin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]

_MODEL_DIR = REPO_ROOT / "assets" / "zerith" / "curobo"
ZERITH_CUROBO_YAML = _MODEL_DIR / "zerith.yml"
ZERITH_PLANNING_URDF = _MODEL_DIR / "zerith_planning.urdf"

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

# Fixed offset from the wrist frame controlled by ``setArm_high`` to the
# gripper's end-effector tool frame in the planning URDF.
WRIST_T_END_EFFECTOR = np.array(
    [
        [1.0, 0.0, 0.0, 0.1435],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

EXPECTED_CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"
