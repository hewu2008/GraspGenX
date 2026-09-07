"""Shared constants for the cuRobo planning stack.

Single source of truth for the Zerith model constants, model file paths, and
the wrist->end-effector offset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

_MODEL_DIR = REPO_ROOT / "assets" / "zerith" / "curobo"
ZERITH_CUROBO_YAML = _MODEL_DIR / "zerith.yml"

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

# Software position limits, aligned with the URDF mechanical joint limits
# (assets/zerith/urdf/ZR_H1PRO-1.2.00.H.V4.3_URDF_2025.12.02.urdf) so that any
# joint solution produced by the pinocchio / cuRobo IK (which clips to the URDF
# limits) is commandable by the LOW_LEVEL SDK without a limit-violation.  Both
# the generated cuRobo planning URDF and LOW_LEVEL execution validation consume
# this mapping so a plan accepted by cuRobo is commandable by the SDK.
ZERITH_SOFTWARE_POSITION_LIMITS = {
    "daogui_joint": (0.0, 0.8),
    "body_pitch_joint": (-0.05236, 1.309),
    "body_yaw_joint": (-1.0472, 1.0472),
    "left_shoulder_pitch_joint": (-2.7925, 1.5708),
    "left_shoulder_roll_joint": (-0.5236, 2.0944),
    "left_shoulder_yaw_joint": (-2.9671, 2.9671),
    "left_elbow_joint": (-1.5184, 1.5708),
    "left_wrist_roll_joint": (-2.9671, 2.9671),
    "left_wrist_yaw_joint": (-1.0472, 1.0472),
    "left_wrist_pitch_joint": (-1.0472, 1.0472),
    "right_shoulder_pitch_joint": (-2.7925, 1.5708),
    "right_shoulder_roll_joint": (-2.0944, 0.5236),
    "right_shoulder_yaw_joint": (-2.9671, 2.9671),
    "right_elbow_joint": (-1.5184, 1.5708),
    "right_wrist_roll_joint": (-2.9671, 2.9671),
    "right_wrist_yaw_joint": (-1.0472, 1.0472),
    "right_wrist_pitch_joint": (-1.0472, 1.0472),
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
