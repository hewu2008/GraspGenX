"""SDK-independent constants for the Zerith LOW_LEVEL driver.

Values mirror ``curobo_planning.constants`` (and the reference tanzhen repo's
``joint_state_bridge.py`` / ``zerith_curobo.py``) so that a cuRobo-accepted
joint trajectory is commandable by the SDK.  This package must NOT import
``curobo_planning``; the equality with the planning package is asserted in
``tests/test_curobo_sdk_constants.py`` instead.
"""

from __future__ import annotations

# Keep in sync with curobo_planning.constants.ZERITH_ACTIVE_JOINTS.
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

NUM_ACTIVE_JOINTS = len(ZERITH_ACTIVE_JOINTS)  # 17

# Keep in sync with curobo_planning.constants.ZERITH_SOFTWARE_POSITION_LIMITS.
# Zerith H1 PRO SDK V4.0 section 2.2.3 software control limits, deliberately
# narrower than the mechanical limits in the vendor URDF.
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

# Mirror reference joint_state_bridge.JOINT_TO_MOTOR_NAME.
JOINT_TO_MOTOR_NAME: dict[str, str] = {
    "daogui_joint": "MOTOR_LIFT",
    "body_pitch_joint": "MOTOR_WAIST_DOWN",
    "body_yaw_joint": "MOTOR_WAIST_UP",
    "left_shoulder_pitch_joint": "MOTOR_LEFT_ARM_1",
    "left_shoulder_roll_joint": "MOTOR_LEFT_ARM_2",
    "left_shoulder_yaw_joint": "MOTOR_LEFT_ARM_3",
    "left_elbow_joint": "MOTOR_LEFT_ARM_4",
    "left_wrist_roll_joint": "MOTOR_LEFT_ARM_5",
    "left_wrist_yaw_joint": "MOTOR_LEFT_ARM_6",
    "left_wrist_pitch_joint": "MOTOR_LEFT_ARM_7",
    "right_shoulder_pitch_joint": "MOTOR_RIGHT_ARM_1",
    "right_shoulder_roll_joint": "MOTOR_RIGHT_ARM_2",
    "right_shoulder_yaw_joint": "MOTOR_RIGHT_ARM_3",
    "right_elbow_joint": "MOTOR_RIGHT_ARM_4",
    "right_wrist_roll_joint": "MOTOR_RIGHT_ARM_5",
    "right_wrist_yaw_joint": "MOTOR_RIGHT_ARM_6",
    "right_wrist_pitch_joint": "MOTOR_RIGHT_ARM_7",
}

# Active-command motor ID ranges (see EtherCAT_Motor_Index in the SDK stub).
WAIST_MOTOR_IDS: tuple[int, ...] = (2, 3, 4)
LEFT_ARM_MOTOR_IDS: tuple[int, ...] = tuple(range(7, 14))
RIGHT_ARM_MOTOR_IDS: tuple[int, ...] = tuple(range(15, 22))
EXPECTED_ACTIVE_MOTOR_IDS = frozenset(
    (*WAIST_MOTOR_IDS, *LEFT_ARM_MOTOR_IDS, *RIGHT_ARM_MOTOR_IDS)
)
FORBIDDEN_MOTOR_IDS: frozenset[int] = frozenset((0, 1))  # wheels

# Motor IDs for the parallel grippers (MOTOR_LEFT_ARM_8 / MOTOR_RIGHT_ARM_8).
GRIPPER_MOTOR_ID: dict[str, int] = {"left": 14, "right": 22}
GRIPPER_MOTOR_NAME: dict[str, str] = {
    "left": "MOTOR_LEFT_ARM_8",
    "right": "MOTOR_RIGHT_ARM_8",
}

GRIPPER_OPEN_POSITION: float = 0.0
GRIPPER_CLOSED_POSITION: float = 1.5

# Absorb encoder-zero jitter near a soft-limit boundary (mirror reference).
FEEDBACK_LIMIT_TOLERANCE: float = 5.0e-3

# Motor_Control sentinels (KP/KD = -1 means "not set, use controller default").
DEFAULT_KP: float = -1.0
DEFAULT_KD: float = -1.0

# Mode / init constants for the fake robot.  The real SDK path reads these
# dynamically from the injected sdk module and never hard-codes the ints.
LOW_LEVEL_MODE: int = 1
HIGH_LEVEL_MODE: int = 2
INIT_COMPLETE: int = 2