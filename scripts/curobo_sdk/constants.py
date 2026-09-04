"""SDK-independent constants for the Zerith LOW_LEVEL driver.

The active-joint order and software position limits are shared with the cuRobo
planning stack, so this module re-uses them from ``curobo_planning.constants``
(the single source of truth) rather than duplicating them.  The remaining
constants (motor mapping, motor-ID whitelist, gripper IDs, MIT sentinels,
feedback tolerance, mode/init states) are specific to the SDK low-level driver.
"""

from __future__ import annotations

from curobo_planning.constants import (
    ZERITH_ACTIVE_JOINTS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)

NUM_ACTIVE_JOINTS = len(ZERITH_ACTIVE_JOINTS)  # 17

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