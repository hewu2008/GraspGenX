"""curobo_sdk: thin LOW_LEVEL encapsulation of the Zerith SDK for end2end.

Public surface:
    - ``LowLevelRobot``        driver wrapping real or fake robot primitives
    - ``create_low_level_robot`` factory (real ``fake=False`` or fake robot)
    - ``FakeSDKRobot``          in-memory SDK stand-in for unit tests
    - ``MotorFeedback`` / ``GripperFeedback`` feedback dataclasses
    - ``JointCalibration``      and motor<->model conversion functions
    - constants mirroring ``curobo_planning.constants``
    - the full ``exceptions`` hierarchy

The package must NOT import ``curobo_planning`` (it is a lower-level driver);
constant equality with the planning package is asserted only in tests.
"""

from __future__ import annotations

from .api import create_low_level_robot
from .calibration import (
    JointCalibration,
    motor_to_model,
    motor_velocity_to_model,
    model_to_motor,
    normalize_feedback_position,
)
from .constants import (
    DEFAULT_KD,
    DEFAULT_KP,
    EXPECTED_ACTIVE_MOTOR_IDS,
    FORBIDDEN_MOTOR_IDS,
    GRIPPER_CLOSED_POSITION,
    GRIPPER_MOTOR_ID,
    GRIPPER_MOTOR_NAME,
    GRIPPER_OPEN_POSITION,
    JOINT_TO_MOTOR_NAME,
    LEFT_ARM_MOTOR_IDS,
    NUM_ACTIVE_JOINTS,
    RIGHT_ARM_MOTOR_IDS,
    WAIST_MOTOR_IDS,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from .exceptions import (
    CalibrationError,
    CommandError,
    ConfigurationError,
    ControlModeError,
    CuroboSdkError,
    FeedbackError,
    InitStateError,
    LimitViolationError,
    RobotConnectionError,
)
from .fake_robot import FakeSDKRobot
from .low_level import GripperFeedback, LowLevelRobot, MotorFeedback

__all__ = [
    # facade
    "create_low_level_robot",
    "LowLevelRobot",
    "FakeSDKRobot",
    # feedback
    "MotorFeedback",
    "GripperFeedback",
    # calibration
    "JointCalibration",
    "motor_to_model",
    "model_to_motor",
    "motor_velocity_to_model",
    "normalize_feedback_position",
    # constants
    "ZERITH_ACTIVE_JOINTS",
    "NUM_ACTIVE_JOINTS",
    "ZERITH_SOFTWARE_POSITION_LIMITS",
    "JOINT_TO_MOTOR_NAME",
    "WAIST_MOTOR_IDS",
    "LEFT_ARM_MOTOR_IDS",
    "RIGHT_ARM_MOTOR_IDS",
    "EXPECTED_ACTIVE_MOTOR_IDS",
    "FORBIDDEN_MOTOR_IDS",
    "GRIPPER_MOTOR_ID",
    "GRIPPER_MOTOR_NAME",
    "GRIPPER_OPEN_POSITION",
    "GRIPPER_CLOSED_POSITION",
    "DEFAULT_KP",
    "DEFAULT_KD",
    # exceptions
    "CuroboSdkError",
    "RobotConnectionError",
    "ControlModeError",
    "InitStateError",
    "LimitViolationError",
    "FeedbackError",
    "CommandError",
    "CalibrationError",
    "ConfigurationError",
]