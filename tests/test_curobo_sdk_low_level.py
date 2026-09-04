"""End-to-end tests of LowLevelRobot driven through the FakeSDKRobot.

No hardware / zerith env / CUDA needed.  Exercises lifecycle, mode & init-state
guards, feedback conversion + error flags, soft-limit validation, the motor-ID
whitelist, and gripper open/close.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_sdk import fake_robot
from curobo_sdk.calibration import JointCalibration
from curobo_sdk.constants import (
    EXPECTED_ACTIVE_MOTOR_IDS,
    GRIPPER_MOTOR_ID,
    NUM_ACTIVE_JOINTS,
    ZERITH_ACTIVE_JOINTS,
)
from curobo_sdk.exceptions import (
    CommandError,
    ControlModeError,
    CuroboSdkError,
    FeedbackError,
    InitStateError,
    LimitViolationError,
)
from curobo_sdk.low_level import LowLevelRobot
from curobo_sdk.fake_robot import EtherCAT_Motor_Index, MotorControlMode


def _robot_and_robot(initial=None, error_flags=None):
    robot = fake_robot.FakeSDKRobot(
        initial_position=initial, error_flags=error_flags
    )
    low = LowLevelRobot(
        robot,
        sdk_module=fake_robot,
        calibration=JointCalibration.identity_unverified(),
    )
    return low, robot


def _bringup(low):
    low.ensure_connected_low_level(connect=True, init=True)


def _zero_position():
    return np.zeros(NUM_ACTIVE_JOINTS)


def test_not_prepared_raises():
    low = LowLevelRobot()  # nothing injected -> not prepared
    with pytest.raises(CuroboSdkError):
        low.read_feedback()
    with pytest.raises(CuroboSdkError):
        low.command_joints(_zero_position(), _zero_position())
    with pytest.raises(CuroboSdkError):
        low.connect()


def test_mismatched_robot_and_sdk_rejected():
    with pytest.raises(Exception):
        LowLevelRobot(fake_robot.FakeSDKRobot(), sdk_module=None)
    with pytest.raises(Exception):
        LowLevelRobot(None, sdk_module=fake_robot)


def test_connect_switches_low_level_and_init():
    low, robot = _robot_and_robot()
    assert not robot.isRobotConnected()
    low.ensure_connected_low_level(connect=True, init=True)
    assert robot.isRobotConnected()
    assert robot.getCurrentMode() == MotorControlMode.LOW_LEVEL
    assert low.is_connected()
    assert low.get_current_mode() == MotorControlMode.LOW_LEVEL


def test_wrong_mode_read_raises_control_mode_error():
    low, robot = _robot_and_robot()
    _bringup(low)
    robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
    with pytest.raises(ControlModeError):
        low.read_feedback()


def test_uninitialized_read_raises_init_state_error():
    low, robot = _robot_and_robot()
    low.ensure_connected_low_level(connect=True, init=True)
    robot.robot_deinit()  # init_state -> Deinit_Complete, mode stays LOW_LEVEL
    with pytest.raises(InitStateError):
        low.read_feedback()


def test_feedback_round_trip_with_calibration():
    signs = {name: (1.0 if i % 2 == 0 else -1.0)
             for i, name in enumerate(ZERITH_ACTIVE_JOINTS)}
    offsets = {name: float(i) * 0.1 for i, name in enumerate(ZERITH_ACTIVE_JOINTS)}
    cal = JointCalibration.from_mappings(signs, offsets)
    model_q = np.array([0.2 + 0.0 * i for i in range(NUM_ACTIVE_JOINTS)],
                       dtype=float)
    motor_q = model_q / cal.sign + cal.zero_offset

    robot = fake_robot.FakeSDKRobot(initial_position=motor_q)
    low = LowLevelRobot(robot, sdk_module=fake_robot, calibration=cal)
    _bringup(low)
    fb = low.read_feedback()
    assert fb.model_position.shape == (NUM_ACTIVE_JOINTS,)
    assert np.allclose(fb.model_position, model_q)
    assert np.allclose(fb.motor_position, motor_q)
    assert np.allclose(fb.model_velocity, np.zeros(NUM_ACTIVE_JOINTS))


def test_feedback_error_flag_raises():
    arm_name = "MOTOR_LEFT_ARM_1"
    arm_id = int(getattr(EtherCAT_Motor_Index, arm_name))
    low, _ = _robot_and_robot(error_flags={arm_id: 0x10})
    _bringup(low)
    with pytest.raises(FeedbackError):
        low.read_feedback()


def test_command_soft_limit_violation():
    low, _ = _robot_and_robot()
    _bringup(low)
    pos = _zero_position()
    daogui_idx = ZERITH_ACTIVE_JOINTS.index("daogui_joint")
    pos[daogui_idx] = 0.9  # beyond [0.0, 0.8]
    with pytest.raises(LimitViolationError):
        low.command_joints(pos, _zero_position())


def test_command_bad_shape_rejected():
    low, _ = _robot_and_robot()
    _bringup(low)
    with pytest.raises(ValueError):
        low.command_joints(np.zeros(5), _zero_position())
    with pytest.raises(ValueError):
        low.command_joints(_zero_position(), np.zeros(5))


def test_command_whitelist_no_wheels_or_grippers():
    low, robot = _robot_and_robot()
    _bringup(low)
    low.command_joints(_zero_position(), _zero_position())
    commanded = robot.commanded_ids
    assert commanded == EXPECTED_ACTIVE_MOTOR_IDS
    assert not (commanded & ({0, 1} | {14, 22}))


def test_command_writes_fake_state_identity():
    low, _ = _robot_and_robot()
    _bringup(low)
    pos = np.linspace(0.1, 0.25, NUM_ACTIVE_JOINTS)
    low.command_joints(pos, np.zeros(NUM_ACTIVE_JOINTS))
    fb = low.read_feedback()  # identity cal -> motor == model
    assert np.allclose(fb.model_position, pos)


def test_command_pending_rejects_when_disconnected():
    low, _ = _robot_and_robot()
    # not connected, but prepared -> setWaist_low/setArm_low return False
    with pytest.raises(CommandError):
        low.command_joints(_zero_position(), _zero_position())


def test_gripper_open_close_and_hold_torque():
    low, robot = _robot_and_robot()
    _bringup(low)
    arm = "left"
    fb_open = low.set_gripper_open(arm)
    assert fb_open.error_flag == 0
    recorded = robot.commands_for(GRIPPER_MOTOR_ID[arm])
    assert len(recorded) == 1
    assert recorded[0].is_hold_torque is True
    assert recorded[0].Position == 0.0  # GRIPPER_OPEN_POSITION
    low.set_gripper_close(arm)
    assert robot.commands_for(GRIPPER_MOTOR_ID[arm])[-1].Position == 1.5


def test_gripper_position_range():
    low, _ = _robot_and_robot()
    _bringup(low)
    with pytest.raises(ValueError):
        low.set_gripper("left", 5.0)
    with pytest.raises(ValueError):
        low.set_gripper("unknown_arm", 0.0)


def test_verify_position_limits_preflight():
    low, _ = _robot_and_robot()
    pos = _zero_position()
    low.verify_position_limits(pos)  # ok
    with pytest.raises(LimitViolationError):
        low.verify_position_limits(np.full(NUM_ACTIVE_JOINTS, 5.0))


def test_close_deinit_is_idempotent():
    low, _ = _robot_and_robot()
    _bringup(low)
    low.close()
    # FakeSDKRobot.robot_deinit sets Deinit_Complete but stays connected.
    assert not low._prepared
    # second close is a no-op
    low.close()