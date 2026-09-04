"""Unit tests for curobo_sdk.fake_robot (FakeSDKRobot state machine + commands)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_sdk.fake_robot import (
    EtherCAT_Motor_Index,
    FakeControl,
    FakeSDKRobot,
    InitState,
    MotorControlMode,
)
from curobo_sdk.constants import GRIPPER_MOTOR_ID


def _motor_of_name(name: str) -> int:
    return int(getattr(EtherCAT_Motor_Index, name))


def test_initial_state_disconnected_and_uninitialized():
    robot = FakeSDKRobot()
    assert not robot.isRobotConnected()
    assert robot.getCurrentMode() == MotorControlMode.UNINITIALIZED
    assert robot.getInitState() == InitState.Uninit
    ok, state = robot.getMotorState(_motor_of_name("MOTOR_LIFT"))
    assert not ok or state is not None


def test_connect_and_mode_lifecycle():
    robot = FakeSDKRobot()
    assert robot.robot_connect() is True
    assert robot.isRobotConnected()
    assert robot.switchControlMode(MotorControlMode.LOW_LEVEL) is True
    assert robot.getCurrentMode() == MotorControlMode.LOW_LEVEL
    assert robot.switchControlMode(MotorControlMode.HIGH_LEVEL) is True
    assert robot.getCurrentMode() == MotorControlMode.HIGH_LEVEL
    assert robot.robot_init() is True
    assert robot.getInitState() == InitState.Init_Complete
    assert robot.robot_deinit() is True
    assert robot.getInitState() == InitState.Deinit_Complete


def test_commands_rejected_when_disconnected_or_wheel():
    robot = FakeSDKRobot()
    control = FakeControl(position=0.2)
    # Not connected -> rejected.
    assert robot.setWaist_low(_motor_of_name("MOTOR_WAIST_DOWN"), control) is False
    robot.robot_connect()
    # Wheel IDs 0/1 are forbidden.
    assert robot.setArm_low(_motor_of_name("MOTOR_WHEEL_LEFT"), control) is False


def test_command_writes_internal_state():
    robot = FakeSDKRobot()
    robot.robot_connect()
    arm_id = _motor_of_name("MOTOR_LEFT_ARM_1")
    control = FakeControl(position=0.75, speed=0.2, torque=0.1, kp=5.0, kd=0.2)
    assert robot.setArm_low(arm_id, control) is True
    ok, state = robot.getMotorState(arm_id)
    assert ok
    assert state.Position_Actual == 0.75
    assert state.Speed_Actual == 0.2
    recorded = robot.commands_for(arm_id)
    assert len(recorded) == 1
    assert recorded[0].Position == 0.75


def test_gripper_command_sets_hold_torque():
    robot = FakeSDKRobot()
    robot.robot_connect()
    grip_id = GRIPPER_MOTOR_ID["left"]
    control = FakeControl(position=1.5)
    assert robot.setGripper_low(grip_id, control, is_hold_torque=True) is True
    assert robot.commands_for(grip_id)[0].is_hold_torque is True


def test_error_flag_injection_is_observable():
    robot = FakeSDKRobot()
    robot.robot_connect()
    arm_id = _motor_of_name("MOTOR_LEFT_ARM_2")
    robot.set_error_flag(arm_id, 0x20)
    ok, state = robot.getMotorState(arm_id)
    assert ok and state.Error_flag == 0x20


def test_initial_position_seeded():
    initial = np.full(17, 0.3)
    robot = FakeSDKRobot(initial_position=initial)
    robot.robot_connect()
    lift_id = _motor_of_name("MOTOR_LIFT")
    assert robot.getMotorState(lift_id)[1].Position_Actual == 0.3


def test_bad_initial_position_shape_rejected():
    with pytest.raises(ValueError):
        FakeSDKRobot(initial_position=np.zeros(3))