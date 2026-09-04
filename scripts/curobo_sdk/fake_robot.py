"""A fake Zerith SDK robot for unit-testing curobo_sdk without hardware.

``FakeSDKRobot`` exposes the same low-level call surface as the real H1Robot
(refer to ``assets/zerith/sdk/lib/lib_h1_sdk_python.pyi``): connect / mode
switch / init / deinit / getMotorState / setWaist_low / setArm_low /
setGripper_low / getCurrentMode / getInitState / isRobotConnected.  It also
provides helpers to inspect commands and inject error flags/soft limits so the
``LowLevelRobot`` wrapper can be tested end-to-end without a zerith env or the
``lib_h1_sdk_python.so``.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    FORBIDDEN_MOTOR_IDS,
    GRIPPER_MOTOR_ID,
    JOINT_TO_MOTOR_NAME,
    NUM_ACTIVE_JOINTS,
    ZERITH_ACTIVE_JOINTS,
)


class _FakeEnumMeta(type):
    """Turn a class body of int attributes into an enum-like namespace."""

    def __getattr__(cls, name: str):
        raise AttributeError(name)


class MotorControlMode(metaclass=_FakeEnumMeta):
    """Mirror ``sdk.MotorControlMode`` int values (fake-only)."""

    UNINITIALIZED = 0
    LOW_LEVEL = 1
    HIGH_LEVEL = 2
    GRAVITY_COMPENSATION_LEVEL = 3


class InitState(metaclass=_FakeEnumMeta):
    """Mirror ``sdk.InitState`` int values (fake-only)."""

    Uninit = 0
    Initializing = 1
    Init_Complete = 2
    Deinitializing = 3
    Deinit_Complete = 4
    Error_State = 5


class EtherCAT_Motor_Index(metaclass=_FakeEnumMeta):
    """Mirror ``sdk.EtherCAT_Motor_Index`` int values (fake-only)."""

    MOTOR_WHEEL_LEFT = 0
    MOTOR_WHEEL_RIGHT = 1
    MOTOR_LIFT = 2
    MOTOR_WAIST_DOWN = 3
    MOTOR_WAIST_UP = 4
    MOTOR_HEAD_DOWN = 5
    MOTOR_HEAD_UP = 6
    MOTOR_LEFT_ARM_1 = 7
    MOTOR_LEFT_ARM_2 = 8
    MOTOR_LEFT_ARM_3 = 9
    MOTOR_LEFT_ARM_4 = 10
    MOTOR_LEFT_ARM_5 = 11
    MOTOR_LEFT_ARM_6 = 12
    MOTOR_LEFT_ARM_7 = 13
    MOTOR_LEFT_ARM_8 = 14
    MOTOR_RIGHT_ARM_1 = 15
    MOTOR_RIGHT_ARM_2 = 16
    MOTOR_RIGHT_ARM_3 = 17
    MOTOR_RIGHT_ARM_4 = 18
    MOTOR_RIGHT_ARM_5 = 19
    MOTOR_RIGHT_ARM_6 = 20
    MOTOR_RIGHT_ARM_7 = 21
    MOTOR_RIGHT_ARM_8 = 22


class FakeMotorState:
    """Mirror ``Motor_Information`` fields returned by ``getMotorState``."""

    __slots__ = (
        "Position_Actual",
        "Speed_Actual",
        "Torque_Actual",
        "KP_Actual",
        "KD_Actual",
        "Error_flag",
    )

    def __init__(
        self,
        *,
        position_actual: float = 0.0,
        speed_actual: float = 0.0,
        torque_actual: float = 0.0,
        error_flag: int = 0,
        kp_actual: float = -1.0,
        kd_actual: float = -1.0,
    ) -> None:
        self.Position_Actual = float(position_actual)
        self.Speed_Actual = float(speed_actual)
        self.Torque_Actual = float(torque_actual)
        self.KP_Actual = float(kp_actual)
        self.KD_Actual = float(kd_actual)
        self.Error_flag = int(error_flag)

    def __repr__(self) -> str:
        return (
            f"FakeMotorState(Position_Actual={self.Position_Actual:.6f}, "
            f"Speed_Actual={self.Speed_Actual:.6f}, "
            f"Torque_Actual={self.Torque_Actual:.6f}, "
            f"Error_flag={self.Error_flag})"
        )


class FakeControl:
    """Mirror ``Motor_Control`` used by setWaist_low/setArm_low/setGripper_low."""

    __slots__ = ("Position", "Speed", "Torque", "KP", "KD", "is_hold_torque")

    def __init__(
        self,
        *,
        position: float = 0.0,
        speed: float = 0.0,
        torque: float = 0.0,
        kp: float = -1.0,
        kd: float = -1.0,
        is_hold_torque: bool = True,
    ) -> None:
        self.Position = float(position)
        self.Speed = float(speed)
        self.Torque = float(torque)
        self.KP = float(kp)
        self.KD = float(kd)
        self.is_hold_torque = bool(is_hold_torque)


class FakeSDKRobot:
    """In-memory stand-in for the real ``H1Robot`` low-level contract.

    Motor states are keyed by motor ID.  All 17 active joints plus the two
    grippers are precreated at ``position=0``.  The object records every
    command so tests can assert exactly what was sent.
    """

    def __init__(
        self,
        *,
        initial_position: np.ndarray | None = None,
        error_flags: dict[int, int] | None = None,
    ) -> None:
        self._connected = False
        self._mode: int = MotorControlMode.UNINITIALIZED
        self._init_state: int = InitState.Uninit
        self._states: dict[int, FakeMotorState] = {}
        self._commands: list[tuple[int, FakeControl]] = []

        initial = np.zeros(NUM_ACTIVE_JOINTS, dtype=np.float64) if initial_position is None \
            else np.asarray(initial_position, dtype=np.float64)
        if initial.shape != (NUM_ACTIVE_JOINTS,):
            raise ValueError("initial_position must have shape (17,)")
        flags = error_flags or {}
        for index, joint_name in enumerate(ZERITH_ACTIVE_JOINTS):
            motor_name = JOINT_TO_MOTOR_NAME[joint_name]
            motor_id = getattr(EtherCAT_Motor_Index, motor_name)
            self._states[motor_id] = FakeMotorState(
                position_actual=float(initial[index]),
                speed_actual=0.0,
                error_flag=int(flags.get(motor_id, 0)),
            )
        # Gripper motors start open.
        for gripper_id in GRIPPER_MOTOR_ID.values():
            self._states[gripper_id] = FakeMotorState(position_actual=0.0)

    # -- connection / mode / init -------------------------------------------

    def robot_connect(self) -> bool:
        self._connected = True
        self._mode = MotorControlMode.UNINITIALIZED
        self._init_state = InitState.Uninit
        return True

    def switchControlMode(self, new_mode: int) -> bool:
        if not self._connected:
            return False
        self._mode = int(new_mode)
        return True

    def robot_init(self) -> bool:
        if not self._connected:
            return False
        self._init_state = InitState.Init_Complete
        return True

    def robot_deinit(self) -> bool:
        if not self._connected:
            return False
        self._init_state = InitState.Deinit_Complete
        return True

    def isRobotConnected(self) -> bool:
        return self._connected

    def getCurrentMode(self) -> int:
        return self._mode

    def getInitState(self) -> int:
        return self._init_state

    # -- feedback -------------------------------------------------------------

    def getMotorState(self, motor_id: int) -> tuple[bool, FakeMotorState | None]:
        if not self._connected or motor_id not in self._states:
            return (False, None)
        return (True, self._states[motor_id])

    # -- commanding -----------------------------------------------------------

    def _apply(self, motor_id: int, control: FakeControl) -> bool:
        if not self._connected or motor_id in FORBIDDEN_MOTOR_IDS:
            return False
        if motor_id not in self._states:
            self._states[motor_id] = FakeMotorState()
        state = self._states[motor_id]
        state.Position_Actual = control.Position
        state.Speed_Actual = control.Speed
        state.Torque_Actual = control.Torque
        self._commands.append((motor_id, control))
        return True

    def setWaist_low(self, waist_id: int, control: FakeControl) -> bool:
        return self._apply(int(waist_id), control)

    def setArm_low(self, arm_id: int, control: FakeControl) -> bool:
        return self._apply(int(arm_id), control)

    def setGripper_low(
        self, gripper_id: int, control: FakeControl, is_hold_torque: bool = True
    ) -> bool:
        control.is_hold_torque = bool(is_hold_torque)
        return self._apply(int(gripper_id), control)

    # -- test helpers ---------------------------------------------------------

    def set_error_flag(self, motor_id: int, flag: int) -> None:
        if motor_id in self._states:
            self._states[motor_id].Error_flag = int(flag)

    def commands_for(self, motor_id: int) -> list[FakeControl]:
        return [control for mid, control in self._commands if mid == motor_id]

    @property
    def all_commands(self) -> list[tuple[int, FakeControl]]:
        return list(self._commands)

    @property
    def commanded_ids(self) -> set[int]:
        return {motor_id for motor_id, _ in self._commands}