"""Thin LOW_LEVEL wrapper around the Zerith H1Robot SDK (pinocchio_ik edition).

Minimal, self-contained low-level driver used by the pinocchio low-level replay:
connect / switch to LOW_LEVEL / init, read 17-axis feedback, command joints per
tick (``setWaist_low`` / ``setArm_low``), open/close the gripper
(``setGripper_low``), and read the SDK arm EEF pose (``getHandRelative``) that
the ``S_T_B`` calibration consumes.

Intentionally minimal: no 500 Hz executor thread, no interpolation, no soft-limit
validation (the caller's job).  Keeps ``pinocchio_ik`` free of a ``curobo_sdk`` /
``end2end_pipeline`` dependency; the real SDK is imported lazily so this module
imports without a zerith env and can be driven through an injected stub.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirror curobo_sdk.constants / curobo_planning.constants)
# ---------------------------------------------------------------------------
# Active-joint model order: waist then left arm then right arm (17 axes).
ACTIVE_JOINTS = (
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
NUM_ACTIVE_JOINTS = len(ACTIVE_JOINTS)  # 17

# 7-DoF arm joints per side (subset of ACTIVE_JOINTS).
ARM_JOINTS = {
    "left": ACTIVE_JOINTS[3:10],
    "right": ACTIVE_JOINTS[10:17],
}

# Joint -> SDK motor-name mapping (mirror reference joint_state_bridge).
JOINT_TO_MOTOR_NAME = {
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

# Motor ID ranges (see EtherCAT_Motor_Index in the SDK stub).  Wheel IDs 0/1 are
# forbidden for active commands.
WAIST_MOTOR_IDS = (2, 3, 4)
LEFT_ARM_MOTOR_IDS = tuple(range(7, 14))
RIGHT_ARM_MOTOR_IDS = tuple(range(15, 22))
GRIPPER_MOTOR_ID = {"left": 14, "right": 22}
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 1.5

# Motor_Control sentinels (KP/KD = -1 means "use controller default").
DEFAULT_KP = -1.0
DEFAULT_KD = -1.0

Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Feedback:
    """17-axis feedback read from the robot, in model value order."""

    timestamp: float
    model_position: np.ndarray  # (17,)
    model_velocity: np.ndarray  # (17,)
    motor_position: np.ndarray  # (17,) raw motor positions
    motor_torque: np.ndarray  # (17,)
    error_flags: np.ndarray  # (17,) int


class ZerithLowLevel:
    """Minimal low-level driver for one Zerith H1Robot session.

    Wraps a real ``H1Robot`` (with its ``sdk_module`` exposing the enums and
    ``Motor_Control``) so the pinocchio low-level replay can execute joint-space
    commands without depending on ``curobo_sdk``.  Methods are kept thin: state
    validation / trajectory interpolation are the caller's responsibility.
    """

    def __init__(
        self,
        robot,
        sdk_module,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._robot = robot
        self._sdk = sdk_module
        self._clock = clock
        self._motor_ids: dict[str, object] = {}
        self._motor_ids_mirror = JOINT_TO_MOTOR_NAME
        self._resolve_motor_ids()

    # -- helpers ----------------------------------------------------------------

    def _resolve_motor_ids(self) -> None:
        motor_enum = self._sdk.EtherCAT_Motor_Index
        for joint, motor_name in self._motor_ids_mirror.items():
            self._motor_ids[joint] = getattr(motor_enum, motor_name)

    def _arm_side(self, arm: str) -> str:
        return "LEFT" if str(arm).lower().startswith("l") else "RIGHT"

    def _expected_mode_enum(self):
        # pybind11 binding rejects plain ints: pass the actual LOW_LEVEL member.
        return self._sdk.MotorControlMode.LOW_LEVEL

    def _expected_init_state(self) -> int:
        init_state = getattr(self._sdk, "InitState", None)
        return int(getattr(init_state, "Init_Complete", 2))

    # -- lifecycle ----------------------------------------------------------------

    def connect(self) -> None:
        ok = bool(self._robot.robot_connect())
        if not ok:
            raise RuntimeError("robot_connect returned False")

    def switch_low_level(self) -> None:
        ok = bool(self._robot.switchControlMode(self._expected_mode_enum()))
        if not ok:
            raise RuntimeError("switchControlMode(LOW_LEVEL) returned False")

    def robot_init(self) -> None:
        if self._robot.getInitState() == self._expected_init_state():
            return  # already initialized; reuse rather than re-init
        if not bool(self._robot.robot_init()):
            raise RuntimeError("robot_init returned False")
        if self._robot.getInitState() != self._expected_init_state():
            raise RuntimeError(
                f"robot_init did not reach Init_Complete: "
                f"current={self._robot.getInitState()}"
            )

    def ensure_connected_low_level(self, connect: bool = False, init: bool = False) -> None:
        """Bring up the robot in LOW_LEVEL, optionally connect and init."""
        if connect:
            self.connect()
        if init:
            self.switch_low_level()
            self.robot_init()

    def robot_deinit(self) -> None:
        self._robot.robot_deinit()

    def is_connected(self) -> bool:
        return bool(self._robot.isRobotConnected())

    def get_current_mode(self) -> int:
        return int(self._robot.getCurrentMode())

    def get_init_state(self) -> int:
        return int(self._robot.getInitState())

    # -- feedback ---------------------------------------------------------------

    def read_feedback(self) -> Feedback:
        """Read the 17 active joints once (per ``ACTIVE_JOINTS`` order)."""
        position: list[float] = []
        velocity: list[float] = []
        torque: list[float] = []
        flags: list[int] = []
        for joint in ACTIVE_JOINTS:
            motor_id = self._motor_ids[joint]
            ok, state = self._robot.getMotorState(motor_id)
            if not ok or state is None:
                raise RuntimeError(f"feedback failed for {joint}")
            position.append(float(state.Position_Actual))
            velocity.append(float(state.Speed_Actual))
            torque.append(float(state.Torque_Actual))
            flags.append(int(state.Error_flag))
        return Feedback(
            timestamp=float(self._clock()),
            model_position=np.asarray(position, dtype=np.float64),
            model_velocity=np.asarray(velocity, dtype=np.float64),
            motor_position=np.asarray(position, dtype=np.float64),
            motor_torque=np.asarray(torque, dtype=np.float64),
            error_flags=np.asarray(flags, dtype=np.int64),
        )

    def read_arm_joints(self, arm: str) -> np.ndarray | None:
        """Read the arm's 7 joint motor positions [rad] directly from the SDK."""
        prefix = self._arm_side(arm)
        eci = self._sdk.EtherCAT_Motor_Index
        q: list[float] = []
        for i in range(1, 8):
            mid = getattr(eci, f"MOTOR_{prefix}_ARM_{i}", None)
            if mid is None:
                return None
            ok, info = self._robot.getMotorState(mid)
            if not ok or info is None:
                return None
            q.append(float(info.Position_Actual))
        return np.asarray(q, dtype=np.float64)

    def read_sdk_eef(self, arm: str) -> np.ndarray | None:
        """Return the SDK EEF pose ``S_T_E`` (4x4) via ``getHandRelative``.

        Expressed in the arm motor-zero frame, matching ``setArm_high`` targets.
        Returns None when the backing robot does not expose ``getHandRelative``.
        """
        getter = getattr(self._robot, "getHandRelative", None)
        if getter is None:
            return None
        arm_action = self._sdk.ArmAction
        arm_enum = arm_action.LEFT_ARM if str(arm).lower().startswith("l") else arm_action.RIGHT_ARM
        ok, arm_state = getter(arm_enum)
        if not ok or arm_state is None:
            return None
        pos = getattr(arm_state, "position", None)
        quat = getattr(arm_state, "rotation", None)
        if pos is None or quat is None:
            return None
        from scipy.spatial.transform import Rotation as _R
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _R.from_quat(quat).as_matrix()
        pose[:3, 3] = np.asarray(pos, dtype=np.float64)
        return pose

    # -- commanding -------------------------------------------------------------

    def _make_control(self, position: float, speed: float):
        control_type = getattr(self._sdk, "Motor_Control", None)
        if control_type is None:
            raise RuntimeError("sdk module exposes no Motor_Control")
        control = control_type()
        control.Position = float(position)
        control.Speed = float(speed)
        control.Torque = 0.0
        control.KP = float(DEFAULT_KP)
        control.KD = float(DEFAULT_KD)
        return control

    def command_joints(self, model_position: np.ndarray, model_velocity: np.ndarray | None = None) -> None:
        """Send all 17 joints once via ``setWaist_low``/``setArm_low``."""
        position = np.asarray(model_position, dtype=np.float64)
        velocity = np.zeros(NUM_ACTIVE_JOINTS, dtype=np.float64)
        if model_velocity is not None:
            velocity = np.asarray(model_velocity, dtype=np.float64)
        control = self._make_control(0.0, 0.0)
        failures: list[str] = []
        for index, joint in enumerate(ACTIVE_JOINTS):
            motor_id = int(self._motor_ids[joint])
            control.Position = float(position[index])
            control.Speed = float(velocity[index])
            method = (
                self._robot.setWaist_low
                if motor_id in WAIST_MOTOR_IDS
                else self._robot.setArm_low
            )
            try:
                ok = bool(method(self._motor_ids[joint], control))
            except Exception as exc:
                failures.append(f"{joint}: exception {exc}")
                continue
            if not ok:
                failures.append(f"{joint}: returned false")
        if failures:
            raise RuntimeError(f"joint command failures: {'; '.join(failures)}")

    def set_gripper(self, arm: str, position: float) -> None:
        """Command the gripper to ``position`` with hold-torque MIT control."""
        gid = int(GRIPPER_MOTOR_ID[arm])
        control = self._make_control(float(position), 0.0)
        if not bool(self._robot.setGripper_low(gid, control, is_hold_torque=True)):
            raise RuntimeError(f"setGripper_low({arm}) returned False")

    def set_gripper_open(self, arm: str) -> None:
        self.set_gripper(arm, GRIPPER_OPEN_POSITION)

    def set_gripper_close(self, arm: str) -> None:
        self.set_gripper(arm, GRIPPER_CLOSED_POSITION)

    # -- context manager ----------------------------------------------------------

    def close(self) -> None:
        if self.is_connected():
            try:
                self.robot_deinit()
            except Exception:
                pass

    def __enter__(self) -> "ZerithLowLevel":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_low_level_robot(*, robot=None, sdk_module=None) -> "ZerithLowLevel":
    """Build a :class:`ZerithLowLevel` bound to the real SDK.

    Args:
        robot: An already-created SDK ``H1Robot`` instance.  When None, a fresh
            ``H1Robot`` is instantiated.
        sdk_module: The SDK module exposing enums / ``Motor_Control``.  When None,
            ``lib_h1_sdk_python`` is imported lazily.
    """
    if sdk_module is None:
        import importlib
        sdk_module = importlib.import_module("lib_h1_sdk_python")
    if robot is None:
        robot = sdk_module.H1Robot()
    return ZerithLowLevel(robot, sdk_module)