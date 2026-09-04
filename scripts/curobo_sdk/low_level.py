"""Thin LOW_LEVEL wrapper around the Zerith H1Robot SDK.

Provides the low-level primitives end2end needs to execute a cuRobo joint
trajectory: connect / switch mode / init, read 17-axis feedback, command joints
per-tick (``setWaist_low`` / ``setArm_low``), open/close the gripper
(``setGripper_low``), and validate against the vendor soft position limits.

This is intentionally *thin*: it does NOT include a 500 Hz executor thread,
Hermite resampling, or deadline monitoring (those are the caller's job).  The
real SDK is loaded lazily so the module imports without a zerith env and can be
driven entirely through ``fake_robot.FakeSDKRobot`` for unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from . import calibration as _cal
from .constants import (
    EXPECTED_ACTIVE_MOTOR_IDS,
    GRIPPER_CLOSED_POSITION,
    GRIPPER_MOTOR_ID,
    GRIPPER_OPEN_POSITION,
    JOINT_TO_MOTOR_NAME,
    NUM_ACTIVE_JOINTS,
    WAIST_MOTOR_IDS,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from .exceptions import (
    CommandError,
    ConfigurationError,
    ControlModeError,
    CuroboSdkError,
    FeedbackError,
    InitStateError,
    LimitViolationError,
    RobotConnectionError,
)
from .calibration import JointCalibration

Clock = Callable[[], float]


@dataclass(frozen=True)
class MotorFeedback:
    """17-axis feedback read from the robot, already in model coordinates."""

    timestamp: float
    model_position: np.ndarray  # (17,) model joint positions
    model_velocity: np.ndarray  # (17,) model joint velocities
    motor_torque: np.ndarray  # (17,) raw motor torque
    error_flags: np.ndarray  # (17,) int error flags
    motor_position: np.ndarray  # (17,) raw motor positions


@dataclass(frozen=True)
class GripperFeedback:
    """Feedback after a gripper command."""

    position: float
    speed: float
    torque: float
    error_flag: int
    elapsed_s: float


class LowLevelRobot:
    """Low-level driver for one robot session's 17-joint trajectory execution.

    Accepts either a fake robot (unit tests) or the real ``H1Robot`` plus its
    ``sdk_module`` (hardware).  All three of ``robot``/``sdk_module``/
    ``calibration`` must be provided together for a working driver; otherwise a
    fake path is assumed and they are loaded lazily/filled with defaults.
    """

    def __init__(
        self,
        robot=None,
        *,
        sdk_module=None,
        calibration: JointCalibration | None = None,
        position_limits: Mapping[str, tuple[float, float]] | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if (robot is None) != (sdk_module is None):
            raise ConfigurationError("robot and sdk_module must be provided together")
        try:
            enum_is_int = isinstance(sdk_module.EtherCAT_Motor_Index, type)
        except AttributeError:
            enum_is_int = True
        del enum_is_int

        self._robot = robot
        self._sdk = sdk_module
        self._calibration = calibration or JointCalibration.identity_unverified()
        self._position_limits = dict(
            position_limits or ZERITH_SOFTWARE_POSITION_LIMITS
        )
        self._clock = clock

        self._motor_ids: dict[str, object] = {}
        self._prepared = False
        if self._robot is not None and self._sdk is not None:
            self._motor_ids = self._resolve_motor_ids()
            self._prepared = True

    # -- helpers ---------------------------------------------------------------

    def _resolve_motor_ids(self) -> dict[str, object]:
        motor_enum = self._sdk.EtherCAT_Motor_Index
        result: dict[str, object] = {}
        for joint, motor_name in JOINT_TO_MOTOR_NAME.items():
            motor_id = getattr(motor_enum, motor_name)
            result[joint] = motor_id
        ids = tuple(int(motor_id) for motor_id in result.values())
        if len(ids) != len(ZERITH_ACTIVE_JOINTS) or len(set(ids)) != len(ids):
            raise ConfigurationError(
                "The 17-axis joint-to-motor mapping is not bijective"
            )
        if set(ids) != EXPECTED_ACTIVE_MOTOR_IDS:
            raise ConfigurationError(
                "Active-joint mapping must be exactly waist IDs 2-4 and arm "
                "IDs 7-13/15-21; wheel IDs 0/1 are forbidden"
            )
        return result

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise CuroboSdkError(
                "LowLevelRobot is not prepared: pass robot, sdk_module and "
                "calibration together (or use create_low_level_robot)"
            )

    def _expected_mode(self) -> int:
        return int(self._sdk.MotorControlMode.LOW_LEVEL)

    def _expected_init_state(self) -> int:
        init_state = getattr(self._sdk, "InitState", None)
        return int(getattr(init_state, "Init_Complete", 2))

    def _check_operational_state(self) -> None:
        try:
            connected = bool(self._robot.isRobotConnected())
            mode = int(self._robot.getCurrentMode())
            init_state = int(self._robot.getInitState())
        except Exception as exc:
            raise FeedbackError(f"failed to query connection/mode/init state: {exc}") from exc
        if not connected:
            raise RobotConnectionError(
                "robot connection lost; software hold cannot be guaranteed, use physical E-stop"
            )
        if mode != self._expected_mode():
            raise ControlModeError(
                f"control mode changed from LOW_LEVEL: current={mode}"
            )
        if init_state != self._expected_init_state():
            raise InitStateError(
                f"robot init state is not Init_Complete: current={init_state}"
            )

    def _validate_position_vector(self, position: np.ndarray, *, name: str) -> np.ndarray:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (NUM_ACTIVE_JOINTS,) or not np.isfinite(value).all():
            raise ValueError(
                f"{name} must contain {NUM_ACTIVE_JOINTS} finite model positions"
            )
        violations: list[str] = []
        for index, joint_name in enumerate(ZERITH_ACTIVE_JOINTS):
            lower, upper = self._position_limits[joint_name]
            val = float(value[index])
            if not lower <= val <= upper:
                violations.append(
                    f"{joint_name}={val:.6f} outside [{lower:.6f}, {upper:.6f}]"
                )
        if violations:
            raise LimitViolationError(f"{name} exceeds soft limits: {', '.join(violations)}")
        return value

    # -- lifecycle ---------------------------------------------------------------

    def connect(self) -> None:
        self._require_prepared()
        try:
            ok = bool(self._robot.robot_connect())
        except Exception as exc:
            raise RobotConnectionError(f"robot_connect failed: {exc}") from exc
        if not ok:
            raise RobotConnectionError("robot_connect returned False")

    def switch_low_level(self) -> None:
        self._require_prepared()
        try:
            ok = bool(self._robot.switchControlMode(self._expected_mode()))
        except Exception as exc:
            raise ControlModeError(f"switchControlMode(LOW_LEVEL) failed: {exc}") from exc
        if not ok:
            raise ControlModeError("switchControlMode(LOW_LEVEL) returned False")

    def robot_init(self) -> None:
        self._require_prepared()
        if self._robot.getInitState() == self._expected_init_state():
            return  # already initialized; reuse rather than re-init
        try:
            ok = bool(self._robot.robot_init())
        except Exception as exc:
            raise InitStateError(f"robot_init failed: {exc}") from exc
        if not ok:
            raise InitStateError("robot_init returned False")
        if self._robot.getInitState() != self._expected_init_state():
            raise InitStateError(
                f"robot_init did not reach Init_Complete: current={self._robot.getInitState()}"
            )

    def ensure_connected_low_level(self, connect: bool = False, init: bool = False) -> None:
        """Bring up the robot in LOW_LEVEL, optionally connect and init."""

        if connect:
            self.connect()
        self._require_prepared()
        if init:
            self.switch_low_level()
            self.robot_init()
        self._check_operational_state()

    def robot_deinit(self) -> None:
        self._require_prepared()
        try:
            self._robot.robot_deinit()
        except Exception as exc:
            raise CuroboSdkError(f"robot_deinit failed: {exc}") from exc

    # -- introspection ----------------------------------------------------------

    def is_connected(self) -> bool:
        return bool(self._robot.isRobotConnected())

    def get_current_mode(self) -> int:
        return int(self._robot.getCurrentMode())

    def get_init_state(self) -> int:
        return int(self._robot.getInitState())

    # -- feedback ---------------------------------------------------------------

    def read_feedback(self) -> MotorFeedback:
        self._require_prepared()
        self._check_operational_state()
        position: list[float] = []
        velocity: list[float] = []
        torque: list[float] = []
        flags: list[int] = []
        for joint_name in ZERITH_ACTIVE_JOINTS:
            motor_id = self._motor_ids[joint_name]
            try:
                ok, state = self._robot.getMotorState(motor_id)
            except Exception as exc:
                raise FeedbackError(f"feedback exception for {joint_name}: {exc}") from exc
            if not ok:
                raise FeedbackError(f"feedback failed for {joint_name}")
            position.append(float(state.Position_Actual))
            velocity.append(float(state.Speed_Actual))
            torque.append(float(state.Torque_Actual))
            flags.append(int(state.Error_flag))
        motor_q = np.asarray(position, dtype=np.float64)
        motor_qd = np.asarray(velocity, dtype=np.float64)
        torque_array = np.asarray(torque, dtype=np.float64)
        flag_array = np.asarray(flags, dtype=np.int64)
        if not (
            np.isfinite(motor_q).all()
            and np.isfinite(motor_qd).all()
            and np.isfinite(torque_array).all()
        ):
            raise FeedbackError("feedback contains NaN or Inf")
        bad = {
            ZERITH_ACTIVE_JOINTS[index]: int(flag)
            for index, flag in enumerate(flag_array)
            if flag != 0
        }
        if bad:
            raise FeedbackError(f"active motor Error_flag is non-zero: {bad}")
        model_q = _cal.motor_to_model(motor_q, self._calibration)
        model_qd = _cal.motor_velocity_to_model(motor_qd, self._calibration)
        model_q = _cal.normalize_feedback_position(
            model_q,
            position_limits=self._position_limits,
        )
        self._validate_position_vector(model_q, name="feedback")
        return MotorFeedback(
            timestamp=float(self._clock()),
            model_position=model_q,
            model_velocity=model_qd,
            motor_torque=torque_array,
            error_flags=flag_array,
            motor_position=motor_q,
        )

    # -- commanding ---------------------------------------------------------------

    def _resolve_control_type(self):
        sdk = self._sdk
        control_type = getattr(sdk, "Motor_Control", None)
        if control_type is None:
            control_type = getattr(sdk, "FakeControl", None)
        if control_type is None:
            raise ConfigurationError(
                "sdk module exposes neither Motor_Control nor FakeControl"
            )
        return control_type

    def _make_control(self, position: float, speed: float) -> object:
        from ._control import make_control

        return make_control(
            self._resolve_control_type(),
            position=position,
            speed=speed,
        )

    def command_joints(self, model_position: np.ndarray, model_velocity: np.ndarray) -> None:
        """Send all 17 joints once via setWaist_low/setArm_low (MIT fields)."""

        self._require_prepared()
        self._validate_position_vector(model_position, name="command")
        velocity = np.asarray(model_velocity, dtype=np.float64)
        if velocity.shape != (NUM_ACTIVE_JOINTS,) or not np.isfinite(velocity).all():
            raise ValueError("command velocity must contain 17 finite values")
        motor_position = _cal.model_to_motor(model_position, self._calibration)
        motor_velocity = velocity / self._calibration.sign
        control = self._make_control(0.0, 0.0)
        failures: list[str] = []
        for index, joint_name in enumerate(ZERITH_ACTIVE_JOINTS):
            motor_id = self._motor_ids[joint_name]
            if int(motor_id) in (0, 1):
                raise CommandError(f"internal safety violation: wheel command for {joint_name}")
            control.Position = float(motor_position[index])
            control.Speed = float(motor_velocity[index])
            method = (
                self._robot.setWaist_low
                if int(motor_id) in WAIST_MOTOR_IDS
                else self._robot.setArm_low
            )
            try:
                ok = bool(method(motor_id, control))
            except Exception as exc:
                failures.append(f"{joint_name}: exception {exc}")
                continue
            if not ok:
                failures.append(f"{joint_name}: returned false")
        if failures:
            raise CommandError(f"joint command failures: {'; '.join(failures)}", failures)

    def _gripper_id(self, arm: str) -> int:
        if arm not in GRIPPER_MOTOR_ID:
            raise ValueError(f"Unsupported arm for gripper: {arm!r}")
        gripper_id = int(GRIPPER_MOTOR_ID[arm])
        if gripper_id in (0, 1):
            raise CommandError("internal safety violation: wheel command for gripper")
        return gripper_id

    def set_gripper(self, arm: str, position: float) -> GripperFeedback:
        """Command the gripper to ``position`` with hold-torque MIT control."""

        self._require_prepared()
        if not (
            GRIPPER_OPEN_POSITION <= float(position) <= GRIPPER_CLOSED_POSITION
        ):
            raise ValueError(
                f"gripper position must be in "
                f"[{GRIPPER_OPEN_POSITION}, {GRIPPER_CLOSED_POSITION}], got {position}"
            )
        gripper_id = self._gripper_id(arm)
        control = self._make_control(float(position), 0.0)
        start = self._clock()
        try:
            ok = bool(
                self._robot.setGripper_low(gripper_id, control, is_hold_torque=True)
            )
        except Exception as exc:
            raise CommandError(f"setGripper_low failed: {exc}") from exc
        if not ok:
            raise CommandError("setGripper_low returned False")
        elapsed = self._clock() - start
        try:
            ok, state = self._robot.getMotorState(gripper_id)
        except Exception as exc:
            raise FeedbackError(f"gripper feedback exception: {exc}") from exc
        if not ok or state is None:
            raise FeedbackError("gripper feedback failed")
        if int(state.Error_flag) != 0:
            raise FeedbackError(f"gripper Error_flag is non-zero: {state.Error_flag}")
        return GripperFeedback(
            position=float(state.Position_Actual),
            speed=float(state.Speed_Actual),
            torque=float(state.Torque_Actual),
            error_flag=int(state.Error_flag),
            elapsed_s=float(elapsed),
        )

    def set_gripper_open(self, arm: str) -> GripperFeedback:
        return self.set_gripper(arm, GRIPPER_OPEN_POSITION)

    def set_gripper_close(self, arm: str) -> GripperFeedback:
        return self.set_gripper(arm, GRIPPER_CLOSED_POSITION)

    def verify_position_limits(self, position: np.ndarray) -> np.ndarray:
        """Public soft-limit check (e.g. preflight a planned trajectory)."""

        return self._validate_position_vector(position, name="position")

    # -- context manager ----------------------------------------------------------

    def close(self) -> None:
        if self._prepared and self.is_connected():
            try:
                self.robot_deinit()
            except Exception:
                pass
        self._prepared = False

    def __enter__(self) -> "LowLevelRobot":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()