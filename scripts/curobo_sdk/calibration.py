"""Joint calibration and motor<->model conversions for LOW_LEVEL execution.

Pure numpy; no robot SDK / cuRobo dependency.  Mirrors the reference
``joint_state_bridge.JointCalibration`` conversions so an accepted plan's
model joint positions can be faithfully commanded and its feedback read back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .constants import (
    FEEDBACK_LIMIT_TOLERANCE,
    NUM_ACTIVE_JOINTS,
    ZERITH_ACTIVE_JOINTS,
)
from .exceptions import CalibrationError


@dataclass(frozen=True)
class JointCalibration:
    """Per-joint direction and motor zero-offset calibration."""

    joint_names: tuple[str, ...]
    sign: np.ndarray
    zero_offset: np.ndarray
    verified: bool = False

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        sign = np.asarray(self.sign, dtype=np.float64)
        offset = np.asarray(self.zero_offset, dtype=np.float64)
        if names != tuple(ZERITH_ACTIVE_JOINTS):
            raise CalibrationError(
                "Calibration joint_names must match Zerith's 17-joint order"
            )
        if sign.shape != (len(names),) or offset.shape != (len(names),):
            raise CalibrationError(
                f"Calibration sign/zero_offset must each have shape "
                f"({len(names)},)"
            )
        if not np.isfinite(sign).all() or not np.isfinite(offset).all():
            raise CalibrationError("Calibration contains NaN or Inf")
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise CalibrationError("Every calibration sign must be exactly -1 or +1")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "sign", sign.copy())
        object.__setattr__(self, "zero_offset", offset.copy())

    @classmethod
    def from_mappings(
        cls,
        signs: Mapping[str, float],
        zero_offsets: Mapping[str, float],
        *,
        verified: bool = False,
    ) -> "JointCalibration":
        active_set = set(ZERITH_ACTIVE_JOINTS)
        missing_sign = active_set - set(signs)
        missing_offset = active_set - set(zero_offsets)
        extra = (set(signs) | set(zero_offsets)) - active_set
        if missing_sign or missing_offset or extra:
            raise CalibrationError(
                "Invalid calibration keys: "
                f"missing_sign={sorted(missing_sign)}, "
                f"missing_offset={sorted(missing_offset)}, extra={sorted(extra)}"
            )
        return cls(
            tuple(ZERITH_ACTIVE_JOINTS),
            np.asarray([signs[j] for j in ZERITH_ACTIVE_JOINTS], dtype=np.float64),
            np.asarray(
                [zero_offsets[j] for j in ZERITH_ACTIVE_JOINTS], dtype=np.float64
            ),
            verified=verified,
        )

    @classmethod
    def identity_unverified(cls) -> "JointCalibration":
        return cls(
            tuple(ZERITH_ACTIVE_JOINTS),
            np.ones(NUM_ACTIVE_JOINTS),
            np.zeros(NUM_ACTIVE_JOINTS),
            verified=False,
        )


def motor_to_model(
    motor_position: np.ndarray,
    calibration: JointCalibration,
) -> np.ndarray:
    """Apply ``q_model = sign * (q_motor - zero_offset)``."""

    motor_position = np.asarray(motor_position, dtype=np.float64)
    if motor_position.shape[-1] != len(calibration.joint_names):
        raise ValueError("Motor position has the wrong final dimension")
    return calibration.sign * (motor_position - calibration.zero_offset)


def model_to_motor(
    model_position: np.ndarray,
    calibration: JointCalibration,
) -> np.ndarray:
    """Apply ``q_motor = q_model / sign + zero_offset``."""

    model_position = np.asarray(model_position, dtype=np.float64)
    if model_position.shape[-1] != len(calibration.joint_names):
        raise ValueError("Model position has the wrong final dimension")
    return model_position / calibration.sign + calibration.zero_offset


def motor_velocity_to_model(
    motor_velocity: np.ndarray,
    calibration: JointCalibration,
) -> np.ndarray:
    """Apply ``q_model_dot = sign * q_motor_dot``."""

    motor_velocity = np.asarray(motor_velocity, dtype=np.float64)
    if motor_velocity.shape[-1] != len(calibration.joint_names):
        raise ValueError("Motor velocity has the wrong final dimension")
    return calibration.sign * motor_velocity


def normalize_feedback_position(
    position: np.ndarray,
    *,
    position_limits: Mapping[str, tuple[float, float]],
    joint_names: tuple[str, ...] = ZERITH_ACTIVE_JOINTS,
    tolerance: float = FEEDBACK_LIMIT_TOLERANCE,
) -> np.ndarray:
    """Clamp only tiny measured boundary jitter; leave real violations intact.

    A measured value that overshoots a soft limit by a small ``tolerance`` is
    sucked back onto the boundary (encoder-zero jitter).  Genuine out-of-limit
    values are returned unchanged so the caller can detect real violations.
    """

    value = np.asarray(position, dtype=np.float64)
    joint_names = tuple(joint_names)
    if value.shape != (len(joint_names),) or not np.isfinite(value).all():
        raise ValueError("feedback must contain finite model positions")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("feedback limit tolerance must be finite and non-negative")
    result = value.copy()
    for index, joint_name in enumerate(joint_names):
        lower, upper = position_limits[joint_name]
        current = float(result[index])
        if lower - tolerance <= current < lower:
            result[index] = lower
        elif upper < current <= upper + tolerance:
            result[index] = upper
    return result