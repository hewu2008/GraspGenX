"""Unit tests for curobo_sdk.calibration (motor<->model conversions)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_sdk.calibration import (
    JointCalibration,
    motor_to_model,
    motor_velocity_to_model,
    model_to_motor,
    normalize_feedback_position,
)
from curobo_sdk.constants import (
    FEEDBACK_LIMIT_TOLERANCE,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from curobo_sdk.exceptions import CalibrationError


def _cal(signs=None):
    n = len(ZERITH_ACTIVE_JOINTS)
    sign = np.ones(n) if signs is None else np.asarray(signs, dtype=float)
    offset = np.linspace(0.0, 0.5, n)
    return JointCalibration(tuple(ZERITH_ACTIVE_JOINTS), sign, offset)


def test_identity_round_trip():
    cal = JointCalibration.identity_unverified()
    q_model = np.linspace(-1.0, 1.0, 17)
    assert np.allclose(model_to_motor(q_model, cal), q_model)
    assert np.allclose(motor_to_model(model_to_motor(q_model, cal), cal), q_model)
    assert not cal.verified


def test_sign_and_offset_round_trip():
    signs = np.where(np.arange(17) % 2 == 0, 1.0, -1.0)
    cal = _cal(signs)
    q_model = np.linspace(-2.0, 2.0, 17)
    q_motor = model_to_motor(q_model, cal)
    assert np.allclose(motor_to_model(q_motor, cal), q_model)
    # model_to_motor: q_motor = q_model / sign + offset
    assert np.allclose(q_motor, q_model / cal.sign + cal.zero_offset)


def test_velocity_conversion():
    signs = np.where(np.arange(17) % 2 == 0, 1.0, -1.0)
    cal = _cal(signs)
    motor_qd = np.linspace(0.1, 0.9, 17)
    assert np.allclose(
        motor_velocity_to_model(motor_qd, cal), cal.sign * motor_qd
    )


def test_from_mappings_missing_keys():
    with pytest.raises(CalibrationError):
        JointCalibration.from_mappings(
            {name: 1.0 for name in ZERITH_ACTIVE_JOINTS[:-1]},
            {name: 0.0 for name in ZERITH_ACTIVE_JOINTS},
        )


def test_from_mappings_bad_sign():
    signs = {name: 2.0 for name in ZERITH_ACTIVE_JOINTS}
    offsets = {name: 0.0 for name in ZERITH_ACTIVE_JOINTS}
    with pytest.raises(CalibrationError):
        JointCalibration.from_mappings(signs, offsets)


def test_shape_mismatch():
    with pytest.raises(CalibrationError):
        JointCalibration(tuple(ZERITH_ACTIVE_JOINTS), np.ones(15), np.zeros(17))


def test_wrong_order_rejected():
    with pytest.raises(CalibrationError):
        JointCalibration(tuple(reversed(ZERITH_ACTIVE_JOINTS)), np.ones(17), np.zeros(17))


def test_normalize_sucks_tiny_jitter_but_leaves_real_violation():
    val = np.zeros(17)
    # daogui_joint is [0.0, 0.8]; a tiny negative jitter should be sucked to 0.
    val[0] = -FEEDBACK_LIMIT_TOLERANCE / 2
    out = normalize_feedback_position(
        val, position_limits=ZERITH_SOFTWARE_POSITION_LIMITS
    )
    assert out[0] == 0.0
    # A real violation beyond the tolerance stays unchanged.
    val[0] = -1.0
    out2 = normalize_feedback_position(
        val, position_limits=ZERITH_SOFTWARE_POSITION_LIMITS
    )
    assert out2[0] == -1.0