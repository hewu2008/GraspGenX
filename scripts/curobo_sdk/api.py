"""Public facade for the curobo_sdk LOW_LEVEL driver.

``create_low_level_robot`` is the single entry point end2end uses to obtain a
:class:`LowLevelRobot` bound either to the real Zerith SDK or to a fake robot
(unit tests / dry runs / sim).  The real SDK module is imported lazily so this
module (and the whole package) imports without a zerith env or the ``.so``.
"""

from __future__ import annotations

import importlib
import time
from typing import Mapping

from .calibration import JointCalibration
from .exceptions import ConfigurationError
from .low_level import LowLevelRobot


def _load_real_sdk_module():
    """Lazily import the real SDK (only present in the zerith env)."""
    return importlib.import_module("lib_h1_sdk_python")


def create_low_level_robot(
    *,
    fake: bool = False,
    robot=None,
    sdk_module=None,
    calibration: JointCalibration | None = None,
    position_limits: Mapping[str, tuple[float, float]] | None = None,
    clock=time.monotonic,
) -> LowLevelRobot:
    """Build a :class:`LowLevelRobot` backed by real or fake hardware.

    Args:
        fake: When True, bind a ``FakeSDKRobot`` with identity calibration so
            the driver is fully exercisable without hardware or the zerith env.
        robot: An already-created SDK ``H1Robot`` instance (real path).  When
            None and ``fake`` is False, a fresh ``H1Robot`` is instantiated.
        sdk_module: The SDK module exposing enums / ``Motor_Control``.  When
            None and ``fake`` is False, ``lib_h1_sdk_python`` is imported
            lazily.  For ``fake=True`` it is ignored and the fake module used.
        calibration: Optional :class:`JointCalibration`; defaults to identity
            (unverified) for both paths.
        position_limits: Optional per-joint soft-limit override; defaults to
            the mirrored ``ZERITH_SOFTWARE_POSITION_LIMITS``.
        clock: Optional monotonic clock for feedback timestamps.

    Returns:
        A prepared :class:`LowLevelRobot` (real or fake).
    """
    if fake:
        from . import fake_robot as _sdk_module

        if robot is not None or sdk_module is not None:
            raise ConfigurationError(
                "fake=True ignores explicitly provided robot/sdk_module"
            )
        return LowLevelRobot(
            _sdk_module.FakeSDKRobot(),
            sdk_module=_sdk_module,
            calibration=calibration or JointCalibration.identity_unverified(),
            position_limits=position_limits,
            clock=clock,
        )

    if sdk_module is None:
        sdk_module = _load_real_sdk_module()
    if robot is None:
        robot = sdk_module.H1Robot()
    return LowLevelRobot(
        robot,
        sdk_module=sdk_module,
        calibration=calibration or JointCalibration.identity_unverified(),
        position_limits=position_limits,
        clock=clock,
    )