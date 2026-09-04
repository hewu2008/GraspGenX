"""Construct an SDK motor-control object (real or fake) with MIT sentinels.

The real ``lib_h1_sdk_python.Motor_Control`` and the fake ``FakeControl`` share
an identical MIT field set (``Position``/``Speed``/``Torque``/``KP``/``KD``).
``make_control`` takes the *type* (never the module) so it is agnostic to which
SDK backend ``LowLevelRobot`` is bound to; the caller resolves the type via the
injected ``sdk_module``.
"""

from __future__ import annotations

from typing import Type

from .constants import DEFAULT_KD, DEFAULT_KP


def make_control(
    control_type: Type,
    *,
    position: float = 0.0,
    speed: float = 0.0,
    torque: float = 0.0,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
):
    """Build a fresh control object with ``position``/``speed`` and defaults.

    ``control_type`` is either the SDK's ``Motor_Control`` (no-arg constructor)
    or ``FakeControl`` (all-keyword-args constructor); both are instantiated
    bare and then the MIT fields are assigned uniformly.
    """
    control = control_type()
    control.Position = float(position)
    control.Speed = float(speed)
    control.Torque = float(torque)
    control.KP = float(kp)
    control.KD = float(kd)
    return control