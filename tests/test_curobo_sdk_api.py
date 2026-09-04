"""Tests for the create_low_level_robot factory (fake path end-to-end)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_sdk.api import create_low_level_robot
from curobo_sdk.constants import NUM_ACTIVE_JOINTS
from curobo_sdk.exceptions import ConfigurationError


def test_fake_factory_full_smoke():
    with create_low_level_robot(fake=True) as low:
        low.ensure_connected_low_level(connect=True, init=True)
        assert low.is_connected()
        fb = low.read_feedback()
        assert fb.model_position.shape == (NUM_ACTIVE_JOINTS,)
        pos = np.zeros(NUM_ACTIVE_JOINTS)
        low.command_joints(pos, np.zeros(NUM_ACTIVE_JOINTS))
        low.set_gripper_open("left")
        low.verify_position_limits(pos)


def test_fake_ignores_explicit_robot_and_sdk():
    with pytest.raises(ConfigurationError):
        create_low_level_robot(fake=True, robot=object())
    with pytest.raises(ConfigurationError):
        create_low_level_robot(fake=True, sdk_module=object())


def test_real_factory_with_fake_sdk_module():
    from curobo_sdk import fake_robot

    robot = create_low_level_robot(
        fake=False, robot=fake_robot.FakeSDKRobot(), sdk_module=fake_robot
    )
    with robot as low:
        low.ensure_connected_low_level(connect=True, init=True)
        assert low.get_current_mode() == 1  # LOW_LEVEL
        assert low.read_feedback().model_position.shape == (NUM_ACTIVE_JOINTS,)