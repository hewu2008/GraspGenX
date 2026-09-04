"""Constant mirrors: curobo_sdk.constants must equal curobo_planning.constants.

The test-only reverse dependency on ``curobo_planning`` is intentional -- the
runtime package must not import the planning package, so equality is asserted
here instead.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_planning import constants as planning_constants
from curobo_sdk import constants as sdk_constants


def test_active_joints_match_planning():
    assert sdk_constants.ZERITH_ACTIVE_JOINTS == (
        planning_constants.ZERITH_ACTIVE_JOINTS
    )
    assert sdk_constants.NUM_ACTIVE_JOINTS == len(
        planning_constants.ZERITH_ACTIVE_JOINTS
    )


def test_soft_position_limits_match_planning():
    assert dict(sdk_constants.ZERITH_SOFTWARE_POSITION_LIMITS) == dict(
        planning_constants.ZERITH_SOFTWARE_POSITION_LIMITS
    )
    # Every active joint must have a finite lower/upper limit.
    for name in sdk_constants.ZERITH_ACTIVE_JOINTS:
        low, high = sdk_constants.ZERITH_SOFTWARE_POSITION_LIMITS[name]
        assert low < high
        assert low == low and high == high  # finite


def test_joint_to_motor_mapping_is_bijective():
    mapping = sdk_constants.JOINT_TO_MOTOR_NAME
    assert set(mapping) == set(sdk_constants.ZERITH_ACTIVE_JOINTS)
    assert len(mapping) == len(set(mapping.values()))


def test_motor_id_whitelist_and_forbidden():
    assert sdk_constants.WAIST_MOTOR_IDS == (2, 3, 4)
    assert sdk_constants.LEFT_ARM_MOTOR_IDS == tuple(range(7, 14))
    assert sdk_constants.RIGHT_ARM_MOTOR_IDS == tuple(range(15, 22))
    assert sdk_constants.EXPECTED_ACTIVE_MOTOR_IDS == frozenset(
        (*sdk_constants.WAIST_MOTOR_IDS,
         *sdk_constants.LEFT_ARM_MOTOR_IDS,
         *sdk_constants.RIGHT_ARM_MOTOR_IDS)
    )
    assert sdk_constants.FORBIDDEN_MOTOR_IDS == frozenset((0, 1))
    assert not (sdk_constants.EXPECTED_ACTIVE_MOTOR_IDS
                & sdk_constants.FORBIDDEN_MOTOR_IDS)


def test_gripper_constants():
    assert sdk_constants.GRIPPER_MOTOR_ID == {"left": 14, "right": 22}
    assert sdk_constants.GRIPPER_MOTOR_NAME == {
        "left": "MOTOR_LEFT_ARM_8",
        "right": "MOTOR_RIGHT_ARM_8",
    }
    assert sdk_constants.GRIPPER_OPEN_POSITION < sdk_constants.GRIPPER_CLOSED_POSITION


def test_mit_sentinels():
    assert sdk_constants.DEFAULT_KP == -1.0
    assert sdk_constants.DEFAULT_KD == -1.0
    assert sdk_constants.FEEDBACK_LIMIT_TOLERANCE > 0.0