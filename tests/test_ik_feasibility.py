"""Tests for scripts/end2end_pipeline/ik_feasibility.py.

These run the pinocchio-backed IK feasibility check against the real Zerith
URDF. The reduced-arm model used by the check is a separate concern from the
robot SDK, so no SDK stub is required here — only ``pinocchio``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="session")
def ik_mod():
    import importlib
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    mod = importlib.import_module("end2end_pipeline.ik_feasibility")
    yield mod


@pytest.mark.parametrize("side", ["left", "right"])
def test_reduced_model_has_seven_arm_dof(ik_mod, side):
    """The reduced chain is a fixed-base 7-DOF serial arm with one EEF frame."""
    model = ik_mod._load_model(side)
    assert model is not None
    m, _data, joint_names, frame_id, _aj, q_indices = model
    assert m.nq == 7
    assert len(joint_names) == 7
    # q indices should be 0..6 for a serial chain.
    np.testing.assert_array_equal(q_indices, np.arange(7))
    assert m.frames[frame_id].name.endswith("_end_effector_link")


@pytest.mark.parametrize("side", ["left", "right"])
def test_ready_pose_is_reachable(ik_mod, side):
    """The symmetric ready pose must be considered reachable on both arms."""
    reachable, detail = ik_mod.check_eef_reachable(
        side, [-0.1, 0.0, 0.30], [0.0, 0.0, 0.0, 1.0], seeds=8
    )
    assert reachable is True
    assert detail["residual"] <= 0.02
    assert "min_joint_margin_deg" in detail
    assert "q_arm" in detail and len(detail["q_arm"]) == 7


@pytest.mark.parametrize("side", ["left", "right"])
def test_far_pose_is_unreachable(ik_mod, side):
    """A target far outside the arm workspace must be rejected."""
    reachable, detail = ik_mod.check_eef_reachable(
        side, [2.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0], seeds=8
    )
    assert reachable is False
    assert detail["residual"] > ik_mod.DEFAULT_RESIDUAL_TOL


def test_high_pose_is_unreachable(ik_mod):
    """A target well above the shoulder must be rejected."""
    for side in ("left", "right"):
        reachable, detail = ik_mod.check_eef_reachable(
            side, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], seeds=8
        )
        assert reachable is False
        assert detail["residual"] > ik_mod.DEFAULT_RESIDUAL_TOL


def test_pose_behind_mount_is_unreachable(ik_mod):
    """A point far behind (and below) the arm mount is unreachable on both sides."""
    for side in ("left", "right"):
        reachable, detail = ik_mod.check_eef_reachable(
            side, [-1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], seeds=8
        )
        assert reachable is False
        assert detail["residual"] > ik_mod.DEFAULT_RESIDUAL_TOL