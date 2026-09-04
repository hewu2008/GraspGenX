"""Unit tests for the public ``curobo_planning.api.CuroboPlanning`` surface.

The GPU-backed ``CuroboGraspPlanner`` is replaced with a stub so no CUDA device
is required.  These cover the lazy lifecycle, argument normalization, frame and
artifact delegation, and the context manager.

Run with the ``zerith_graspgen`` environment (needed for the top-level
cuRobo/torch import pulled in by ``planner``):

    python -m pytest tests/test_curobo_planning_api.py -q
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_planning import frames
from curobo_planning.api import (
    CuroboPlannerConfig,
    CuroboPlanning,
    GraspCandidates,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_CUROBO_YAML,
)


class _FakePlanner:
    """Stand-in for ``CuroboGraspPlanner`` (avoids CUDA)."""

    def __init__(self, arm, full_start_position, config, *, locked_joint_position=None):
        self.arm = arm
        self.full_start_position = full_start_position
        self.config = config
        self.locked_joint_position = locked_joint_position
        self.plan_calls = []
        self.world_updates = []
        self.destroyed = False

    def plan_grasp(self, candidates, **kwargs):
        self.plan_calls.append((candidates, kwargs))
        return "motion"

    def update_world(self, scene):
        self.world_updates.append(scene)

    def destroy(self):
        self.destroyed = True


def _full_joint_position(offset: float = 0.0) -> dict[str, float]:
    return {name: float(offset) for name in ZERITH_ACTIVE_JOINTS}


def _transform(position=None, euler_xyz=None):
    transform = np.eye(4, dtype=np.float64)
    if euler_xyz is not None:
        transform[:3, :3] = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    if position is not None:
        transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _valid_candidates() -> GraspCandidates:
    pose = _transform([0.3, 0.2, 0.6], [0.1, -0.2, 0.4])
    return GraspCandidates(
        poses_world=pose[None, ...],
        confidence=np.array([0.9]),
        tags=np.array(["obj_1"], dtype="U16"),
        source_path=Path("/tmp/candidates.npz").resolve(),
    )


# ---------------------------------------------------------------------------
# Joint-position normalization
# ---------------------------------------------------------------------------

def test_normalize_mapping_uses_model_order():
    api = CuroboPlanning("left", _full_joint_position(0.25))
    expected = np.asarray([0.25] * len(ZERITH_ACTIVE_JOINTS), dtype=np.float64)
    np.testing.assert_allclose(api.full_start_position, expected)


def test_normalize_reorders_mapping_to_public_joint_order():
    mapping = _full_joint_position(0.0)
    mapping["right_elbow_joint"] = 1.5
    api = CuroboPlanning("left", mapping)
    index = ZERITH_ACTIVE_JOINTS.index("right_elbow_joint")
    assert api.full_start_position[index] == 1.5


def test_normalize_missing_joint_raises():
    mapping = _full_joint_position(0.0)
    mapping.pop("body_yaw_joint")
    try:
        CuroboPlanning("left", mapping)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing joint")


def test_normalize_vector_passthrough():
    vector = np.arange(len(ZERITH_ACTIVE_JOINTS), dtype=np.float64)
    api = CuroboPlanning("left", vector)
    np.testing.assert_allclose(api.full_start_position, vector)


def test_normalize_wrong_vector_shape_raises():
    try:
        CuroboPlanning("left", np.zeros(5))
    except ValueError as exc:
        assert "all" in str(exc) and "joints" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong vector shape")


# ---------------------------------------------------------------------------
# Lazy GPU planner lifecycle
# ---------------------------------------------------------------------------

def test_planner_created_only_on_first_access(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    assert api.planner_created is False
    planner = api.planner
    assert isinstance(planner, _FakePlanner)
    assert api.planner_created is True


def test_locked_joint_position_forwarded(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    locked = {"left_jaw_left_finger_joint": 0.0, "left_jaw_right_finger_joint": 0.0}
    api = CuroboPlanning(
        "left",
        _full_joint_position(),
        locked_joint_position=locked,
    )
    assert api.planner.locked_joint_position == locked


def test_close_destroys_planner(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    planner = api.planner
    api.close()
    assert planner.destroyed is True
    assert api.planner_created is False


def test_context_manager_closes(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    with CuroboPlanning("left", _full_joint_position()) as api:
        planner = api.planner
        assert api.planner_created is True
    assert planner.destroyed is True


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

def test_plan_delegates_to_planner(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    candidates = _valid_candidates()
    motion = api.plan(
        candidates,
        world_T_base=_transform([0.0, 0.0, 0.0]),
        grasp_T_wrist=_transform(),
        scene_digest="digest",
        object_label="obj_1",
    )
    assert motion == "motion"
    called_candidates, kwargs = api.planner.plan_calls[0]
    assert called_candidates is candidates
    assert kwargs["scene_digest"] == "digest"
    assert kwargs["object_label"] == "obj_1"


def test_update_world_is_noop_before_creation(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    api.update_world("scene")
    assert api.planner_created is False


def test_update_world_forwards_after_creation(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    planner = api.planner
    api.update_world("scene")
    assert planner.world_updates == ["scene"]


def test_default_config_is_used(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    api = CuroboPlanning("left", _full_joint_position())
    planner = api.planner
    assert isinstance(planner.config, CuroboPlannerConfig)


def test_grasp_to_tool_base_matches_frames_delegation():
    world_T_grasp = _transform([0.3, -0.2, 0.5], [0.2, 0.1, -0.3])
    world_T_base = _transform([0.01, 0.02, -0.03], [0.0, 0.0, 0.1])
    grasp_T_wrist = _transform([0.0, 0.0, 0.05], [0.4, 0.0, 0.0])
    api = CuroboPlanning("left", _full_joint_position())
    expected = frames.grasp_world_to_tool_base(
        world_T_grasp, world_T_base, grasp_T_wrist
    )
    np.testing.assert_allclose(
        api.grasp_to_tool_base(world_T_grasp, world_T_base, grasp_T_wrist),
        expected,
        atol=1e-9,
    )


def test_save_artifacts_delegates(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    captured = {}

    def fake_save(motion, output_root, *, robot_yaml_path=None):
        captured["motion"] = motion
        captured["output_root"] = output_root
        captured["robot_yaml_path"] = robot_yaml_path
        return Path("/tmp/out")

    monkeypatch.setattr("curobo_planning.api.save_plan_artifacts", fake_save)
    api = CuroboPlanning("left", _full_joint_position())
    result = api.save_artifacts("motion", Path("/tmp/out"))
    assert result == Path("/tmp/out")
    assert captured["motion"] == "motion"
    assert captured["output_root"] == Path("/tmp/out")
    assert captured["robot_yaml_path"] == ZERITH_CUROBO_YAML


def test_save_artifacts_uses_custom_yaml_path(monkeypatch):
    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _FakePlanner)
    captured = {}

    def fake_save(_motion, _output_root, *, robot_yaml_path=None):
        captured["robot_yaml_path"] = robot_yaml_path
        return Path("/tmp/out")

    monkeypatch.setattr("curobo_planning.api.save_plan_artifacts", fake_save)
    api = CuroboPlanning("left", _full_joint_position(), yaml_path=Path("/tmp/robot.yml"))
    api.save_artifacts("motion", Path("/tmp/out"))
    assert captured["robot_yaml_path"] == Path("/tmp/robot.yml")