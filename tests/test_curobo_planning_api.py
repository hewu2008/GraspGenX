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
from curobo_planning.trajectory import PlannedMotion, TrajectorySegment


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


# ---------------------------------------------------------------------------
# Print the planned result (illustrative: run with pytest -s to see output)
# ---------------------------------------------------------------------------

def test_plan_prints_result(monkeypatch):
    """Return a populated ``PlannedMotion`` from the stub and print it.

    Use ``python -m pytest tests/test_curobo_planning_api.py::test_plan_prints_result -s``
    to see the printed planning result.
    """

    joint_names = tuple(f"j{i}" for i in range(7))
    positions = np.zeros((10, 7))
    object_label = "soda_can"
    scene_digest = "scene-20260904-001"

    class _ReturningPlanner(_FakePlanner):
        def plan_grasp(self, candidates, **kwargs):
            motion = PlannedMotion(
                plan_id="abc123def456",
                arm="left",
                object_label=kwargs.get("object_label", object_label),
                goalset_index=0,
                source_candidate_index=3,
                candidate_confidence=0.87,
                approach=TrajectorySegment(
                    name="approach", joint_names=joint_names, position=positions + 0.2,
                    velocity=None, acceleration=None, jerk=None, dt_s=0.02,
                ),
                grasp=TrajectorySegment(
                    name="grasp", joint_names=joint_names, position=positions + 0.4,
                    velocity=None, acceleration=None, jerk=None, dt_s=0.02,
                ),
                status="success",
                planning_time_s=1.2345,
                scene_digest=kwargs.get("scene_digest", scene_digest),
                selected_tool_pose_base=_transform([0.3, 0.2, 0.6], [0.1, -0.2, 0.4]),
                curobo_version="0.0.1",
                curobo_commit=None,
                metadata={"input_candidate_count": 8},
            )
            self.plan_calls.append((candidates, kwargs))
            return motion

    monkeypatch.setattr("curobo_planning.api.CuroboGraspPlanner", _ReturningPlanner)
    api = CuroboPlanning("left", _full_joint_position())
    motion = api.plan(
        _valid_candidates(),
        world_T_base=_transform([0.0, 0.0, 0.0]),
        grasp_T_wrist=_transform(),
        object_label=object_label,
        scene_digest=scene_digest,
    )

    print("\n=== 规划结果 (PlannedMotion) ===")
    print(f"plan_id               : {motion.plan_id}")
    print(f"arm / object          : {motion.arm} / {motion.object_label}")
    print(f"status                : {motion.status}")
    print(f"planning_time_s       : {motion.planning_time_s:.3f}")
    print(f"source_candidate_index: {motion.source_candidate_index} "
          f"(conf={motion.candidate_confidence})")
    print(f"scene_digest          : {motion.scene_digest}")
    print(f"curobo_version        : {motion.curobo_version}")
    print("--- 轨迹段 ---")
    for segment_name, segment in (("approach", motion.approach), ("grasp", motion.grasp)):
        print(f"[{segment_name}] waypoints={segment.waypoint_count}, dt={segment.dt_s}, "
              f"joints={segment.joint_names}")
        print(f"  position shape        : {segment.position.shape}")
        print(f"  position[0]           : {segment.position[0]}")
        print(f"  position[-1] (末端)    : {segment.position[-1]}")
    print("--- 选中候选的工具位姿 (base 系) ---")
    print("selected_tool_pose_base:")
    print(motion.selected_tool_pose_base)

    assert motion.status == "success"
    assert motion.approach.joint_names == joint_names