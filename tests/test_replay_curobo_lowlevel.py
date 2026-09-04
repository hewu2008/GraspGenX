"""Unit tests for the curobo_lowlevel replay module (no hardware/GPU).

Pure-math helpers and the LOW_LEVEL command/execution path are exercised with a
``FakeSDKRobot``; the cuRobo planner is replaced by a stub so no CUDA/GPU is
required.  Run in the ``zerith_graspgen`` env from the repo root.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from curobo_planning.constants import (
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    WRIST_T_END_EFFECTOR,
)
from curobo_planning.frames import grasp_world_to_tool_base
from curobo_planning.trajectory import PlannedMotion, TrajectorySegment
from replay.replay_curobo_lowlevel import (
    _GRIPPER_TO_ARM,
    build_grasp_T_wrist,
    build_world_T_base,
    execute_trajectory,
    grasp_cycle,
    retract_to_ready,
    segment_to_17,
    tool_pose_to_world_grasp,
)

LEFT_COLS = tuple(
    ZERITH_ACTIVE_JOINTS.index(name) for name in ZERITH_ARM_JOINTS["left"]
)


def _left_segment(position, dt=0.02):
    return TrajectorySegment(
        name="grasp",
        joint_names=tuple(ZERITH_ARM_JOINTS["left"]),
        position=np.asarray(position, dtype=np.float64),
        velocity=None,
        acceleration=None,
        jerk=None,
        dt_s=dt,
    )


# ---------------------------------------------------------------------------
# Coordinate math
# ---------------------------------------------------------------------------
def test_build_grasp_T_wrist_relabel():
    T = build_grasp_T_wrist()
    assert np.allclose(T[:3, :3], [[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_build_world_T_base_matches_wrist_fk():
    # Recompute the wrist-camera FK inline with the same constants.
    from scipy.spatial.transform import Rotation as R

    imu_wxyz = np.array([0.98, 0.02, 0.03, 0.19])
    q = np.asarray(imu_wxyz)
    w, x, y, z = q
    R_chassis = R.from_quat([x, y, z, w]).as_matrix()
    pos = np.linspace(-0.1, 0.5, 17)
    q_lift = pos[0]
    q_bp = pos[1]
    q_by = pos[2]
    R_body_pitch = R.from_euler("Y", q_bp, degrees=False).as_matrix()
    R_body_yaw = R.from_euler("Z", q_by, degrees=False).as_matrix()
    R_body = R_chassis @ R_body_pitch @ R_body_yaw
    _O_BP = np.array([0.1478, 0.0, 0.1275])
    _O_BY = np.array([2.71e-5, -1.21e-4, 0.1572])
    t_body_yaw = np.array([0.0, 0.0, q_lift]) + R_chassis @ (
        _O_BP + R_body_pitch @ R_body_yaw @ _O_BY
    )
    expected = np.eye(4)
    expected[:3, :3] = R_body
    expected[:3, 3] = t_body_yaw
    assert np.allclose(build_world_T_base(imu_wxyz, pos), expected, atol=1e-12)


def test_tool_pose_to_world_grasp_roundtrip():
    world_T_base = np.eye(4)
    world_T_base[:3, 3] = [0.2, -0.1, 0.4]
    world_T_base[:3, :3] = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float
    )
    grasp_T_wrist = build_grasp_T_wrist()
    B_T_E = np.eye(4)
    B_T_E[:3, 3] = [0.3, 0.1, 0.5]
    B_T_E[:3, :3] = np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float
    )
    world_grasp = tool_pose_to_world_grasp(world_T_base, B_T_E, grasp_T_wrist)
    recovered = grasp_world_to_tool_base(
        world_grasp, world_T_base, grasp_T_wrist, WRIST_T_END_EFFECTOR
    )
    assert np.allclose(recovered, B_T_E, atol=1e-9)


def test_gripper_to_arm_map():
    assert _GRIPPER_TO_ARM == {
        "zerith_left_gripper": "left",
        "zerith_right_gripper": "right",
    }


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def test_segment_to_17_mapping_and_forward_diff_velocity():
    snapshot = np.zeros(17)
    seg = _left_segment(np.linspace(0.0, 0.5, 5)[:, None].repeat(7, axis=1), dt=0.02)
    pos17, vel17 = segment_to_17(snapshot, seg)
    assert pos17.shape == (5, 17)
    assert vel17.shape == (5, 17)
    # Locked slots keep the snapshot position and zero velocity.
    locked = [i for i in range(17) if i not in LEFT_COLS]
    assert np.allclose(pos17[:, locked], 0.0)
    assert np.allclose(vel17[:, locked], 0.0)
    # Arm slots copy the trajectory; velocity is the forward difference.
    # linspace(0, 0.5, 5) advances 0.125/step, so vel = 0.125 / dt = 6.25.
    assert np.allclose(pos17[:, list(LEFT_COLS)], seg.position)
    assert np.allclose(vel17[1:, list(LEFT_COLS)], (0.5 / 4) / 0.02)
    assert np.allclose(vel17[0, list(LEFT_COLS)], vel17[1, list(LEFT_COLS)])


def test_segment_to_17_mapping_with_velocity():
    snapshot = np.full(17, 0.1)
    pos = np.zeros((3, 7))
    vel = np.ones((3, 7)) * 0.2
    seg = TrajectorySegment(
        name="grasp",
        joint_names=tuple(ZERITH_ARM_JOINTS["left"]),
        position=pos,
        velocity=vel,
        acceleration=None,
        jerk=None,
        dt_s=0.01,
    )
    _, vel17 = segment_to_17(snapshot, seg)
    assert np.allclose(vel17[:, list(LEFT_COLS)], 0.2)
    assert np.allclose(vel17[:, [i for i in range(17) if i not in LEFT_COLS]], 0.0)


# ---------------------------------------------------------------------------
# Execution on a fake robot
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_low():
    from curobo_sdk.api import create_low_level_robot

    low = create_low_level_robot(fake=True)
    low.ensure_connected_low_level(connect=True, init=True)
    return low


def test_execute_trajectory_commands_exactly_active_motors(fake_low):
    from curobo_sdk.constants import EXPECTED_ACTIVE_MOTOR_IDS
    from curobo_sdk.fake_robot import EtherCAT_Motor_Index

    seg = _left_segment(np.linspace(0.0, 0.2, 3)[:, None].repeat(7, axis=1), dt=0.02)
    snapshot = np.zeros(17)
    execute_trajectory(fake_low, seg, snapshot, hold_s=0.0)
    robot = fake_low._robot
    assert robot.commanded_ids == EXPECTED_ACTIVE_MOTOR_IDS
    assert not (robot.commanded_ids & {0, 1, 14, 22})
    # The arm motor reached the trajectory's final position.  command_joints
    # shares one control object across joints (recorded by reference), so assert
    # against the motor state, which _apply() snapshots at send time.
    arm_id = int(getattr(EtherCAT_Motor_Index, "MOTOR_LEFT_ARM_1"))
    _, state = robot.getMotorState(arm_id)
    assert np.isclose(state.Position_Actual, 0.2)


def test_segment_soft_limit_violation_raises(fake_low):
    # body_yaw_joint is locked at 0 here, but command a far-out arm target.
    seg = _left_segment(np.full((2, 7), 2.0), dt=0.02)
    with pytest.raises(Exception):
        execute_trajectory(fake_low, seg, np.zeros(17), hold_s=0.0)


def test_retract_to_ready_ramps_arm_back(fake_low):
    initial = np.zeros(17)
    current = np.zeros(17)
    current[list(LEFT_COLS)] = 0.3
    # Small duration to keep the test fast.
    retract_to_ready(fake_low, current, initial, list(LEFT_COLS), duration=0.05)
    robot = fake_low._robot
    arm_id = int(
        getattr(__import__("curobo_sdk.fake_robot", fromlist=["EtherCAT_Motor_Index"]),
                "EtherCAT_Motor_Index", None) and
        __import__("curobo_sdk.fake_robot", fromlist=["EtherCAT_Motor_Index"]).EtherCAT_Motor_Index.MOTOR_LEFT_ARM_1
    )
    last = robot.commands_for(arm_id)[-1]
    assert np.isclose(last.Position, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Full cycle with a stub planner (no GPU)
# ---------------------------------------------------------------------------
class _StubPlanner:
    """Returns a synthetic single-waypoint motion; tracks close()."""

    def __init__(self, arm, start_17):
        self.arm = arm
        self.start_17 = np.asarray(start_17, dtype=np.float64)
        self.closed = False
        joint_names = tuple(ZERITH_ARM_JOINTS[arm])
        cols = tuple(ZERITH_ACTIVE_JOINTS.index(n) for n in joint_names)
        pose = np.eye(4)
        # A trivial, in-limit tool pose in the base frame.
        self.B_T_E = pose
        self._pos = self.start_17[list(cols)][None, :].copy()
        self.motion = PlannedMotion(
            plan_id="stub",
            arm=arm,
            object_label="label",
            goalset_index=0,
            source_candidate_index=0,
            candidate_confidence=1.0,
            grasp=TrajectorySegment(
                name="grasp",
                joint_names=joint_names,
                position=self._pos,
                velocity=None,
                acceleration=None,
                jerk=None,
                dt_s=0.02,
            ),
            status="ok",
            planning_time_s=0.0,
            scene_digest="test",
            selected_tool_pose_base=self.B_T_E,
            curobo_version="",
            curobo_commit=None,
        )

    def plan(self, candidates, **kwargs):
        return self.motion

    def close(self):
        self.closed = True

    def destroy(self):
        self.closed = True


def test_grasp_cycle_phase_order_with_stub_planner(fake_low, monkeypatch):
    from replay import replay_curobo_lowlevel as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    planner_log = []

    def _fake_new_planner(arm, start_17):
        p = _StubPlanner(arm, start_17)
        planner_log.append(p)
        return p

    monkeypatch.setattr(mod, "_new_planner", _fake_new_planner)

    initial = np.zeros(17)
    world_T_base = np.eye(4)
    grasp_T_wrist = build_grasp_T_wrist()
    grasp4x4 = np.eye(4)
    grasp4x4[:3, 3] = [0.3, 0.1, 0.5]

    grasp_cycle(
        fake_low, "left", grasp4x4, "obj",
        world_T_base=world_T_base, grasp_T_wrist=grasp_T_wrist,
        initial_snapshot=initial, cols=tuple(LEFT_COLS),
    )

    # Planners were created fresh per phase (approach + lift + place) and closed.
    assert len(planner_log) == 3
    assert all(p.closed for p in planner_log)
    # Gripper was closed then opened.
    robot = fake_low._robot
    gripper_id = 14  # MOTOR_LEFT_ARM_8
    grip_cmds = robot.commands_for(gripper_id)
    assert grip_cmds, "gripper was never commanded"
    # set_gripper_close uses hard-coded close control with position 1.5.
    close_cmds = [c for c in grip_cmds if c.Position > 1.0]
    open_cmds = [c for c in grip_cmds if c.Position < 0.1]
    assert close_cmds and open_cmds


def test_run_replay_importable():
    from replay import replay_curobo_lowlevel as mod
    assert callable(mod.run_curobo_lowlevel_replay)