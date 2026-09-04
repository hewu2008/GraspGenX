"""curobo_lowlevel grasp replay: cuRobo plan + LOW_LEVEL SDK execution, full cycle.

Self-contained coordinate math for feeding world-frame GraspGenX poses into the
cuRobo planner (``curobo_planning``) and executing the resulting joint
trajectories through the LOW_LEVEL SDK driver (``curobo_sdk``):

  - ``world_T_base``  : base (``body_yaw_link``) -> world, built from the raw
                        IMU quat + lift/body_pitch/body_yaw off the joint
                        snapshot (replicates the wrist-camera FK inline instead
                        of importing ``end2end_pipeline.camera_pose``).
  - ``grasp_T_wrist`` : GraspGenX grasp base -> ZR wrist-pitch base relabeling
                        (replicates ``grasp_executor``'s ``T_grasp_to_wrist``).

Execution model: for EVERY phase the planner is created fresh from the CURRENT
joint feedback snapshot, because ``CuroboGraspPlanner`` plans its trajectory
from the start state captured at construction.  Reusing one planner would plan
later phases from the *initial* arm pose, not the arm's actual pose after the
previous phase.

The full cycle per grasp is: approach to the grasp pose -> close gripper ->
lift -> move to place -> release -> retract to the ready/initial joint
configuration, repeated over rounds.

Only the 7 joints of the replaying arm move; the lift/daogui/waist/opposite-arm
joints are held rigid at the snapshot (the cuRobo model locks them).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from curobo_planning.config import GraspCandidates
from curobo_planning.constants import (
    WRIST_T_END_EFFECTOR,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
)
from curobo_planning.frames import invert_transform
from curobo_planning.logging_utils import get_logger
from curobo_planning.trajectory import TrajectorySegment

logger = get_logger(__name__)

# Map the gripper-name directories used by the grasp plan to planner arm ids.
_GRIPPER_TO_ARM = {
    "zerith_left_gripper": "left",
    "zerith_right_gripper": "right",
}

_RATE_HZ: float = 500.0
_DT: float = 1.0 / _RATE_HZ

# URDF/empirical offsets for the wrist-camera FK (see end2end_pipeline.camera_pose).
_O_BP = np.array([0.1478, 0.0, 0.1275])
_O_BY = np.array([2.71e-5, -1.21e-4, 0.1572])

# Initial observation waist posture (mirrors end2end_pipeline.config).
_WAIST_NORMAL_Z: float = 0.67
_WAIST_PITCH: float = 1.2


# ---------------------------------------------------------------------------
# Coordinate math (self-contained, replicates the wrist-camera FK + grasps pivot)
# ---------------------------------------------------------------------------
def build_world_T_base(imu_quat_wxyz, model_position_17) -> np.ndarray:
    """Return ``base -> world`` (4x4) from raw IMU quat + a 17-joint snapshot.

    ``base`` is the URDF ``body_yaw_link``.  Uses the wrist-camera convention
    for body pitch (``R_y(q_bp)``): the wrist-anchored grasp .npz files live in
    the world frame defined by ``compute_hand_camera_pose``, so the FK must
    match that chain exactly.  (The head-camera chain uses ``R_x(-q_bp)``
    instead; do not mix the two conventions.)
    """
    position = np.asarray(model_position_17, dtype=np.float64)
    if position.shape != (17,):
        raise ValueError("model_position_17 must have shape (17,)")
    w, x, y, z = imu_quat_wxyz  # SDK quat is [w, x, y, z]
    R_chassis = R.from_quat([float(x), float(y), float(z), float(w)]).as_matrix()

    q_lift = float(position[ZERITH_ACTIVE_JOINTS.index("daogui_joint")])
    q_bp = float(position[ZERITH_ACTIVE_JOINTS.index("body_pitch_joint")])
    q_by = float(position[ZERITH_ACTIVE_JOINTS.index("body_yaw_joint")])

    R_body_pitch = R.from_euler("Y", q_bp, degrees=False).as_matrix()
    R_body_yaw = R.from_euler("Z", q_by, degrees=False).as_matrix()
    R_body = R_chassis @ R_body_pitch @ R_body_yaw

    t_body_yaw = np.array([0.0, 0.0, q_lift]) + R_chassis @ (
        _O_BP + R_body_pitch @ R_body_yaw @ _O_BY
    )

    T = np.eye(4)
    T[:3, :3] = R_body
    T[:3, 3] = t_body_yaw
    return T


def build_grasp_T_wrist() -> np.ndarray:
    """Return ``G_T_U``: GraspGenX grasp base -> ZR wrist-pitch base (rotation only).

    Replicates the axis relabeling in ``grasp_executor``: GraspGenX grasp
    (Z=approach/X=closing) -> ZR hand (X=approach/Y=closing).  The Planner adds
    the fixed wrist->end-effector offset via ``WRIST_T_END_EFFECTOR``, so the
    0.1435 m EEF offset must NOT be baked in here (avoiding double counting).
    """
    T = np.eye(4)
    T[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    return T


def tool_pose_to_world_grasp(
    world_T_base: np.ndarray,
    B_T_E: np.ndarray,
    grasp_T_wrist: np.ndarray,
    wrist_T_eff: np.ndarray = WRIST_T_END_EFFECTOR,
) -> np.ndarray:
    """Invert the planner forward chain to express an EEF-in-base target as a world grasp.

    ``B_T_E`` is an end-effector pose in the base frame (as produced by
    ``selected_tool_pose_base``).  Rewriting the planner's
    ``B_T_E = inv(W_T_B) @ W_T_G @ G_T_U @ U_T_E`` for ``W_T_G`` gives
    ``W_T_G = W_T_B @ B_T_E @ inv(U_T_E) @ inv(G_T_U)``, letting the same
    ``plan(candidates, world_T_base, grasp_T_wrist)`` target arbitrary EEF poses
    (e.g. lift/place).
    """
    return (
        np.asarray(world_T_base, dtype=np.float64)
        @ np.asarray(B_T_E, dtype=np.float64)
        @ invert_transform(wrist_T_eff)
        @ invert_transform(grasp_T_wrist)
    )


def _single_candidate(world_grasp, label: str) -> GraspCandidates:
    """Wrap a single world-frame grasp pose into a one-entry GraspCandidates."""
    return GraspCandidates(
        poses_world=np.asarray(world_grasp, dtype=np.float64)[None, ...],
        confidence=np.array([1.0]),
        tags=np.array([str(label)], dtype="U64"),
        source_path=Path("<replay-curobo_lowlevel>"),
    )


# ---------------------------------------------------------------------------
# Command construction / trajectory execution
# ---------------------------------------------------------------------------
def segment_to_17(snapshot_17, segment):
    """Spread a 7-joint arm trajectory segment over the 17-joint command vector.

    Returns ``(pos17, vel17)``, each ``(N, 17)``: the 7 target-arm columns copy
    the segment, the other 10 joints (lift/waist/opposite arm) keep the snapshot
    positions with zero velocity.  When ``segment.velocity`` is None the arm
    velocity is a forward difference.
    """
    snapshot = np.asarray(snapshot_17, dtype=np.float64)
    if snapshot.shape != (17,):
        raise ValueError("snapshot_17 must have shape (17,)")
    position = np.asarray(segment.position, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != len(segment.joint_names):
        raise ValueError(
            "segment.position must have shape (N, len(segment.joint_names))"
        )
    n = int(position.shape[0])
    name_to_idx = {name: i for i, name in enumerate(ZERITH_ACTIVE_JOINTS)}
    cols = [name_to_idx[name] for name in segment.joint_names]

    pos17 = np.broadcast_to(snapshot, (n, 17)).copy()
    pos17[:, cols] = position

    if segment.velocity is not None:
        velocity = np.asarray(segment.velocity, dtype=np.float64)
        if velocity.shape != position.shape:
            raise ValueError("segment.velocity must have shape (N, len(joint_names))")
    else:
        velocity = np.zeros_like(position)
        if n >= 2:
            velocity[1:] = np.diff(position, axis=0) / float(segment.dt_s)
            velocity[0] = velocity[1]

    vel17 = np.zeros((n, 17), dtype=np.float64)
    vel17[:, cols] = velocity
    return pos17, vel17


def execute_trajectory(low, segment, snapshot_17, *, hold_s: float = 1.0) -> None:
    """Play a joint trajectory at its lockstep ``dt_s``, then hold the final pose."""
    pos17, vel17 = segment_to_17(snapshot_17, segment)
    n = int(pos17.shape[0])
    start = time.perf_counter()
    for i in range(n):
        target_when = start + i * float(segment.dt_s)
        low.command_joints(pos17[i], vel17[i])
        delay = target_when - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
    # Hold the terminal pose briefly so the next phase reads a settled state.
    low.command_joints(pos17[-1], np.zeros(17))
    if hold_s > 0.0:
        time.sleep(hold_s)


def retract_to_ready(
    low, current_17, initial_17, cols, *, duration: float = 2.0
) -> None:
    """Ramp the target-arm joints back to their initial values (joint space)."""
    current = np.asarray(current_17, dtype=np.float64)
    initial = np.asarray(initial_17, dtype=np.float64)
    if current.shape != (17,) or initial.shape != (17,):
        raise ValueError("current_17/initial_17 must each have shape (17,)")
    # `cols` may arrive as a tuple; coerce to an int array so advanced indexing
    # selects joints instead of unpacking the tuple into separate dimensions.
    cols = np.asarray(list(cols), dtype=np.int64)
    steps = max(1, int(duration * _RATE_HZ))
    ratios = np.linspace(0.0, 1.0, steps + 1)[1:]
    grid = current[cols][None, :] + (initial[cols] - current[cols])[None, :] * ratios[
        :, None
    ]
    segment = TrajectorySegment(
        name="retract_to_ready",
        joint_names=tuple(ZERITH_ACTIVE_JOINTS[c] for c in cols),
        position=grid,
        velocity=None,
        acceleration=None,
        jerk=None,
        dt_s=1.0 / _RATE_HZ,
    )
    # Fill locked joints from ``initial`` (= current for non-arm slots).
    execute_trajectory(low, segment, initial, hold_s=0.5)


# ---------------------------------------------------------------------------
# Planner / driver helpers
# ---------------------------------------------------------------------------
def _new_planner(arm, start_17):
    from curobo_planning.api import CuroboPlanning

    return CuroboPlanning(arm, np.asarray(start_17, dtype=np.float64))


def _plan_to_pose(planning, world_T_base, grasp_T_wrist, world_grasp, label):
    return planning.plan(
        _single_candidate(world_grasp, label),
        world_T_base=world_T_base,
        grasp_T_wrist=grasp_T_wrist,
        object_label=label,
        scene_digest="curobo_lowlevel_replay",
    )


def _lift_tool_pose(B_T_E_grasp) -> np.ndarray:
    """Back off the approach axis by 0.10 m then rise 0.05 m (rotation unchanged)."""
    target = np.asarray(B_T_E_grasp, dtype=np.float64).copy()
    approach_dir = target[:3, :3][:, 0]
    target[:3, 3] = target[:3, 3] - 0.10 * approach_dir + np.array([0.0, 0.0, 0.05])
    return target


def _place_tool_pose(B_T_E_grasp, arm: str) -> np.ndarray:
    """Offset the EEF target by [0.17, 0.30*y_sign, 0] in the base frame."""
    y_sign = 1.0 if arm == "left" else -1.0
    target = np.asarray(B_T_E_grasp, dtype=np.float64).copy()
    target[:3, 3] = target[:3, 3] + np.array([0.17, 0.30 * y_sign, 0.0])
    return target


def _exec_planned_phase(
    low, arm, world_T_base, grasp_T_wrist, world_grasp, label
) -> np.ndarray:
    """Plan from the current feedback snapshot to ``world_grasp`` and execute.

    Returns the achieved ``B_T_E`` (tool pose in base) for chaining lift/place.
    """
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    planning = _new_planner(arm, current)
    try:
        motion = _plan_to_pose(planning, world_T_base, grasp_T_wrist, world_grasp, label)
        B_T_E = np.asarray(motion.selected_tool_pose_base, dtype=np.float64)
        execute_trajectory(low, motion.grasp, current)
        return B_T_E
    finally:
        planning.close()


def grasp_cycle(
    low, arm, grasp4x4_world, label, *, world_T_base, grasp_T_wrist, initial_snapshot, cols
) -> None:
    """Full low-level grasp cycle for one world grasp pose."""
    logger.info(
        f"[Cycle][{arm}] {label}: pos={np.asarray(grasp4x4_world)[:3, 3].tolist()}"
    )

    # 1. Approach: plan from initial snapshot directly to the grasp pose.
    B_T_E_grasp = _exec_planned_phase(
        low, arm, world_T_base, grasp_T_wrist, np.asarray(grasp4x4_world), label
    )
    logger.info("[Cycle] Grasp reached; closing gripper")
    low.set_gripper_close(arm)
    time.sleep(2.0)

    # 2. Lift: back off the approach axis and rise, rotation unchanged.
    lift_world_grasp = tool_pose_to_world_grasp(
        world_T_base, _lift_tool_pose(B_T_E_grasp), grasp_T_wrist
    )
    logger.info("[Cycle] Lifting grasped object")
    _exec_planned_phase(low, arm, world_T_base, grasp_T_wrist, lift_world_grasp, f"{label}:lift")

    # 3. Move to place.
    place_world_grasp = tool_pose_to_world_grasp(
        world_T_base, _place_tool_pose(B_T_E_grasp, arm), grasp_T_wrist
    )
    logger.info("[Cycle] Moving to place")
    _exec_planned_phase(low, arm, world_T_base, grasp_T_wrist, place_world_grasp, f"{label}:place")

    # 4. Release.
    logger.info("[Cycle] Releasing gripper")
    low.set_gripper_open(arm)
    time.sleep(2.0)

    # 5. Retract to the ready configuration.
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    logger.info("[Cycle] Retracting to ready")
    retract_to_ready(low, current, np.asarray(initial_snapshot, dtype=np.float64), cols)


def read_imu_wxyz(low) -> np.ndarray:
    """Return the IMU quaternion ``[w,x,y,z]``, or identity when unavailable (fake)."""
    try:
        ok, imu = low._robot.getIMU_State()
        if ok and imu is not None and hasattr(imu, "quat"):
            return np.asarray(imu.quat, dtype=np.float64)
    except Exception:
        pass
    return np.array([1.0, 0.0, 0.0, 0.0])


def run_curobo_lowlevel_replay(
    scene_dir,
    grasps_dir=None,
    top_grasps=1,
    rounds=1,
    *,
    fake=False,
) -> int:
    """Replay the grasp plan with cuRobo planning + LOW_LEVEL SDK execution."""
    from curobo_sdk.api import create_low_level_robot, prepare_robot_posture
    from replay.replay_sdk_highlevel import collect_grasp_plan

    low = create_low_level_robot(fake=fake)
    low.ensure_connected_low_level(connect=True, init=True)
    try:
        # Move the waist to the initial observation posture before snapshotting,
        # so initial_snapshot / world_T_base / retract targets all use it.
        prepare_robot_posture(low, 0.0, 0.0, _WAIST_NORMAL_Z, _WAIST_PITCH)
        imu_wxyz = read_imu_wxyz(low)
        initial_snapshot = np.asarray(
            low.read_feedback().model_position, dtype=np.float64
        )
        world_T_base = build_world_T_base(imu_wxyz, initial_snapshot)
        grasp_T_wrist = build_grasp_T_wrist()
        logger.info("[Replay] world_T_base translated by build_world_T_base")

        plan = collect_grasp_plan(scene_dir, grasps_dir=grasps_dir, top_grasps=top_grasps)
        if not plan:
            logger.warning("[Replay] Empty grasp plan; nothing to execute.")
            return 0

        for r in range(max(1, int(rounds))):
            logger.info(f"[Replay] ======== round {r + 1}/{max(1, int(rounds))} ========")
            for gripper, label, _gidx, grasp4x4_world in plan:
                if gripper not in _GRIPPER_TO_ARM:
                    logger.warning(
                        f"[Replay] Unknown gripper '{gripper}'; skipping."
                    )
                    continue
                arm = _GRIPPER_TO_ARM[gripper]
                cols = tuple(
                    ZERITH_ACTIVE_JOINTS.index(name) for name in ZERITH_ARM_JOINTS[arm]
                )
                grasp_cycle(
                    low,
                    arm,
                    grasp4x4_world,
                    label,
                    world_T_base=world_T_base,
                    grasp_T_wrist=grasp_T_wrist,
                    initial_snapshot=initial_snapshot,
                    cols=cols,
                )
        logger.info("[Replay] All rounds complete.")
        return 0
    finally:
        low.close()