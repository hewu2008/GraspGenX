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
    REPO_ROOT,
    WRIST_T_END_EFFECTOR,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
)
from curobo_planning.frames import invert_transform, matrix_to_wxyz
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

# SDK ready pose (mirrors replay_sdk_highlevel._READY_XYZ / _READY_QUAT): the
# wrist pose, relative to the arm motor-zero frame, that HIGH_LEVEL setArm_high
# targets in ``move_arm_to_ready_pose``.
_READY_XYZ = np.array([-0.1, 0.0, 0.30], dtype=np.float64)
_READY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # identity


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


def _return_to_initial_pose(low, initial_17, *, duration: float = 2.0) -> None:
    """Restore waist Z/pitch + both arms to the initial observation posture.

    LOW_LEVEL analog of ``replay_sdk_highlevel._return_to_initial_pose`` (no
    chassis motion): the waist is driven with ``prepare_robot_posture`` and both
    arms are ramped in joint space back to their ``initial_17`` snapshot values
    via :func:`retract_to_ready`, so the whole 17-axis vector returns to the
    initial observation posture.
    """
    prepare_robot_posture(low, 0.0, 0.0, _WAIST_NORMAL_Z, _WAIST_PITCH)
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    arm_cols = tuple(
        ZERITH_ACTIVE_JOINTS.index(name)
        for name in (ZERITH_ARM_JOINTS["left"] + ZERITH_ARM_JOINTS["right"])
    )
    retract_to_ready(
        low,
        current,
        np.asarray(initial_17, dtype=np.float64),
        arm_cols,
        duration=duration,
    )


def _ready_pose_base(robot, arm: str) -> np.ndarray | None:
    """Map the SDK ready pose into the planning tool frame in the base frame.

    ``B_T_E = inv(S_T_B) @ S_T_ready``: ``S_T_ready`` is the SDK ready pose —
    the end-effector pose in the arm motor-zero frame that ``setArm_high``
    targets and ``getHandRelative`` reports (verified: the SDK arm-end frame
    already includes the wrist->EEF offset, so no ``U_T_E`` is applied).  The
    ``S_T_B`` (pinocchio_ik.ik) is measured live from the robot's current arm
    pose and maps the SDK frame into the URDF ``body_yaw_link`` frame.  Returns
    None when ``S_T_B`` cannot be computed (robot must be connected/stationary).

    Per ``end2end_pipeline.robot_motion.move_arm_to_ready_pose`` the two arms
    do NOT share a single ready target: the left arm goes to ``_READY_XYZ``
    while the right arm mirrors the Y component
    ``[x, -y, z]`` (the arms are physically mirrored about the body XZ plane,
    so their SDK-motor-zero ready poses are opposite in Y).
    """
    from pinocchio_ik.ik import get_s_t_b

    s_t_b = get_s_t_b(robot, arm)
    if s_t_b is None:
        logger.warning(
            f"[Ready] No S_T_B calibration for '{arm}'; run "
            "verify_fk_against_sdk() first to move the arm to the ready pose."
        )
        return None
    y_sign = 1.0 if str(arm).lower().startswith("l") else -1.0
    s_ready = np.eye(4, dtype=np.float64)
    s_ready[:3, 3] = [
        _READY_XYZ[0], _READY_XYZ[1] * y_sign, _READY_XYZ[2]
    ]
    s_ready[:3, :3] = R.from_quat(_READY_QUAT).as_matrix()
    return invert_transform(s_t_b) @ s_ready


def _solve_arm_ik(arm: str, start_17, b_t_e_target: np.ndarray) -> np.ndarray | None:
    """cuRobo IK: solve the 7 target-arm joint angles for ``b_t_e_target``.

    Mirrors the getting_started ``inverse_kinematics`` example: builds the
    single-arm planning config with every other joint locked at ``start_17``
    and solves for the arm's tool frame pose.  ``b_t_e_target`` is in the URDF
    body_yaw_link frame (as produced by :func:`_ready_pose_base`); it is
    converted into the planning model's world frame (``dipan_link``) first.
    Returns the 7 joint angles in model order (``ZERITH_ARM_JOINTS[arm]``),
    or None on failure.
    """
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, Pose
    from curobo_planning.model import build_single_arm_planning_config

    full_by_name = dict(
        zip(ZERITH_ACTIVE_JOINTS, np.asarray(start_17, dtype=np.float64).tolist())
    )
    robot_cfg = build_single_arm_planning_config(arm, full_by_name)
    ik = InverseKinematics(
        InverseKinematicsCfg.create(robot=robot_cfg, num_seeds=64)
    )
    try:
        target_link = ik.tool_frames[0]
        # body_yaw_link-frame target -> planning model world frame (dipan_link).
        d_t_e = body_yaw_to_dipan(start_17) @ np.asarray(b_t_e_target, dtype=np.float64)
        goal_pose = Pose(
            position=torch.tensor(
                d_t_e[:3, 3][None, :], device="cuda", dtype=torch.float32
            ),
            quaternion=torch.tensor(
                matrix_to_wxyz(d_t_e[:3, :3])[None, :],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        # Seed the LM/optimizer with the CURRENT arm config: the ready pose is
        # close to where the arm already is, and solving from the zero config
        # (default) tends to get pushed into joint limits and fail even for
        # reachable targets.  ``seed_config`` is (batch, n, dof); n < num_seeds
        # are completed with random seeds.  (``current_state`` is NOT used: it
        # would enable the velocity-aware IK cost and crash on the (1, dof)-vs-
        # (dof,) shape mismatch.)
        name_to_start = full_by_name
        cur_7 = np.asarray(
            [name_to_start[name] for name in ik.joint_names], dtype=np.float64
        )
        seed_config = torch.tensor(
            cur_7[None, None, :], device="cuda", dtype=torch.float32
        )
        result = ik.solve_pose(
            GoalToolPose.from_poses({target_link: goal_pose}, num_goalset=1),
            seed_config=seed_config,
        )
        if not bool(result.success.item()):
            logger.error(f"[IK] cuRobo IK failed for {arm} ready pose.")
            return None
        js = np.asarray(result.js_solution.position[0, 0].cpu(), dtype=np.float64)
        name_to_angle = dict(zip(result.js_solution.joint_names, js))
        return np.asarray(
            [name_to_angle[name] for name in ZERITH_ARM_JOINTS[arm]],
            dtype=np.float64,
        )
    finally:
        ik.reset_seed()


def move_arms_to_ready_pose(low) -> None:
    """Move both arms to the SDK ready pose (cuRobo IK + joint-space ramp).

    Each arm's ready pose is converted into the base frame via ``S_T_B``,
    solved with cuRobo IK, and ramped in joint space with
    :func:`retract_to_ready`.  Arms without a calibrated ``S_T_B`` (or with an
    unreachable ready pose) are skipped with a warning.
    """
    arm_cols = {
        a: np.asarray(
            [ZERITH_ACTIVE_JOINTS.index(n) for n in ZERITH_ARM_JOINTS[a]],
            dtype=np.int64,
        )
        for a in ("left", "right")
    }

    def _log_eef(label: str) -> None:
        for a in ("left", "right"):
            s_t_e = low.read_sdk_arm_eef(a)
            if s_t_e is None:
                logger.info(f"[Ready] {label} | {a}: (SDK pose unavailable)")
                continue
            logger.info(
                f"[Ready] {label} | {a}: pos={np.round(s_t_e[:3, 3], 4).tolist()} "
                f"rpy_deg={np.round(np.degrees(R.from_matrix(s_t_e[:3, :3]).as_euler('xyz')), 2).tolist()}"
            )

    for arm in ("left", "right"):
        import pdb; pdb.set_trace()
        _log_eef("before move")
        b_t_e = _ready_pose_base(low._robot, arm)
        if b_t_e is None:
            continue
        current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
        # Use the real robot's verified ready-pose joint feedback from the
        # asset CSVs (left/right each recorded at the 09:39 run) instead of the
        # cuRobo IK solution.  cuRobo IK for the SDK ready pose fails because
        # the URDF model deviates from the real robot once the arm is bent
        # (measured 34.7 mm / 9 deg at the ready pose), so the model's only
        # reachable branch is an over-folded config that does not match the
        # physical ready pose.  The CSV values ARE the physical ready config.
        import csv

        side = "left" if arm == "left" else "right"
        csv_path = REPO_ROOT / f"assets/zerith/csv/arm_move_{side}_20260907_093956.csv"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        last = rows[-1]
        target_7 = np.asarray(
            [float(last[f"joint_{i}_pos"]) for i in range(1, 8)],
            dtype=np.float64,
        )
        if target_7 is None:
            continue
        cols = arm_cols[arm]
        target_17 = current.copy()
        target_17[cols] = target_7
        logger.info(f"[Ready] {arm} ready joints: {target_7.tolist()}")
        retract_to_ready(low, current, target_17, cols, duration=10.0)
        _log_eef("after move")


# URDF origins of the base-chain joints (assets/zerith/curobo/zerith_planning.urdf),
# used to convert body_yaw_link-frame targets into the planning model's world
# frame (base_link = dipan_link).  Distinct from _O_BP/_O_BY above, which are the
# wrist-camera hand-eye offsets used only by build_world_T_base.
_O_BP_URDF = np.array([0.1518, 0.0, 0.1275])
_O_BY_URDF = np.array([2.7149e-05, -0.00012105, 0.1572])


def body_yaw_to_dipan(start_17) -> np.ndarray:
    """Return ``D_T_B``: dipan_link -> body_yaw_link at the locked base joints.

    The planning model roots at ``dipan_link`` (zerith.yml ``base_link``) with
    daogui/body_pitch/body_yaw locked at ``start_17``, so an end-effector pose
    expressed in the body_yaw_link frame must be pre-multiplied by this fixed
    transform before feeding it to the model (``D_T_E = D_T_B @ B_T_E``).
    """
    by_name = dict(
        zip(ZERITH_ACTIVE_JOINTS, np.asarray(start_17, dtype=np.float64).tolist())
    )
    q_lift = float(by_name["daogui_joint"])
    q_bp = float(by_name["body_pitch_joint"])
    q_by = float(by_name["body_yaw_joint"])
    T = np.eye(4)
    T[:3, :3] = (
        R.from_euler("Y", q_bp).as_matrix() @ R.from_euler("Z", q_by).as_matrix()
    )
    T[:3, 3] = (
        np.array([0.0, 0.0, q_lift])
        + _O_BP_URDF
        + R.from_euler("Y", q_bp).as_matrix() @ _O_BY_URDF
    )
    return T


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
    from replay.replay_common import collect_grasp_plan

    low = create_low_level_robot(fake=fake)
    low.ensure_connected_low_level(connect=True, init=True)
    try:
        # Move the waist to the initial observation posture before snapshotting,
        # so initial_snapshot / world_T_base / retract targets all use it.
        prepare_robot_posture(low, 0.0, 0.0, _WAIST_NORMAL_Z, _WAIST_PITCH)
        # Calibrate S_T_B (SDK motor-zero frame <-> URDF body_yaw_link) from the
        # current static arm pose, so the ready-pose IK below can map the SDK
        # ready target into the base frame.  Arms stay static here (waist-only
        # move), which is what verify_fk_against_sdk's sampled reads require.
        if not fake:
            from pinocchio_ik.ik import verify_fk_against_sdk
            verify_fk_against_sdk(
                low._robot, side=None, samples=3, settle_s=0.3
            )
        # Then move both arms to the SDK ready pose (cuRobo IK + joint-space
        # ramp), so the initial snapshot records the full ready posture.
        move_arms_to_ready_pose(low)
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

        # for r in range(max(1, int(rounds))):
        #     logger.info(f"[Replay] ======== round {r + 1}/{max(1, int(rounds))} ========")
        #     # Return waist + both arms to the initial observation posture.
        #     _return_to_initial_pose(low, initial_snapshot)
        #     for gripper, label, _gidx, grasp4x4_world in plan:
        #         if gripper not in _GRIPPER_TO_ARM:
        #             logger.warning(
        #                 f"[Replay] Unknown gripper '{gripper}'; skipping."
        #             )
        #             continue
        #         arm = _GRIPPER_TO_ARM[gripper]
        #         cols = tuple(
        #             ZERITH_ACTIVE_JOINTS.index(name) for name in ZERITH_ARM_JOINTS[arm]
        #         )
        #         grasp_cycle(
        #             low,
        #             arm,
        #             grasp4x4_world,
        #             label,
        #             world_T_base=world_T_base,
        #             grasp_T_wrist=grasp_T_wrist,
        #             initial_snapshot=initial_snapshot,
        #             cols=cols,
        #         )
        logger.info("[Replay] All rounds complete.")
        return 0
    finally:
        low.close()