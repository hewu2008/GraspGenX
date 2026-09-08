"""pinocchio_lowlevel grasp replay: pinocchio IK + LOW_LEVEL SDK execution.

A cuRobo-free variant of ``replay_curobo_lowlevel``: instead of solving each
goal with the cuRobo optimizer, the target-arm EEF goal (in the URDF
``body_yaw_link`` frame) is solved with the single-arm pinocchio IK model from
``pinocchio_ik``, then the resulting 7 joint angles are ramped to in joint
space and executed through the LOW_LEVEL SDK driver.

Scope: this module first implements the two-arm *inverse kinematics* step
(``_solve_arm_ik``) on top of the pinocchio reduced model, plus a full grasp
cycle (approach -> close -> lift -> move to place -> release -> retract) that
reuses the coordinate / trajectory helpers from ``replay_curobo_lowlevel``.

All the self-contained coordinate math (``world_T_base``, ``grasp_T_wrist``)
and the LOW_LEVEL execution helpers are imported from ``replay_curobo_lowlevel``
to guarantee both backends share one source of truth.  Instead of depending on
``curobo_sdk``, the LOW_LEVEL driver is the thin ``pinocchio_ik.sdk`` wrapper
(``ZerithLowLevel``), so ``pinocchio_ik`` stays free of a ``curobo_sdk`` /
``end2end_pipeline`` dependency.
"""

from __future__ import annotations

import numpy as np
import time

from curobo_planning.constants import WRIST_T_END_EFFECTOR
from curobo_planning.logging_utils import get_logger

from pinocchio_ik.ik import _solve_arm_ik
from pinocchio_ik.sdk import ZerithLowLevel
from pinocchio_ik.sdk import ACTIVE_JOINTS, ARM_JOINTS, create_low_level_robot

from replay.replay_curobo_lowlevel import (
    _GRIPPER_TO_ARM,
    _WAIST_NORMAL_Z,
    _WAIST_PITCH,
    _lift_tool_pose,
    _place_tool_pose,
    build_grasp_T_wrist,
    build_world_T_base,
    read_imu_wxyz,
    retract_to_ready,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pinocchio IK (the core of this backend)
# ---------------------------------------------------------------------------
def _tool_pose_from_world_grasp(
    world_T_base: np.ndarray,
    world_grasp: np.ndarray,
    grasp_T_wrist: np.ndarray,
    wrist_T_eff: np.ndarray = WRIST_T_END_EFFECTOR,
) -> np.ndarray:
    """``B_T_E = inv(W_T_B) @ W_T_G @ G_T_U @ U_T_E``."""
    wb = np.asarray(world_T_base, dtype=np.float64)
    wg = np.asarray(world_grasp, dtype=np.float64)
    gu = np.asarray(grasp_T_wrist, dtype=np.float64)
    ue = np.asarray(wrist_T_eff, dtype=np.float64)

    inv_wb = np.eye(4)
    inv_wb[:3, :3] = wb[:3, :3].T
    inv_wb[:3, 3] = -wb[:3, :3].T @ wb[:3, 3]
    out = inv_wb @ wg @ gu @ ue

    logger.debug(
        "[IK] world grasp -> B_T_E: pos=%s",
        np.round(out[:3, 3], 4).tolist(),
    )
    return out


def _ramp_to_joints(driver, arm, target_7, duration: float = 3.0) -> None:
    """Ramp the target arm's 7 joints to ``target_7`` in joint space (LOW_LEVEL)."""
    current = np.asarray(driver.read_feedback().model_position, dtype=np.float64)
    cols = np.asarray(
        [ACTIVE_JOINTS.index(n) for n in ARM_JOINTS[arm]],
        dtype=np.int64,
    )
    target_17 = current.copy()
    target_17[cols] = np.asarray(target_7, dtype=np.float64)
    retract_to_ready(driver, current, target_17, cols, duration=duration)


# ---------------------------------------------------------------------------
# Helper: FK achieved EEF pose from the recorded joint feedback
# ---------------------------------------------------------------------------
def _fk_tool_pose(arm, q_7):
    """Return ``B_T_E`` (4x4) for the given 7 arm joint angles via pinocchio FK.

    Reuses ``ik_feasibility._fk_eef_pose`` (same reduced model), so the achieved
    pose matches the model that the IK solved against.
    """
    from pinocchio_ik.ik import _fk_eef_pose

    return _fk_eef_pose(arm, np.asarray(q_7, dtype=np.float64))


# ---------------------------------------------------------------------------
# Grasp cycle (pinocchio IK variant)
# ---------------------------------------------------------------------------
def _exec_moved_to_pose(driver, arm, b_t_e_target, duration=3.0) -> np.ndarray | None:
    """Solve + execute the arm to ``b_t_e_target``; return the achieved B_T_E."""
    q_7 = _solve_arm_ik(arm, b_t_e_target)
    if q_7 is None:
        return None
    _ramp_to_joints(driver, arm, q_7, duration=duration)
    current = np.asarray(driver.read_feedback().model_position, dtype=np.float64)
    cur_7 = np.asarray(
        [current[ACTIVE_JOINTS.index(n)] for n in ARM_JOINTS[arm]],
        dtype=np.float64,
    )
    return _fk_tool_pose(arm, cur_7)


def grasp_cycle(
    driver, arm, grasp4x4_world, label, *, world_T_base, grasp_T_wrist, initial_snapshot, cols
) -> None:
    """Full pinocchio-IK grasp cycle for one world grasp pose."""
    logger.info(
        f"[Cycle][{arm}] {label}: pos={np.asarray(grasp4x4_world)[:3, 3].tolist()}"
    )

    # 1. Approach: solve + move EEF to the grasp pose.
    B_T_E_grasp = _exec_moved_to_pose(
        driver, arm, _tool_pose_from_world_grasp(world_T_base, grasp4x4_world, grasp_T_wrist)
    )
    if B_T_E_grasp is None:
        logger.error(f"[Cycle][{arm}] approach IK failed; aborting cycle.")
        return
    logger.info("[Cycle] Grasp reached; closing gripper")
    driver.set_gripper_close(arm)
    time.sleep(2.0)

    # 2. Lift.
    lift_b_t_e = _lift_tool_pose(B_T_E_grasp)
    logger.info("[Cycle] Lifting grasped object")
    _exec_moved_to_pose(driver, arm, lift_b_t_e, duration=3.0)

    # 3. Move to place.
    place_b_t_e = _place_tool_pose(B_T_E_grasp, arm)
    logger.info("[Cycle] Moving to place")
    _exec_moved_to_pose(driver, arm, place_b_t_e, duration=3.0)

    # 4. Release.
    logger.info("[Cycle] Releasing gripper")
    driver.set_gripper_open(arm)
    time.sleep(2.0)

    # 5. Retract to the ready configuration.
    logger.info("[Cycle] Retracting to ready")
    _return_arm_to_initial(driver, np.asarray(initial_snapshot, dtype=np.float64), cols)


def _return_arm_to_initial(driver, initial_17, cols) -> None:
    """Ramp one arm back to its initial observed joints (joint space)."""
    current = np.asarray(driver.read_feedback().model_position, dtype=np.float64)
    retract_to_ready(driver, current, np.asarray(initial_17, dtype=np.float64), cols)


_WAIST_Z_JOINT = "daogui_joint"
_WAIST_PITCH_JOINT = "body_pitch_joint"


def _prepare_waist_posture(
    driver,
    tar_z: float,
    tar_pitch: float,
    *,
    duration: float = 3.0,
    rate: float = 500.0,
) -> None:
    """Drive the waist Z + pitch to the observation posture (LOW_LEVEL).

    Port of ``curobo_sdk.api.prepare_robot_posture`` onto the thin
    ``pinocchio_ik.sdk`` driver (which exposes no waist helper): ``daogui_joint``
    (Z) and ``body_pitch_joint`` (pitch) are interpolated from the current
    feedback to ``tar_z``/``tar_pitch``, Z first then pitch.  A low-level tick
    must command all 17 joints, so each tick composes the waist onto the base
    snapshot via :meth:`ZerithLowLevel.command_joints`.
    """
    base = np.asarray(driver.read_feedback().model_position, dtype=np.float64)
    wz = ACTIVE_JOINTS.index(_WAIST_Z_JOINT)
    wp = ACTIVE_JOINTS.index(_WAIST_PITCH_JOINT)
    cur_z = float(base[wz])
    cur_p = float(base[wp])
    steps = max(1, int(duration * rate))
    dt = 1.0 / rate

    def send(z: float, pitch: float) -> None:
        pos = base.copy()
        pos[wz] = z
        pos[wp] = pitch
        driver.command_joints(pos)

    for i in range(1, steps + 1):
        send(cur_z + (tar_z - cur_z) * i / steps, cur_p)
        time.sleep(dt)
    for i in range(1, steps + 1):
        send(tar_z, cur_p + (tar_pitch - cur_p) * i / steps)
        time.sleep(dt)
    time.sleep(1.5)


def _return_to_initial_pose(driver) -> None:
    """Restore waist Z/pitch + both arms to the ready observation posture.

    LOW_LEVEL analog of ``replay_sdk_highlevel._return_to_initial_pose`` (which
    also needs no snapshot): the waist is driven back with
    :func:`_prepare_waist_posture` and both arms are moved to the ready pose via
    pinocchio IK (:func:`_move_arm_to_ready`), so the full robot returns to the
    initial observation posture.
    """
    _prepare_waist_posture(driver, _WAIST_NORMAL_Z, _WAIST_PITCH)
    _move_arm_to_ready(driver, "left", ready_xyz=(-0.1, 0.0, 0.30), ready_quat=(0.0, 0.0, 0.0, 1.0))
    _move_arm_to_ready(driver, "right", ready_xyz=(-0.1, 0.0, 0.30), ready_quat=(0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Replay entry point
# ---------------------------------------------------------------------------
def _move_arm_to_ready(
    driver,
    arm,
    ready_xyz=(-0.1, 0.0, 0.30),
    ready_quat=(0.0, 0.0, 0.0, 1.0),
) -> None:
    """Move ``arm``'s EEF to the ready pose via pinocchio IK.

    The SDK ready pose is mapped into the URDF ``body_yaw_link`` frame via the
    live ``S_T_B`` calibration and solved with :func:`_solve_arm_ik`.  The arm's
    EEF goes to ``ready_xyz`` (the right arm mirrors the Y component
    ``[x, -y, z]``, as the arms are physically mirrored about the body XZ plane)
    with orientation ``ready_quat`` (wxyz).  An arm without a calibration (or an
    IK failure) is skipped with a warning.
    """
    from scipy.spatial.transform import Rotation as R
    from curobo_planning.frames import invert_transform
    from pinocchio_ik.ik import get_s_t_b

    b_t_e = get_s_t_b(driver._robot, arm)
    if b_t_e is None:
        logger.warning(
            f"[Ready] No S_T_B calibration for '{arm}'; skipping."
        )
        return
    s_ready = np.eye(4, dtype=np.float64)
    s_ready[:3, 3] = [ready_xyz[0], ready_xyz[1], ready_xyz[2]]
    s_ready[:3, :3] = R.from_quat(ready_quat).as_matrix()
    b_t_e = invert_transform(b_t_e) @ s_ready

    q_7 = _solve_arm_ik(arm, b_t_e)
    if q_7 is None:
        logger.error(f"[Ready] pinocchio IK failed for {arm}; skipping.")
        return
    logger.info(f"[Ready] pinocchio IK {arm}: {np.round(q_7, 4).tolist()}")
    _ramp_to_joints(driver, arm, q_7, duration=3.0)


def run_pinocchio_lowlevel_replay(
    scene_dir,
    grasps_dir=None,
    top_grasps=1,
    rounds=1,
) -> int:
    """Replay the grasp plan with pinocchio IK + LOW_LEVEL SDK execution."""
    from replay.replay_common import collect_grasp_plan

    driver: ZerithLowLevel = create_low_level_robot()
    driver.ensure_connected_low_level(connect=True, init=True)
    try:
        _return_to_initial_pose(driver)

        # initial_snapshot = np.asarray(
        #     driver.read_feedback().model_position, dtype=np.float64
        # )
        # world_T_base = build_world_T_base(read_imu_wxyz(driver), initial_snapshot)
        # grasp_T_wrist = build_grasp_T_wrist()

        # plan = collect_grasp_plan(scene_dir, grasps_dir=grasps_dir, top_grasps=top_grasps)
        # if not plan:
        #     logger.warning("[Replay] Empty grasp plan; nothing to execute.")
        #     return 0

        # arm_cols = {
        #     a: tuple(ACTIVE_JOINTS.index(n) for n in ARM_JOINTS[a])
        #     for a in ("left", "right")
        # }

        # for r in range(max(1, int(rounds))):
        #     logger.info(f"[Replay] ======== round {r + 1}/{max(1, int(rounds))} ========")
        #     for gripper, label, _gidx, grasp4x4_world in plan:
        #         if gripper not in _GRIPPER_TO_ARM:
        #             logger.warning(
        #                 f"[Replay] Unknown gripper '{gripper}'; skipping."
        #             )
        #             continue
        #         arm = _GRIPPER_TO_ARM[gripper]
        #         grasp_cycle(
        #             driver,
        #             arm,
        #             grasp4x4_world,
        #             label,
        #             world_T_base=world_T_base,
        #             grasp_T_wrist=grasp_T_wrist,
        #             initial_snapshot=initial_snapshot,
        #             cols=arm_cols[arm],
        #         )
        logger.info("[Replay] All rounds complete.")
        return 0
    finally:
        driver.close()