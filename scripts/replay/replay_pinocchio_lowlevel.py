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
to guarantee both backends share one source of truth.
"""

from __future__ import annotations

import numpy as np
import time

from curobo_planning.constants import (
    WRIST_T_END_EFFECTOR,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
)
from curobo_planning.logging_utils import get_logger

from pinocchio_ik.ik import _solve_arm_ik

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


def _ramp_to_joints(low, arm, target_7, duration: float = 3.0) -> None:
    """Ramp the target arm's 7 joints to ``target_7`` in joint space (LOW_LEVEL)."""
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    cols = np.asarray(
        [ZERITH_ACTIVE_JOINTS.index(n) for n in ZERITH_ARM_JOINTS[arm]],
        dtype=np.int64,
    )
    target_17 = current.copy()
    target_17[cols] = np.asarray(target_7, dtype=np.float64)
    retract_to_ready(low, current, target_17, cols, duration=duration)


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
def _exec_moved_to_pose(low, arm, b_t_e_target, duration=3.0) -> np.ndarray | None:
    """Solve + execute the arm to ``b_t_e_target``; return the achieved B_T_E."""
    q_7 = _solve_arm_ik(arm, b_t_e_target)
    if q_7 is None:
        return None
    _ramp_to_joints(low, arm, q_7, duration=duration)
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    cur_7 = np.asarray(
        [current[ZERITH_ACTIVE_JOINTS.index(n)] for n in ZERITH_ARM_JOINTS[arm]],
        dtype=np.float64,
    )
    return _fk_tool_pose(arm, cur_7)


def grasp_cycle(
    low, arm, grasp4x4_world, label, *, world_T_base, grasp_T_wrist, initial_snapshot, cols
) -> None:
    """Full pinocchio-IK grasp cycle for one world grasp pose."""
    logger.info(
        f"[Cycle][{arm}] {label}: pos={np.asarray(grasp4x4_world)[:3, 3].tolist()}"
    )

    # 1. Approach: solve + move EEF to the grasp pose.
    B_T_E_grasp = _exec_moved_to_pose(
        low, arm, _tool_pose_from_world_grasp(world_T_base, grasp4x4_world, grasp_T_wrist)
    )
    if B_T_E_grasp is None:
        logger.error(f"[Cycle][{arm}] approach IK failed; aborting cycle.")
        return
    logger.info("[Cycle] Grasp reached; closing gripper")
    low.set_gripper_close(arm)
    time.sleep(2.0)

    # 2. Lift.
    lift_b_t_e = _lift_tool_pose(B_T_E_grasp)
    logger.info("[Cycle] Lifting grasped object")
    _exec_moved_to_pose(low, arm, lift_b_t_e, duration=3.0)

    # 3. Move to place.
    place_b_t_e = _place_tool_pose(B_T_E_grasp, arm)
    logger.info("[Cycle] Moving to place")
    _exec_moved_to_pose(low, arm, place_b_t_e, duration=3.0)

    # 4. Release.
    logger.info("[Cycle] Releasing gripper")
    low.set_gripper_open(arm)
    time.sleep(2.0)

    # 5. Retract to the ready configuration.
    logger.info("[Cycle] Retracting to ready")
    _return_arm_to_initial(low, np.asarray(initial_snapshot, dtype=np.float64), cols)


def _return_arm_to_initial(low, initial_17, cols) -> None:
    """Ramp one arm back to its initial observed joints (joint space)."""
    current = np.asarray(low.read_feedback().model_position, dtype=np.float64)
    retract_to_ready(low, current, np.asarray(initial_17, dtype=np.float64), cols)


def _return_to_initial_pose(low, initial_17, *, duration: float = 2.0) -> None:
    """Restore waist Z/pitch + both arms to the initial observation posture.

    LOW_LEVEL analog of ``replay_sdk_highlevel._return_to_initial_pose``: the
    waist is driven back with ``prepare_robot_posture`` and both arms are ramped
    in joint space to their ``initial_17`` snapshot values via
    :func:`retract_to_ready`, so the full 17-axis vector returns to the initial
    observation posture.
    """
    from curobo_sdk.api import prepare_robot_posture

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


# ---------------------------------------------------------------------------
# Replay entry point
# ---------------------------------------------------------------------------
def _move_arms_to_ready(low) -> None:
    """Move both arms to the ready pose via pinocchio IK.

    The ready pose is mapped into the URDF ``body_yaw_link`` frame via the
    calibrated ``S_T_B`` and solved with :func:`_solve_arm_ik`; an arm without a
    calibration (or an IK failure) is skipped with a warning.
    """
    from replay.replay_curobo_lowlevel import _ready_pose_base

    for arm in ("left", "right"):
        b_t_e = _ready_pose_base(low._robot, arm)
        if b_t_e is None:
            logger.warning(
                f"[Ready] No S_T_B calibration / ready target for {arm}; skipping."
            )
            continue
        q_7 = _solve_arm_ik(arm, b_t_e)
        if q_7 is None:
            logger.error(f"[Ready] pinocchio IK failed for {arm}; skipping.")
            continue
        logger.info(f"[Ready] pinocchio IK {arm}: {np.round(q_7, 4).tolist()}")
        import pdb; pdb.set_trace()
        _ramp_to_joints(low, arm, q_7, duration=10.0)


def run_pinocchio_lowlevel_replay(
    scene_dir,
    grasps_dir=None,
    top_grasps=1,
    rounds=1,
    *,
    fake=False,
) -> int:
    """Replay the grasp plan with pinocchio IK + LOW_LEVEL SDK execution."""
    from curobo_sdk.api import create_low_level_robot, prepare_robot_posture
    from replay.replay_common import collect_grasp_plan

    low = create_low_level_robot(fake=fake)
    low.ensure_connected_low_level(connect=True, init=True)
    try:
        prepare_robot_posture(low, 0.0, 0.0, _WAIST_NORMAL_Z, _WAIST_PITCH)
        # Diagnose the SDK-vs-URDF base-frame offset from the current static arm
        # pose (waist-only move), so _move_arms_to_ready can map the SDK ready
        # target into the base frame.  S_T_B itself is measured live per call.
        if not fake:
            from pinocchio_ik.ik import verify_fk_against_sdk

            verify_fk_against_sdk(
                low._robot, side=None, samples=3, settle_s=0.3
            )
        _move_arms_to_ready(low)

        imu_wxyz = read_imu_wxyz(low)
        initial_snapshot = np.asarray(
            low.read_feedback().model_position, dtype=np.float64
        )
        world_T_base = build_world_T_base(imu_wxyz, initial_snapshot)
        grasp_T_wrist = build_grasp_T_wrist()

        plan = collect_grasp_plan(scene_dir, grasps_dir=grasps_dir, top_grasps=top_grasps)
        if not plan:
            logger.warning("[Replay] Empty grasp plan; nothing to execute.")
            return 0

        arm_cols = {
            a: tuple(
                ZERITH_ACTIVE_JOINTS.index(n) for n in ZERITH_ARM_JOINTS[a]
            )
            for a in ("left", "right")
        }

        for r in range(max(1, int(rounds))):
            logger.info(f"[Replay] ======== round {r + 1}/{max(1, int(rounds))} ========")
            for gripper, label, _gidx, grasp4x4_world in plan:
                if gripper not in _GRIPPER_TO_ARM:
                    logger.warning(
                        f"[Replay] Unknown gripper '{gripper}'; skipping."
                    )
                    continue
                arm = _GRIPPER_TO_ARM[gripper]
                grasp_cycle(
                    low,
                    arm,
                    grasp4x4_world,
                    label,
                    world_T_base=world_T_base,
                    grasp_T_wrist=grasp_T_wrist,
                    initial_snapshot=initial_snapshot,
                    cols=arm_cols[arm],
                )
        logger.info("[Replay] All rounds complete.")
        return 0
    finally:
        low.close()