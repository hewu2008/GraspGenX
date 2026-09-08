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

from .replay_sdk_highlevel import world_grasp_to_hand_cam, resolve_grasp_target_hand

from replay.replay_curobo_lowlevel import (
    _GRIPPER_TO_ARM,
    _WAIST_NORMAL_Z,
    _WAIST_PITCH,
    retract_to_ready,
)

from end2end_pipeline.robot_motion import compose_relative_pose, get_arm_relative_pose

_WAIST_Z_JOINT = "daogui_joint"
_WAIST_PITCH_JOINT = "body_pitch_joint"

logger = get_logger(__name__)


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


def grasp_cycle(
    driver, arm, target_pos, target_quat, label, initial_snapshot, cols
) -> None:
    """Full pinocchio-IK grasp cycle for one world grasp pose."""
    logger.info(
        f"[Cycle][{arm}] {label}: pos={target_pos.tolist()}, quat={target_quat.tolist()}"
    )

    target_arm = driver._sdk.ArmAction.LEFT_ARM if arm == "left" else driver._sdk.ArmAction.RIGHT_ARM
    arm_pos_rel, arm_quat_rel = get_arm_relative_pose(driver, target_arm)
    target_abs, target_abs_quat = compose_relative_pose(
        arm_pos_rel, arm_quat_rel, target_pos, target_quat
    )
    _move_arm_to_ready(driver, arm, ready_xyz=target_abs, ready_quat=target_abs_quat)

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
    return _move_arm_to_pose(driver, arm, ready_xyz, ready_quat, duration=3.0)

def _move_arm_to_pose(driver, arm, target_pos, target_quat, duration=3.0):
    """Move ``arm``'s EEF to the target pose via pinocchio IK."""
    from scipy.spatial.transform import Rotation as R
    from curobo_planning.frames import invert_transform
    from pinocchio_ik.ik import get_s_t_b

    b_t_e = get_s_t_b(driver._robot, arm)
    if b_t_e is None:
        logger.warning(
            f"[Move] No S_T_B calibration for '{arm}'; skipping."
        )
        return
    s_ready = np.eye(4, dtype=np.float64)
    s_ready[:3, 3] = [target_pos[0], target_pos[1], target_pos[2]]
    s_ready[:3, :3] = R.from_quat(target_quat).as_matrix()
    b_t_e = invert_transform(b_t_e) @ s_ready

    q_7 = _solve_arm_ik(arm, b_t_e)
    if q_7 is None:
        logger.error(f"[Ready] pinocchio IK failed for {arm}; skipping.")
        return
    logger.info(f"[Move] pinocchio IK {arm}: {np.round(q_7, 4).tolist()}")
    _ramp_to_joints(driver, arm, q_7, duration=duration)


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

        initial_snapshot = np.asarray(
            driver.read_feedback().model_position, dtype=np.float64
        )
        logger.info(f"[Replay] initial_snapshot: {initial_snapshot}")

        arm_cols = {
            a: tuple(ACTIVE_JOINTS.index(n) for n in ARM_JOINTS[a])
            for a in ("left", "right")
        }

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
                arm_action = driver._sdk.ArmAction
                high_arm = arm_action.LEFT_ARM if arm == "left" else arm_action.RIGHT_ARM
                logger.info(
                    f"[Replay] arm: {arm}, label: {label}, grasp4x4_world: {grasp4x4_world}"
                )

                T_obj_cam = world_grasp_to_hand_cam(driver._robot, high_arm, grasp4x4_world)
                if T_obj_cam is None:
                    continue
                logger.info(f"[Replay] T_obj_cam: {T_obj_cam}")

                target_pos, target_quat = resolve_grasp_target_hand(driver._robot, T_obj_cam)
                if target_pos is None:
                    logger.error("[Replay] resolve_grasp_target_hand failed; skipping grasp.")
                    continue
                logger.info(f"[Replay] target_pos: {target_pos}, target_quat: {target_quat}")

                grasp_cycle(
                    driver,
                    arm,
                    target_pos,
                    target_quat,
                    label,
                    initial_snapshot=initial_snapshot,
                    cols=arm_cols[arm],
                )
        logger.info("[Replay] All rounds complete.")
        return 0
    finally:
        driver.close()