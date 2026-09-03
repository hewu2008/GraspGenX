"""Low-level robot motion primitives: waist, chassis, and arm interpolation."""

from __future__ import annotations

import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from lib_h1_sdk_python import ArmAction, ArmPose, ArmEndPose, EtherCAT_Motor_Index

from .config import (
    RATE_HZ, DT, TARGET_ARM, WAIST_PITCH, WAIST_MOVE_DURATION,
    APPROACH_MAX_TRANS_SPEED_MPS, APPROACH_MAX_ANG_SPEED_RPS,
    APPROACH_MIN_DURATION_S,
    APPROACH_ARRIVE_POS_TOL_M, APPROACH_ARRIVE_ANG_TOL_DEG,
    APPROACH_ARRIVE_TIMEOUT_EXTRA_S,
)
from .logging_utils import get_logger

logger = get_logger(__name__)


def compose_relative_pose(start_xyz, start_quat, relative_xyz, relative_quat):
    """Compose an SDK pose with a target transform expressed in the hand frame.

    ``calculate_target_relative_pose`` returns ``current_E_T_target_E``.  The
    SDK, however, expects ``zero_E_T_target_E`` in ``setArm_high``.  Translation
    therefore has to be rotated by the current hand orientation and rotation
    has to be composed; component-wise addition is only valid when the current
    hand orientation is identity.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    relative_xyz = np.asarray(relative_xyz, dtype=np.float64)
    start_rotation = R.from_quat(start_quat)
    relative_rotation = R.from_quat(relative_quat)

    absolute_xyz = start_xyz + start_rotation.apply(relative_xyz)
    absolute_quat = (start_rotation * relative_rotation).as_quat()
    return absolute_xyz, absolute_quat


def compute_approach_duration(start_xyz, start_quat, target_xyz, target_quat):
    """Derive an approach duration from the actual commanded motion.

    The straight-line approach ("stage 3") formerly used a FIXED 1s duration, so
    the commanded end-effector line/angular speed scaled with the approach
    distance+rotation and exceeded the SDK tracking capability, leaving the arm
    ~2-7cm short (sdk_accuracy exp, +X undershoot; worse when large rotation is
    demanded). The duration is now chosen so speeds stay below the config limits.

    Args:
        start_xyz: absolute start SDK position.
        start_quat: absolute start SDK quaternion.
        target_xyz: absolute target SDK position.
        target_quat: absolute target SDK quaternion.

    Returns:
        float duration in seconds that respects the speed limits (with a floor).
    """
    dist = np.linalg.norm(np.asarray(target_xyz, dtype=np.float64)
                          - np.asarray(start_xyz, dtype=np.float64))
    rot_delta = R.from_quat(target_quat) * R.from_quat(start_quat).inv()
    ang = float(rot_delta.magnitude())
    return max(
        dist / APPROACH_MAX_TRANS_SPEED_MPS,
        ang / APPROACH_MAX_ANG_SPEED_RPS,
        APPROACH_MIN_DURATION_S,
    )


def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    logger.info("[A] Adjusting robot to initial observation posture...")
    waist_steps = int(3.0 * RATE_HZ)

    waist_pose = ArmPose()
    waist_pose.x = waist_pose.y = 0.0
    waist_pose.z = cur_waist_z
    waist_pose.roll = 0.0
    waist_pose.pitch = cur_waist_pitch
    waist_pose.yaw = 0.0

    diff_z = tar_waist_z - cur_waist_z
    diff_pitch = tar_waist_pitch - cur_waist_pitch

    def send():
        end = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end)

    for _ in range(1, waist_steps + 1):
        waist_pose.z += diff_z / waist_steps
        send()
        time.sleep(DT)

    for _ in range(1, waist_steps + 1):
        waist_pose.pitch += diff_pitch / waist_steps
        send()
        time.sleep(DT)

    time.sleep(1.5)


def _move_arm_to_pose(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
             duration=3.0, rate=RATE_HZ, dt=DT):
    """Smooth interpolation for a single arm."""
    steps = int(duration * rate)
    sx, sy, sz = start_xyz
    dx, dy, dz = dest_xyz

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps + 1):
        ratio = i / steps
        x = sx + (dx - sx) * ratio
        y = sy + (dy - sy) * ratio
        z = sz + (dz - sz) * ratio
        quat = slerp(ratio).as_quat()

        pose = ArmEndPose()
        pose.position = [x, y, z]
        pose.rotation = [quat[0], quat[1], quat[2], quat[3]]

        robot.setArm_high(arm, pose)
        time.sleep(dt)


def _hold_until_arrived(robot, arm, dest_xyz, dest_quat, poll_period=0.1):
    """Keep re-sending the final target until the arm converges to it.

    ``setArm_high`` is fire-and-forget: once the interpolation loop finishes,
    the arm may still be lagging behind the last commanded pose (sdk_accuracy
    exp run 5/6, ~6cm / ~13deg residual). This routine periodically re-sends
    the destination pose and polls ``getHandRelative`` until the residual
    position/orientation error drops below the config tolerances, or the
    timeout (expected duration + extra) elapses.

    Returns True when converged within tolerance, False on timeout.
    """
    dest_xyz = np.asarray(dest_xyz, dtype=np.float64)
    dest_rot = R.from_quat(dest_quat)
    deadline = time.time() + APPROACH_ARRIVE_TIMEOUT_EXTRA_S
    pos_err = 0.0
    ang_err_deg = 0.0

    while time.time() < deadline:
        # Re-send the destination so the controller keeps tracking it.
        pose = ArmEndPose()
        pose.position = list(dest_xyz)
        pose.rotation = list(dest_quat)
        robot.setArm_high(arm, pose)

        time.sleep(poll_period)

        ok, arm_state = robot.getHandRelative(arm)
        if not ok or arm_state is None:
            continue
        cur_pos = np.asarray(arm_state.position, dtype=np.float64)
        cur_quat = arm_state.rotation
        if cur_quat is None:
            continue
        pos_err = float(np.linalg.norm(cur_pos - dest_xyz))
        ang_err_deg = np.degrees(
            (dest_rot * R.from_quat(cur_quat).inv()).magnitude()
        )
        if (pos_err <= APPROACH_ARRIVE_POS_TOL_M
                and ang_err_deg <= APPROACH_ARRIVE_ANG_TOL_DEG):
            logger.info(
                f"[Move] Arm {arm} arrived: pos_err={pos_err:.4f} m, "
                f"ang_err={ang_err_deg:.2f} deg."
            )
            return True

    logger.warning(
        f"[Move] Arm {arm} did not converge within "
        f"{APPROACH_ARRIVE_TIMEOUT_EXTRA_S:.1f}s: "
        f"residual pos_err={pos_err:.4f} m, ang_err={ang_err_deg:.2f} deg."
    )
    _log_arm_joint_states(robot, arm)
    return False


def _log_arm_joint_states(robot, arm):
    """Print the target arm's joint motor states (pos/speed/torque/error_flag).

    Called after a non-convergence timeout in :func:`_hold_until_arrived` to help
    tell whether the residual is a tracking lag or an actual hard limit / pinch.
    A joint whose position stopped changing while still being commanded, or an
    ``error_flag != 0``, points to a limit rather than simple undershoot.
    """
    side = "LEFT" if arm == ArmAction.LEFT_ARM else "RIGHT"
    logger.warning(f"[Move] Arm {arm} joint states (limit check):")
    for i in range(1, 8):  # arm joints MOTOR_{LEFT,RIGHT}_ARM_1..7
        mid = getattr(EtherCAT_Motor_Index, f"MOTOR_{side}_ARM_{i}", None)
        if mid is None:
            continue
        ok, info = robot.getMotorState(mid)
        if not ok or info is None:
            logger.warning(f"  motor {mid}: <unreadable>")
            continue
        logger.warning(
            f"  motor {mid}: pos={info.Position_Actual:.4f} "
            f"speed={info.Speed_Actual:.3f} torque={info.Torque_Actual:.3f} "
            f"error_flag={info.Error_flag}"
        )


def _move_arm_to_pose_adaptive(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat):
    """Interpolate to a target pose, then hold the command until arrival.

    Two-step fix for the undershoot documented in the sdk_accuracy experiments
    (runs 1-6: the arm stops short of the commanded pose, worst when a large
    rotation is involved):
      1. the interpolation duration is derived adaptively from the commanded
         translation/rotation via ``compute_approach_duration`` so commanded
         speeds stay within the SDK tracking limits;
      2. after the interpolation finishes, the destination pose is re-sent and
         the arm pose polled (``getHandRelative``) until the residual error is
         below the arrival tolerances, instead of returning after a fixed
         sleep(0.5) that read a mid-motion pose.
    """
    expected_duration = compute_approach_duration(
        start_xyz, start_quat, dest_xyz, dest_quat
    )
    logger.info(f"[Move] Adaptive move duration: {expected_duration:.2f} s")
    _move_arm_to_pose(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
                      expected_duration)
    _hold_until_arrived(robot, arm, dest_xyz, dest_quat)


def move_arm_to_ready_pose(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    """Move both arms from relative zero to a symmetric ready pose.

    The left arm goes to ``dest_xyz``; the right arm goes to the Y-mirrored
    pose. Both arms interpolate concurrently so they reach the standby pose
    at the same time.
    """
    logger.info("[E] Moving both arms from relative zero to target smoothly...")

    left_dest = [dest_xyz[0], dest_xyz[1], dest_xyz[2]]
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]

    t_left = threading.Thread(
        target=_move_arm_to_pose,
        args=(robot, ArmAction.LEFT_ARM, cur_xyz, cur_quat, left_dest, dest_quat, 2),
    )
    t_right = threading.Thread(
        target=_move_arm_to_pose,
        args=(robot, ArmAction.RIGHT_ARM, cur_xyz, cur_quat, right_dest, dest_quat, 2),
    )

    t_left.start()
    t_right.start()
    t_left.join()
    t_right.join()

    time.sleep(0.5)
    logger.info(" -> Arms reached target.")


def move_arm_relative(robot, dx, dy, dz, arm=TARGET_ARM):
    """Relative retract move for the target arm."""
    duration_arm = 2.0
    steps_arm = int(duration_arm * RATE_HZ)

    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    start_x, start_y, start_z = arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]
    start_quat = arm_quat_rel

    dest_x = arm_pos_rel[0] + dx
    dest_y = arm_pos_rel[1] + dy
    dest_z = arm_pos_rel[2] + dz
    dest_quat = arm_quat_rel

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps_arm + 1):
        ratio = i / steps_arm
        x = start_x + (dest_x - start_x) * ratio
        y = start_y + (dest_y - start_y) * ratio
        z = start_z + (dest_z - start_z) * ratio
        interp_quat = slerp(ratio).as_quat()

        target_end_pose = ArmEndPose()
        target_end_pose.position = [x, y, z]
        target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        robot.setArm_high(arm, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    logger.info(" -> Reached target.")


def get_arm_relative_pose(robot, arm=TARGET_ARM):
    """Get the current SDK pose of ``arm`` relative to its own zero position.

    Mirrors the state read used inside ``move_arm_to_grasp``/``move_arm_relative``:
    the returned position/quaternion is the "current" pose that ``setArm_high``
    expects when composing a relative target (see ``compose_relative_pose``).

    Args:
        robot: connected H1Robot instance.
        arm: arm identifier (default TARGET_ARM).

    Returns:
        (pos, quat) as read from the SDK, or (None, None) if the read failed
        or returned incomplete data.
    """
    ok, arm_state = robot.getHandRelative(arm)
    if not ok or arm_state is None:
        logger.error(f"[Move] Failed to read pose for arm {arm}.")
        return None, None
    pos = getattr(arm_state, "position", None)
    quat = getattr(arm_state, "rotation", None)
    if pos is None or quat is None:
        logger.error(f"[Move] Incomplete pose for arm {arm}.")
        return None, None
    logger.info(
        f"[Move] Arm {arm} relative pose: pos={list(pos)}, quat={list(quat)}"
    )
    return pos, quat


def move_arm_to_grasp(robot, target_pos, target_quat, arm=TARGET_ARM):
    """Approach: rotate to grasp orientation, then straight-line approach.

    target_pos/target_quat form the target transform relative to the current
    SDK end-effector. Flow:
      1. rotate in place to the grasp orientation, keeping the current position
         (grasp approach axis = hand +X / finger long axis);
      2. move straight along the grasp axis to a pre-grasp point 10 cm behind
         the target;
      3. approach straight along the grasp axis to the target point.

    Returns the absolute (SDK zero-frame) pre-grasp waypoint so the caller can
    retract back to it after grasping.
    """
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    target_abs, target_abs_quat = compose_relative_pose(
        arm_pos_rel, arm_quat_rel, target_pos, target_quat
    )

    # Approach axis: the hand +X (finger long axis) in the target orientation.
    approach_dir = R.from_quat(target_abs_quat).as_matrix()[:, 0]
    logger.info(
        f"[Move] Resolved SDK EEF target: position={target_abs.tolist()}, "
        f"quaternion={target_abs_quat.tolist()}"
    )

    # Stage 1: rotate in place HALFWAY to the grasp orientation (position kept).
    # The remaining half is finished during Stage 2's translation to the
    # pre-grasp point, so no single stage carries a large pure rotation.
    pre_xyz = (target_abs - 0.10 * approach_dir).tolist()
    mid_quat = Slerp([0, 1], R.from_quat([arm_quat_rel, target_abs_quat]))(0.5).as_quat().tolist()
    logger.info(
        f"[Move] Rotate in place half-angle to grasp orientation "
        f"(position unchanged): pos={list(arm_pos_rel)}, quat(mid)={mid_quat}"
    )
    import pdb; pdb.set_trace()
    _move_arm_to_pose_adaptive(robot, arm, arm_pos_rel, arm_quat_rel,
                      arm_pos_rel, mid_quat)

    logger.info(
        f"[Move] Near pre-grasp waypoint: "
        f"pos={pre_xyz}, quat={target_abs_quat.tolist()}"
    )
    import pdb; pdb.set_trace()
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose_adaptive(robot, arm, arm_pos_rel, arm_quat_rel,
                      pre_xyz, target_abs_quat.tolist())

    # Stage 2: straight-line approach along the grasp axis to the target.
    logger.info(f"[Move] Approach to grasp pose: {target_abs.tolist()}")
    import pdb; pdb.set_trace()
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    _move_arm_to_pose_adaptive(robot, arm, arm_pos_rel, arm_quat_rel,
                               target_abs.tolist(), target_abs_quat.tolist())
    logger.info(" -> Reached grasp pose.")
    return pre_xyz


def move_chassis(robot: "H1Robot", dist):
    dt = 0.2
    speed = 0.2
    distance = abs(dist)
    direction = 1 if dist >= 0 else -1
    velocity = speed * direction
    duration = distance / speed

    logger.info(f"[Chassis] Forward: speed={speed} m/s, distance={distance} m, duration={duration:.2f}s")
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            loop_start = time.perf_counter()
            robot.setChassis_high(velocity, 0.0)

            elapsed_time = time.time() - start_time
            remaining_distance = distance - (speed * elapsed_time)
            logger.info(f"[Chassis] elapsed {elapsed_time:.2f}s, remaining {remaining_distance:.2f} m")

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        robot.setChassis_high(0.0, 0.0)
        logger.info(f"[Chassis] Forward {distance} m complete")

    except KeyboardInterrupt:
        logger.warning("[Chassis] Manual stop")
        robot.setChassis_high(0.0, 0.0)


def move_waist_z(robot, start_z, target_z, duration=WAIST_MOVE_DURATION):
    """Smoothly move the waist along Z.

    The flow restores the waist to WAIST_NORMAL_Z after each grasp group, so
    the known start_z and target_z are used directly instead of reading a
    waist state interface that the SDK does not expose.
    """
    steps = max(1, int(duration * RATE_HZ))

    waist_pose = ArmPose()
    waist_pose.x = 0.0
    waist_pose.y = 0.0
    waist_pose.z = start_z
    waist_pose.roll = 0.0
    waist_pose.pitch = WAIST_PITCH
    waist_pose.yaw = 0.0

    logger.info(f"[Waist] Z: {start_z:.2f} m -> {target_z:.2f} m")
    for i in range(1, steps + 1):
        ratio = i / steps
        waist_pose.z = start_z + (target_z - start_z) * ratio
        waist_end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(waist_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
