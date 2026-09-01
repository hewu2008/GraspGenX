"""Low-level robot motion primitives: waist, chassis, and arm interpolation."""

from __future__ import annotations

import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from lib_h1_sdk_python import ArmAction, ArmPose, ArmEndPose

from .config import (
    RATE_HZ, DT, TARGET_ARM, WAIST_PITCH, WAIST_MOVE_DURATION,
    APPROACH_MAX_TRANS_SPEED_MPS, APPROACH_MAX_ANG_SPEED_RPS,
    APPROACH_MIN_DURATION_S,
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


def _wait_move_done(robot, arm, expected_duration, timeout_extra=5.0):
    """Poll the high-level controller until the arm Move completes.

    ``setArmMove_high`` is issued with ``block=True``; we additionally poll
    ``getHighLevelState`` so callers can rely on the arm having actually reached
    the target (state 4 = DONE). Bounded by a timeout derived from the expected
    duration to avoid hanging on a lost/failed command.
    """
    deadline = time.time() + expected_duration + timeout_extra
    while time.time() < deadline:
        try:
            ok, hls = robot.getHighLevelState()
        except Exception:
            ok, hls = False, None
        if ok and hls is not None:
            state = getattr(hls, "state", None)
            progress = getattr(hls, "progress", None)
            if state == 4:  # HighLevelState.DONE
                logger.info(f"[Move] Arm {arm} move done (progress={progress}).")
                return
            if state == 5:  # HighLevelState.ERROR
                logger.error(f"[Move] Arm {arm} high-level move error.")
                return
        time.sleep(DT)
    logger.warning(f"[Move] Timed out waiting for arm {arm} move completion.")


def _move_arm_to_pose_adaptive(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat):
    """Move to a target pose using the SDK native Move primitive.

    Uses ``setArmMove_high`` (SDK 1.3.9), which generates and executes its own
    trajectory from the requested ``duration`` (derived via
    ``compute_approach_duration`` to respect the end-effector speed limits).
    The former per-2ms ``setArm_high`` interpolation is not used because its
    incremental commands lag the servo and undershoot the target (sdk_accuracy
    exp, +X shortfall scaling with commanded distance/rotation).

    Blocks until the move is reported done so callers can rely on arrival.
    """
    pose = ArmEndPose()
    pose.position = list(dest_xyz)
    pose.rotation = list(dest_quat)

    # Plan A: leave duration=0 so the SDK auto-derives the trajectory from the
    # requested end-effector speed. The adaptive duration is still computed only
    # as a timeout bound for the completion wait, not passed to the Move.
    expected_duration = compute_approach_duration(
        start_xyz, start_quat, dest_xyz, dest_quat
    )
    logger.info(f"[Move] Adaptive move duration: {expected_duration:.2f} s")
    return _move_arm_to_pose(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat, expected_duration)


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
    """Three-stage approach: pre-grasp, orient, straight-line approach.

    target_pos/target_quat form the target transform relative to the current
    SDK end-effector. Flow:
      1. move to a pre-grasp point 10 cm behind the target, along the opposite
         of the grasp approach axis (hand +X / finger long axis);
      2. rotate in place to the grasp orientation;
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

    # Stage 1+2 (merged): move to the pre-grasp waypoint while simultaneously
    # rotating to the grasp orientation.
    pre_xyz = (target_abs - 0.10 * approach_dir).tolist()
    logger.info(
        f"[Move] Pre-grasp waypoint (with grasp orientation): "
        f"pos={pre_xyz}, quat={target_abs_quat.tolist()}"
    )
    import pdb; pdb.set_trace()
    _move_arm_to_pose_adaptive(robot, arm, arm_pos_rel, arm_quat_rel,
                      pre_xyz, target_abs_quat.tolist())
    time.sleep(0.5)

    # Stage 3: straight-line approach along the grasp axis to the target.
    logger.info(f"[Move] Approach to grasp pose: {target_abs.tolist()}")
    import pdb; pdb.set_trace()
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    _move_arm_to_pose_adaptive(robot, arm, arm_pos_rel, arm_quat_rel,
                               target_abs.tolist(), target_abs_quat.tolist())
    time.sleep(0.5)
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
