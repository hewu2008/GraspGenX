"""Low-level robot motion primitives: waist, chassis, and arm interpolation."""

from __future__ import annotations

import threading
import time
from scipy.spatial.transform import Rotation as R, Slerp

from lib_h1_sdk_python import ArmAction, ArmPose, ArmEndPose

from .config import RATE_HZ, DT, TARGET_ARM, WAIST_PITCH, WAIST_MOVE_DURATION
from .logging_utils import get_logger

logger = get_logger(__name__)


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


def move_arm_relative(robot, dx, dy, dz):
    """Relative retract move for the target arm."""
    duration_arm = 2.0
    steps_arm = int(duration_arm * RATE_HZ)

    _, arm_state = robot.getHandRelative(TARGET_ARM)
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

        robot.setArm_high(TARGET_ARM, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    logger.info(" -> Reached target.")


def move_arm_to_grasp(robot, target_pos, target_quat):
    """Three-stage approach: pre-grasp, orient, straight-line approach.

    target_pos is the grasp point relative to the current hand, target_quat the
    grasp orientation. Flow:
      1. move to a pre-grasp point 10 cm behind the target, along the opposite
         of the grasp approach axis (hand +X / finger long axis);
      2. rotate in place to the grasp orientation;
      3. approach straight along the grasp axis to the target point.
    """
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_abs = np.asarray(arm_pos_rel, dtype=np.float64) + target_pos

    # Approach axis: the hand +X (finger long axis) in the target orientation.
    approach_dir = R.from_quat(target_quat).as_matrix()[:, 0]

    # Stage 1: pre-grasp waypoint, 10 cm behind the target along -approach.
    pre_xyz = (target_abs - 0.10 * approach_dir).tolist()
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel,
                      pre_xyz, arm_quat_rel, 2)
    time.sleep(0.5)

    # Stage 2: rotate in place to the grasp orientation.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel,
                      [arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]],
                      target_quat, 1)
    time.sleep(0.5)

    # Stage 3: straight-line approach along the grasp axis to the target.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel,
                      target_abs.tolist(), arm_quat_rel, 1)
    time.sleep(0.5)
    logger.info(" -> Reached grasp pose.")


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
