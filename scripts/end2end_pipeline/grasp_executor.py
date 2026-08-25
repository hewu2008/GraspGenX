"""Grasp pose computation and the single-arm grasp execution cycle."""

import time

import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import Motor_Control

from .config import (
    TARGET_ARM,
    TARGET_GRIPPER_MOTOR,
    WAIST_NORMAL_Z,
    WAIST_RELEASE_Z,
    GRIPPER_RELEASE_WAIT,
)
from .robot_motion import _move_arm_to_pose, move_arm_relative, move_arm_to_grasp, move_waist_z
from .logging_utils import get_logger

logger = get_logger(__name__)


# ================= 4. Grasp computation and execution =================
def load_pose_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    matrix_lines = [line.strip().split() for line in lines[:4]]
    pose = np.array(matrix_lines, dtype=np.float64).reshape(4, 4)
    angle = float(lines[4].strip())
    return pose, angle


def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """Coordinate transform: perception pose -> arm-relative target pose.

    Chain (each matrix maps the object one frame closer to the left arm):
        T1 : head-camera increment from getHeadCameraRelative()  (camera->head-zero)
        T2 : fixed URDF neck_camera_link -> chassis(dipan) transform, at the
             URDF *zero pose* (all revolute/prismatic joints at 0)
        T3 : fixed left-arm mounting offset (chassis -> arm mount frame)
        T4 : inverse of the current arm pose (getHandRelative) -> arm-relative

    T_obj_in_arm = T4 @ T3 @ T2 @ T1 @ T_obj_cam maps the target pose from the
    camera frame into the left-arm relative frame about the arm mount.
    """
    # T1: head camera pose relative to its own zero position (incremental neck
    # motion only; the waist/body angles are NOT included here).
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    # T2: fixed camera -> chassis transform at zero pose.
    #   rotation: URDF joint neck_camera_joint  rpy = (-1.78023593, 0, -1.57079633)
    #             (extrinsic XYZ, same as camera_pose.py _R_NC)
    #   translation: sum of the URDF joint origins along
    #             dipan_link -> daogui_link -> body_pitch_link -> body_yaw_link
    #             -> neck_yaw_link -> neck_pitch_link -> neck_camera_link
    #             at zero pose:
    #       body_pitch_joint [0.1518, 0, 0.1275]
    #       body_yaw_joint   [2.71e-5, -1.21e-4, 0.1572]
    #       neck_yaw_joint   [-2.71e-5, 1.21e-4, 0.2491]
    #       neck_pitch_joint [0, 0, 0.11]
    #       neck_camera_joint[0.06756, 0.0325, -0.03633]
    #     sum = [0.2194, 0.0325, 0.6075]
    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('XYZ', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    # T3: left-arm mounting offset, i.e. the (chassis/body_yaw_link) -> left-arm
    # mounting reference frame. Empirically tuned constant (not a strict zero-pose
    # URDF sum; the URDF arm chain sum differs by a few cm).
    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    # T4: inverse of the current arm pose. getHandRelative() returns the left arm
    # pose relative to its own zero position; inverting it expresses the object
    # relative to the *current* arm, i.e. the target the robot must reach.
    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T4 = np.linalg.inv(T4_inv)

    T_obj_in_arm = T4 @ T3 @ T2 @ T1 @ T_obj_cam

    # Optional local grasp offset on top of the arm-relative pose (identity = none).
    T_grasp_local = np.eye(4)
    T_final = T_obj_in_arm @ T_grasp_local

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()
    return target_pos, target_quat


def select_arm(robot, pose_path):
    """Resolve the target pose for the single (left) arm."""
    logger.info("[D] Reading pose state and solving target matrix...")
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None)
    cam_quat_rel = getattr(cam_state, "rotation", None)

    ok_arm, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    if not (ok_cam and ok_arm):
        logger.error("Sensor pose retrieval failed!")
        return None, None

    T_obj_cam, angle = load_pose_matrix(pose_path)
    target_pos, _ = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam
    )
    return target_pos, angle


def resolve_grasp_target(robot, T_obj_cam):
    """Read camera/arm state and solve for the left-arm relative grasp target.

    Unlike ``select_arm`` (which returns the drop pitch angle), this returns the
    full target pose (translation + full orientation quaternion) so the exact
    grasp orientation produced by GraspGenX can be preserved.

    Args:
        robot: connected H1Robot instance.
        T_obj_cam: 4x4 target (object or grasp) pose in the camera frame.

    Returns:
        target_pos, target_quat (full orientation), or (None, None) on failure.
    """
    logger.info("[D] Reading pose state and solving target matrix...")
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None)
    cam_quat_rel = getattr(cam_state, "rotation", None)

    ok_arm, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    if not (ok_cam and ok_arm):
        logger.error("Sensor pose retrieval failed!")
        return None, None

    target_pos, target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam
    )
    return target_pos, target_quat


def grasp_object(robot, target_pos, target_quat):
    """Full single-arm grasp cycle: approach, close, lift, place, release, retract."""
    logger.info(f" -> Target relative translation: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    pdb.set_trace()
    logger.info("[E] Moving arm to grasp target smoothly...")
    move_arm_to_grasp(robot, target_pos, target_quat)

    pdb.set_trace()
    logger.info("[F] Closing gripper to grasp...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(TARGET_GRIPPER_MOTOR, close_cmd)
    time.sleep(2.0)

    # Lift after grasp.
    pdb.set_trace()
    move_arm_relative(robot, -0.2, 0, 0.05)
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # Move to placement position.
    pdb.set_trace()
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.17, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # Drop: lower the waist before releasing.
    pdb.set_trace()
    logger.info("[Waist] Lowering waist before release...")
    move_waist_z(robot, WAIST_NORMAL_Z, WAIST_RELEASE_Z)
    time.sleep(1.0)

    close_cmd.Position = 0.0
    robot.setGripper_high(TARGET_GRIPPER_MOTOR, close_cmd)
    time.sleep(GRIPPER_RELEASE_WAIT)

    # Restore the waist before retracting the arm.
    pdb.set_trace()
    logger.info("[Waist] Restoring waist after release...")
    move_waist_z(robot, WAIST_RELEASE_Z, WAIST_NORMAL_Z)

    # Retract to a safe waypoint.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    pdb.set_trace()
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    pdb.set_trace()
    _move_arm_to_pose(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [-0.1, 0.0, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
