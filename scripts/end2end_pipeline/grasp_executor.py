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
    WRIST_TO_SDK_EEF_OFFSET_M,
    GRASP_TRIM_OFFSET_M,
)
from .robot_motion import (
    _move_arm_to_pose,
    get_arm_relative_pose,
    move_arm_relative,
    move_arm_to_grasp,
    move_waist_z,
)
from .logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Left-hand camera hand-eye transform (fixed URDF offset).
#
# The left wrist camera (rs/cam_left_wrist, left_jaw_camera_link) is rigidly
# mounted on the arm, so the change of frame from the camera optical frame to
# the SDK arm end-effector frame (left_end_effector_link) is a CONSTANT matrix
# derived only from the Zerith URDF — no IMU/FK/waist needed.
#
#   left_jaw_camera_joint     origin xyz=(0.11933, 0.009, 0.060373)
#                             rpy=(-2.0071, 0, -1.5708)   (extrinsic XYZ; mirrors
#                             camera_pose._R_NC intrinsic "XYZ" convention)
#   left_end_effector_joint   origin xyz=(0.1435, 0, 0)
#
# W = left_wrist_pitch_link, C = hand camera, E = left_end_effector_link.
#   W_T_E = [I, (0.1435,0,0)]  ->  E_T_C = inv(W_T_E) @ W_T_C
# i.e. rotation = W_R_C, translation = t_C - t_E.
# ---------------------------------------------------------------------------
CAM_TO_SDK_EEF_HAND = np.eye(4)
CAM_TO_SDK_EEF_HAND[:3, :3] = R.from_euler(
    "xyz", [-2.0071, 0.0, -1.5708], degrees=False
).as_matrix()
CAM_TO_SDK_EEF_HAND[:3, 3] = (
    np.array([0.11933, 0.009, 0.060373]) - np.array([0.1435, 0.0, 0.0])
)


def _grasp_in_eef_to_sdk_target(T_grasp_in_eef):
    """Convert a grasp base pose (relative to the current SDK eef) into the
    final SDK end-effector target.
    """
    # Coordinate-axis conversion: GraspGenX grasp frame vs. the ZR left-hand
    # frame (left_gripper.urdf, fingers extend along +X, close along ±Y):
    #   GraspGenX grasp: Z = approach (finger long axis), X = closing
    #   ZR hand:         X = approach (finger long axis), Y = closing
    # So the grasp frame's Z/X/Y must be relabelled to the hand frame's X/Y/Z.
    # The configured base_rotation gives the proper rotation (det=+1):
    #   G_T_U columns = [+grasp_Z, -grasp_X, -grasp_Y]
    # The saved GraspGenX pose is the normalized gripper BASE pose.  The
    # rotation below is G_T_U (GraspGenX base -> wrist-pitch base); their
    # origins coincide, hence its translation is zero.
    logger.info(
        f"[_grasp_in_eef_to_sdk_target]   translation={T_grasp_in_eef[:3, 3].tolist()}, "
        f"euler_xyz(deg)={R.from_matrix(T_grasp_in_eef[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
    )
    T_grasp_to_wrist = np.eye(4)
    T_grasp_to_wrist[:3, :3] = np.array(
        [[0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0]], dtype=np.float64)

    # setArm_high() controls the SDK arm end-effector, not the wrist-pitch
    # base.  In the Zerith URDF left_end_effector_joint is fixed 0.1435 m
    # along wrist local +X, plus the ~41.5 mm gripper-center offset.  Omitting
    # U_T_E leaves the real gripper behind the GraspGenX pose by this distance.
    T_wrist_to_sdk_eef = np.eye(4)
    T_wrist_to_sdk_eef[:3, 3] = WRIST_TO_SDK_EEF_OFFSET_M + GRASP_TRIM_OFFSET_M

    T_final = T_grasp_in_eef @ T_grasp_to_wrist @ T_wrist_to_sdk_eef
    logger.info(f"[Transform] after T4 / SDK EEF target (arm-relative):")
    logger.info(
        f"[Transform]   translation={T_final[:3, 3].tolist()}, "
        f"euler_xyz(deg)={R.from_matrix(T_final[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
    )
    return T_final[:3, 3], R.from_matrix(T_final[:3, :3]).as_quat()


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
    logger.info("[Arm] runtime inputs: "
                f"cam_pos_rel={list(cam_pos_rel)}, cam_quat_rel={list(cam_quat_rel)}, "
                f"arm_pos_rel={list(arm_pos_rel)}, arm_quat_rel={list(arm_quat_rel)}")

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
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
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

    def _dbg(label, T):
        """Log a transform's translation + euler_xyz(deg) for debugging."""
        logger.info(f"[Transform] {label}:")
        logger.info(f"[Transform]   translation={T[:3, 3].tolist()}")
        logger.info(f"[Transform]   euler_xyz(deg)={R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True).tolist()}")

    logger.info("[Transform] ---------- camera->arm chain ----------")
    _dbg("input T_obj_cam (camera frame)", T_obj_cam)
    _dbg("after T1 (T1 @ T_obj_cam, camera->head-zero)", T1 @ T_obj_cam)
    _dbg("after T2 (T2 @ T1 @ T_obj_cam, head-zero->chassis)", T2 @ T1 @ T_obj_cam)
    _dbg("after T3 (T3 @ T2 @ T1 @ T_obj_cam, chassis->arm-mount)",
         T3 @ T2 @ T1 @ T_obj_cam)

    T_obj_in_arm = T4 @ T3 @ T2 @ T1 @ T_obj_cam

    _dbg("before T_grasp_local (T_obj_in_arm)", T_obj_in_arm)

    return _grasp_in_eef_to_sdk_target(T_obj_in_arm)


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


def resolve_grasp_target_hand(robot, T_obj_cam):
    """Resolve a left-arm relative grasp target from the HAND camera.

    The hand (wrist) camera is rigidly mounted on the arm, so unlike the head
    path this needs no IMU/waist FK: the grasp pose in the camera frame maps to
    the current-SDK-eef frame by the fixed URDF offset ``CAM_TO_SDK_EEF_HAND``,
    then to the SDK eef target by the same tail alignment as the head path.

    Args:
        robot: connected H1Robot instance (unused; kept for a uniform signature).
        T_obj_cam: 4x4 target (object or grasp) pose in the HAND camera frame.

    Returns:
        target_pos, target_quat (full orientation), or (None, None).
    """
    T_grasp_in_eef = CAM_TO_SDK_EEF_HAND @ np.asarray(T_obj_cam, dtype=np.float64)
    if not np.isfinite(T_grasp_in_eef).all():
        logger.error("[Hand] Invalid grasp pose in hand camera frame.")
        return None, None
    logger.info(
        f"[Hand] T_grasp_in_eef pos={T_grasp_in_eef[:3, 3].tolist()}, "
        f"quat={R.from_matrix(T_grasp_in_eef[:3, :3]).as_quat().tolist()}, "
        f"euler_xyz(deg)={R.from_matrix(T_grasp_in_eef[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
    )
    target_pos, target_quat = _grasp_in_eef_to_sdk_target(T_grasp_in_eef)
    return target_pos, target_quat


def grasp_object(robot, target_pos, target_quat, arm=TARGET_ARM, gripper_motor=TARGET_GRIPPER_MOTOR):
    """Full single-arm grasp cycle: approach, close, lift, place, release, retract."""
    logger.info(
        f" -> Target (arm-relative): pos={target_pos.tolist()}, "
        f"quat={target_quat.tolist()}, "
        f"euler_xyz(deg)={R.from_quat(target_quat).as_euler('xyz', degrees=True).tolist()}"
    )

    # Debug: read the arm pose BEFORE approaching so we can compare the start
    # state with the post-approach pose below (both in the SDK zero frame).
    pre_pos, pre_quat = get_arm_relative_pose(robot, arm=arm)
    if pre_pos is not None:
        logger.info(
            f"[E] Pre-approach arm ({arm}) pose: pos={list(pre_pos)}, "
            f"euler_xyz(deg)={R.from_quat(pre_quat).as_euler('xyz', degrees=True).tolist()}, "
            f"commanded relative target_pos={target_pos.tolist()}"
        )

    logger.info(f"[E] Moving arm ({arm}) to grasp target smoothly...")
    pre_grasp_xyz = move_arm_to_grasp(robot, target_pos, target_quat, arm=arm)

    # Debug: read the arm pose right after approaching to verify the robot
    # actually reached the commanded target. The pose is in the SDK zero frame
    # (same frame as pre_grasp_xyz / target_abs), while target_pos is in the
    # arm-relative frame -- do not compare them directly.
    cur_pos, cur_quat = get_arm_relative_pose(robot, arm=arm)
    if cur_pos is not None:
        logger.info(
            f"[E] Post-approach arm ({arm}) pose: pos={list(cur_pos)}, "
            f"euler_xyz(deg)={R.from_quat(cur_quat).as_euler('xyz', degrees=True).tolist()}, "
            f"pre_grasp_xyz={pre_grasp_xyz}"
        )

    logger.info(f"[F] Closing gripper ({gripper_motor}) to grasp...")
    pdb.set_trace()
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(gripper_motor, close_cmd)
    time.sleep(2.0)

    # Retract to the pre-grasp point (absolute SDK frame), resetting the hand
    # orientation to zero (identity quat) on the way.
    logger.info("[G] Recovering to pre-grasp point, orientation zero...")
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, arm, arm_pos_rel, arm_quat_rel,
                      pre_grasp_xyz, [0, 0, 0, 1], 1)
    time.sleep(1.0)

    logger.info("[G] Lifting arm after grasp...")
    move_arm_relative(robot, -0.2, 0, 0.05, arm=arm)
    time.sleep(1.0)

    # Waypoints: Y is positive for left arm, negated for right arm.
    from lib_h1_sdk_python import ArmAction
    y_sign = 1.0 if arm == ArmAction.LEFT_ARM else -1.0
    lift_mid = [0.0, 0.30 * y_sign, 0.30]
    place_pos = [0.17, 0.30 * y_sign, 0.30]
    retract_ready = [-0.1, 0.0, 0.30]

    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, arm, arm_pos_rel, arm_quat_rel, lift_mid, [0, 0, 0, 1], 1)

    # Move to placement position.
    logger.info("[H] Moving arm to placement position...")
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, arm, arm_pos_rel, arm_quat_rel, place_pos, [0, 0, 0, 1], 1)

    # Drop: lower the waist before releasing.
    logger.info("[Waist] Lowering waist before release...")
    move_waist_z(robot, WAIST_NORMAL_Z, WAIST_RELEASE_Z)

    close_cmd.Position = 0.0
    robot.setGripper_high(gripper_motor, close_cmd)
    time.sleep(GRIPPER_RELEASE_WAIT)

    # Restore the waist before retracting the arm.
    logger.info("[Waist] Restoring waist after release...")
    move_waist_z(robot, WAIST_RELEASE_Z, WAIST_NORMAL_Z)

    # Retract to a safe waypoint.
    logger.info("[I] Retracting arm to safe waypoint...")
    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, arm, arm_pos_rel, arm_quat_rel, lift_mid, [0, 0, 0, 1], 1)
    time.sleep(1.0)

    _, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm_to_pose(robot, arm, arm_pos_rel, arm_quat_rel, retract_ready, [0, 0, 0, 1], 1)
    time.sleep(1.0)
