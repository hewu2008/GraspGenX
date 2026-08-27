"""Camera-to-world pose for the ZR H1 head camera.

Single source of truth for the forward-kinematics chain that turns IMU +
waist/head motor readings into the 4x4 camera-to-world transform written to
``meta_data.json``.

Chain (URDF origins from assets/zerith/urdf/ZR_H1PRO-1.2.00.H.V4.3_URDF_2025.12.02.urdf):

    dipan_link [chassis, IMU]
      -> daogui_link     [lift, prismatic Z]
      -> body_pitch_link [body_pitch: R_x(-q_bp)]
      -> body_yaw_link   [body_yaw:   R_z(q_by)]
      -> neck_yaw_link   [neck_yaw:   R_z(q_ny)]
      -> neck_pitch_link [neck_pitch: R_y(q_np)]
      -> neck_camera_link [neck_camera: fixed R_nc]

Two on-robot calibration corrections (do NOT "fix" these back to the URDF):
  - body_pitch physically rotates around X with the opposite sign of its URDF
    label (URDF axis is Y), so its contribution is R_x(-q_bp), not R_y(+q_bp).
  - O_bp.x = 0.1478 is empirical; the URDF lists 0.1518 but matching the
    printed camera pose requires 0.1478 (~4 mm forward of the URDF value).

The IMU is mounted on the chassis (below the waist) and does NOT capture the
waist pitch/yaw joints, which is why they are read from the motors and applied
explicitly on top of R_chassis.
"""

from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import EtherCAT_Motor_Index

PoseInputs = namedtuple(
    "PoseInputs",
    ["imu", "lift", "body_pitch", "body_yaw", "neck_yaw", "neck_pitch"],
)

# URDF origins (O_bp.x is the empirical correction noted above).
_O_BP = np.array([0.1478, 0, 0.1275])
_O_BY = np.array([2.71e-5, -1.21e-4, 0.1572])
_O_NY = np.array([-2.71e-5, 1.21e-4, 0.2491])
_O_NP = np.array([0, 0, 0.11])
_O_NC = np.array([0.0675568573382885, 0.0324999999999979, -0.0363332072227294])
_R_NC = R.from_euler("XYZ", [-1.78023593389281, 0, -1.5707963267949], degrees=False).as_matrix()


def read_pose_inputs(robot):
    """Read IMU + lift/waist/head motor states. Returns PoseInputs or None."""
    ok_imu, imu = robot.getIMU_State()
    if not ok_imu:
        return None

    ok_lift, info_lift = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_LIFT)
    ok_bp, info_bp = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_WAIST_DOWN)
    ok_by, info_by = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_WAIST_UP)
    ok_ny, info_ny = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_HEAD_DOWN)
    ok_np, info_np = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_HEAD_UP)
    if not (ok_lift and ok_bp and ok_by and ok_ny and ok_np):
        return None

    return PoseInputs(imu, info_lift, info_bp, info_by, info_ny, info_np)


def assemble_camera_pose(inputs):
    """Build the 4x4 camera-to-world transform from PoseInputs."""
    # IMU quat is [w, x, y, z]; scipy expects [x, y, z, w].
    w, x, y, z = inputs.imu.quat
    R_chassis = R.from_quat([x, y, z, w]).as_matrix()

    q_lift = inputs.lift.Position_Actual
    q_bp = inputs.body_pitch.Position_Actual
    q_by = inputs.body_yaw.Position_Actual
    q_ny = inputs.neck_yaw.Position_Actual
    q_np = inputs.neck_pitch.Position_Actual

    R_body_pitch = R.from_euler("X", -q_bp, degrees=False).as_matrix()
    R_body_yaw = R.from_euler("Z", q_by, degrees=False).as_matrix()
    R_body = R_chassis @ R_body_pitch @ R_body_yaw  # body_yaw_link in world

    R_ny = R.from_euler("Z", q_ny, degrees=False).as_matrix()
    R_np = R.from_euler("Y", q_np, degrees=False).as_matrix()

    t_lift = np.array([0.0, 0.0, q_lift])
    t_body_yaw = t_lift + R_chassis @ (_O_BP + R_body_pitch @ R_body_yaw @ _O_BY)

    t_neck = _O_NY + R_ny @ (_O_NP + R_np @ _O_NC)
    R_neck = R_ny @ R_np @ _R_NC

    t_camera = t_body_yaw + R_body @ t_neck
    R_camera = R_body @ R_neck

    T = np.eye(4)
    T[:3, :3] = R_camera
    T[:3, 3] = t_camera
    return T


def compute_camera_pose(robot):
    """Read robot state and return the 4x4 camera-to-world transform, or None."""
    inputs = read_pose_inputs(robot)
    if inputs is None:
        return None
    return assemble_camera_pose(inputs)


def compute_hand_camera_pose(robot):
    """Compute the 4x4 left-hand (wrist) camera-to-world transform independently.

    Chain (direct from chassis, independent of head camera/neck motors):
        T_world_chassis     = [R_chassis (IMU), t_lift]
        T_chassis_arm_mount = inv(T3)  (left-arm chassis mounting offset)
        T_arm_mount_eef     = arm_pose_rel (from getHandRelative)
        T_world_hand_cam    = T_world_chassis @ T_chassis_arm_mount @ T_arm_mount_eef @ CAM_TO_SDK_EEF_HAND
    """
    from .config import TARGET_ARM
    from .grasp_executor import CAM_TO_SDK_EEF_HAND

    ok_imu, imu = robot.getIMU_State()
    ok_lift, info_lift = robot.getMotorState(EtherCAT_Motor_Index.MOTOR_LIFT)
    ok_arm, arm_state = robot.getHandRelative(TARGET_ARM)
    if not (ok_imu and ok_lift and ok_arm):
        return None

    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    if arm_pos_rel is None or arm_quat_rel is None:
        return None

    w, x, y, z = imu.quat
    R_chassis = R.from_quat([x, y, z, w]).as_matrix()
    q_lift = info_lift.Position_Actual

    T_world_chassis = np.eye(4)
    T_world_chassis[:3, :3] = R_chassis
    T_world_chassis[:3, 3] = [0.0, 0.0, q_lift]

    # Chassis -> left arm mount frame (inverse of T3 in grasp_executor)
    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]
    T_chassis_arm_mount = np.linalg.inv(T3)

    # Arm mount -> current SDK EEF
    T_arm_mount_eef = np.eye(4)
    T_arm_mount_eef[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T_arm_mount_eef[:3, 3] = arm_pos_rel

    T_world_eef = T_world_chassis @ T_chassis_arm_mount @ T_arm_mount_eef
    T_world_hand_cam = T_world_eef @ CAM_TO_SDK_EEF_HAND
    return T_world_hand_cam
