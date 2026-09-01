"""System control parameters and shared constants for the grasp pipeline."""

import numpy as np

from lib_h1_sdk_python import ArmAction, EtherCAT_Motor_Index

# ================= System control parameters =================
RATE_HZ = 500              # Low-level control rate 500 Hz
DT = 1.0 / RATE_HZ         # Control period 0.002 s

# Waist placement: normal height 0.67 m, drops to 0.49 m before releasing.
WAIST_NORMAL_Z = 0.67
WAIST_RELEASE_Z = WAIST_NORMAL_Z - 0.18
WAIST_PITCH = 1.2
WAIST_MOVE_DURATION = 2.0
GRIPPER_RELEASE_WAIT = 2.0  # Wait for the gripper to fully open before restoring waist.

# Stage-3 approach speed limits. The straight-line approach ("stage 3") used to
# run over a FIXED 1s duration, so the commanded end-effector line/angular speed
# scaled with the approach distance+rotation and exceeded the SDK tracking
# capability, leaving the arm ~2-7cm short (measured in sdk_accuracy exp, +X
# undershoot). Duration is now computed adaptively from the actual motion while
# keeping speeds below these limits.
APPROACH_MAX_TRANS_SPEED_MPS = 0.1   # max end-effector translation speed (m/s)
APPROACH_MAX_ANG_SPEED_RPS = 0.3     # max end-effector rotation speed (rad/s)
APPROACH_MIN_DURATION_S = 1.5        # floor on the approach duration (s)

# Arrival-convergence tolerances (sdk_accuracy exp run 5/6: interpolation-based
# moves with adaptive duration still undershot by ~6cm / ~13deg, so arrival is
# now verified by polling getHandRelative until the residual error is below
# these tolerances or the timeout elapses).
APPROACH_ARRIVE_POS_TOL_M = 0.02     # arrival position tolerance (m)
APPROACH_ARRIVE_ANG_TOL_DEG = 8.0    # arrival orientation tolerance (deg)
APPROACH_ARRIVE_TIMEOUT_EXTRA_S = 2.0  # extra seconds beyond expected duration

# Fixed transform from the gripper base (wrist_pitch_link) to the SDK
# Cartesian end-effector frame (end_effector_link).  This comes from
# left/right_end_effector_joint in the Zerith URDF.  GraspGenX poses are anchored at
# the gripper base, while setArm_high() commands the arm end-effector pose.
WRIST_TO_SDK_EEF_OFFSET_M = np.array([0.1435, 0.0, 0.0], dtype=np.float64)

# Optional fine-tuning grasp translation offset [dx, dy, dz] in meters (gripper local frame: +X=depth/approach, +Y=closing, +Z=normal).
GRASP_TRIM_OFFSET_M = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Perception client config
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"

# Left-hand (wrist) camera config
LEFT_HAND_CAMERA_NAME = "rs/cam_left_wrist"
LEFT_HAND_CAM_SUFFIX = "_cam_left_wrist"

# Right-hand (wrist) camera config
RIGHT_HAND_CAMERA_NAME = "rs/cam_right_wrist"
RIGHT_HAND_CAM_SUFFIX = "_cam_right_wrist"

# Backward compatibility aliases
HAND_CAMERA_NAME = LEFT_HAND_CAMERA_NAME
HAND_CAM_SUFFIX = LEFT_HAND_CAM_SUFFIX

ZMQ_SERVER_ADDR = "tcp://192.168.3.28:5555"
CLIENT_DEBUG_DIR = "./client_debug"
REGISTER_ITERATIONS = 5
RETRY_COUNT = 3600

# Arm and gripper definitions
LEFT_ARM = ArmAction.LEFT_ARM
RIGHT_ARM = ArmAction.RIGHT_ARM
LEFT_GRIPPER_MOTOR = EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8
RIGHT_GRIPPER_MOTOR = EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8
LEFT_GRIPPER_NAME = "zerith_left_gripper"
RIGHT_GRIPPER_NAME = "zerith_right_gripper"

# Target object classes per arm/camera
LEFT_TARGET_CLASSES = {"elbow_pipe"}
RIGHT_TARGET_CLASSES = {"interior_door_handle"}

# Viser visualization ports per hand
LEFT_VIZ_PORT = 8080
RIGHT_VIZ_PORT = 8081

# Backward compatibility defaults
TARGET_ARM = LEFT_ARM
TARGET_GRIPPER_MOTOR = LEFT_GRIPPER_MOTOR

K_COLOR = np.array([
    [607.62, 0.00, 329.68],
    [0.00, 608.40, 243.36],
    [0.00, 0.00, 1.00],
], dtype=np.float64)

# Left-hand wrist camera intrinsics (pinhole K), from the camera service's live
# intrinsics (camera_intrinsics.yaml, rs/cam_left_wrist color stream).
K_LEFT_HAND_COLOR = np.array([
    [394.747, 0.00, 322.405],
    [0.00, 394.532, 238.677],
    [0.00, 0.00, 1.00],
], dtype=np.float64)

# Right-hand wrist camera intrinsics (pinhole K), from the camera service's live
# intrinsics (camera_intrinsics.yaml, rs/cam_right_wrist color stream).
K_RIGHT_HAND_COLOR = np.array([
    [394.107, 0.00, 320.601],
    [0.00, 393.723, 242.848],
    [0.00, 0.00, 1.00],
], dtype=np.float64)

# Backward compatibility alias
K_HAND_COLOR = K_LEFT_HAND_COLOR

# World-frame workspace bounds [xmin, ymin, zmin, xmax, ymax, zmax] (m),
# written to meta_data.json for grasp-generation scene cropping.
# Expanded to include the full visible scene for both grasp planning and visualization.
SCENE_BOUNDS = [-1.0, -1.0, -0.3, 1.5, 1.0, 1.5]
