"""System control parameters and shared constants for the grasp pipeline."""

import os

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

# Perception client config
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
ZMQ_SERVER_ADDR = "tcp://192.168.3.28:5555"
CLIENT_DEBUG_DIR = "./client_debug"
REGISTER_ITERATIONS = 5
RETRY_COUNT = 3600

# Single-arm operation: left arm only.
TARGET_ARM = ArmAction.LEFT_ARM
TARGET_GRIPPER_MOTOR = EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8

K_COLOR = np.array([
    [607.62, 0.00, 329.68],
    [0.00, 608.40, 243.36],
    [0.00, 0.00, 1.00],
], dtype=np.float64)


# ================= Simulation mode config =================
# Recorded scenes for the simulated perception path. Each subdirectory
# (00, 01, ...) holds rgb.png, depth.npy, seg.png, meta_data.json.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
REAL_SCENE_DIR = os.path.join(_PROJECT_ROOT, "assets", "zerith", "real_scene")
