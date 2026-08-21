#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move the H1 robot to its initial observation pose.

Extracted from scripts/integrated_grasp_pipeline_v16.py. Keeps only the
"reach the initial pose" sequence (waist adjust -> dual-arm pre-position)
and drops all perception, grasp, chassis motion, and post-grasp recovery
motions.

Usage
-----

    # default initial pose, hold after arrival
    python scripts/move_to_initial_pose.py

    # custom waist / arm targets
    python scripts/move_to_initial_pose.py \\
        --waist_z 0.67 --waist_pitch 1.2 \\
        --arm_target -0.1 0.0 0.30 --no-hold
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Local SDK bundles shipped under assets/zerith/sdk/
#   lib/   -> lib_h1_sdk_python.so, camera_client.cpython-310-x86_64-linux-gnu.so
#   proto/ -> robot_pb2.py, robot_pb2_grpc.py
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_sdk_dir = os.path.join(root, "assets", "zerith", "sdk")
for _sub in ("lib", "proto"):
    _p = os.path.join(_sdk_dir, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from lib_h1_sdk_python import (
    H1Robot,
    MotorControlMode,
    ArmAction,
    ArmPose,
    ArmEndPose,
)

from end2end_pipeline.camera_pose import read_pose_inputs, assemble_camera_pose

RATE_HZ = 500             # Low-level control rate 500Hz
DT = 1.0 / RATE_HZ        # Control period 0.002s


def parse_args():
    p = argparse.ArgumentParser(description="Move H1 robot to its initial observation pose.")
    p.add_argument("--waist_z", type=float, default=0.67, help="Target waist Z (m).")
    p.add_argument("--waist_pitch", type=float, default=1.2, help="Target waist pitch (rad).")
    p.add_argument("--arm_target", type=float, nargs=3, default=[-0.1, 0.0, 0.30],
                   help="Target arm end position [x, y, z] (m). Right arm mirrors y.")
    p.add_argument("--arm_quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0],
                   help="Target arm end rotation quaternion [qx, qy, qz, qw].")
    p.add_argument("--arm_origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="Arm motion start position [x, y, z] (m).")
    p.add_argument("--arm_origin_quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0],
                   help="Arm motion start rotation [qx, qy, qz, qw].")
    p.add_argument("--hold", dest="hold", action=argparse.BooleanOptionalAction, default=True,
                   help="Hold (idle) after reaching initial pose until Ctrl+C (default on).")
    return p.parse_args()


def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[Step A] Adjusting robot to initial observation pose...")
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


def _move_arm(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
              duration=3.0, rate=RATE_HZ, dt=DT):
    """Smooth arm interpolation, called from a worker thread."""
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


def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[Step E] Moving both arms from relative zero to target simultaneously...")

    left_dest = [dest_xyz[0], dest_xyz[1], dest_xyz[2]]
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]   # mirror y

    t_left = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.LEFT_ARM, cur_xyz, cur_quat,
              left_dest, dest_quat, 2)
    )
    t_right = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.RIGHT_ARM, cur_xyz, cur_quat,
              right_dest, dest_quat, 2)
    )

    t_left.start()
    t_right.start()

    t_left.join()
    t_right.join()

    time.sleep(0.5)
    print(" -> Both arms reached their targets smoothly!")


def print_camera_pose(robot):
    """Print the camera-to-world 4x4 transform for the current robot state.

    The IMU + motor reads and the URDF forward-kinematics chain live in
    end2end_pipeline.camera_pose (single source of truth); this function just
    surfaces the intermediate values and the resulting matrix for on-robot
    calibration debugging.
    """
    inputs = read_pose_inputs(robot)
    if inputs is None:
        print("[camera_pose] Failed to read IMU / motor states.")
        return None

    imu = inputs.imu
    print("\n[Camera Pose] IMU (base orientation):")
    print(f"  rpy  : {np.array2string(np.asarray(imu.rpy), precision=6, separator=', ')}")
    print(f"  quat : {np.array2string(np.asarray(imu.quat), precision=6, separator=', ')}  [w, x, y, z]")

    q_lift = inputs.lift.Position_Actual
    q_bp = inputs.body_pitch.Position_Actual
    q_by = inputs.body_yaw.Position_Actual
    q_ny = inputs.neck_yaw.Position_Actual
    q_np = inputs.neck_pitch.Position_Actual
    print("\n[Camera Pose] Motors:")
    print(f"  MOTOR_LIFT        -> lift:        {q_lift:.6f}")
    print(f"  MOTOR_WAIST_DOWN  -> body_pitch:  {q_bp:.6f}")
    print(f"  MOTOR_WAIST_UP    -> body_yaw:    {q_by:.6f}")
    print(f"  MOTOR_HEAD_DOWN   -> neck_yaw:    {q_ny:.6f}")
    print(f"  MOTOR_HEAD_UP     -> neck_pitch:  {q_np:.6f}")

    T = assemble_camera_pose(inputs)

    print("\n[Camera Pose] camera-to-world transform:")
    print("  4x4 matrix:")
    for row in T:
        print("    [" + ", ".join(f"{v:12.6f}" for v in row) + "]")
    print("  JSON (for meta_data.json \"camera_pose\"):")
    json_rows = []
    for row in T:
        json_rows.append("      [" + ", ".join(repr(v) for v in row) + "]")
    print("    [\n" + ",\n".join(json_rows) + "\n    ]")
    return T


def main():
    args = parse_args()
    robot = H1Robot()
    try:
        print("[INIT] Instantiating robot and connecting...")
        if not robot.robot_connect():
            print("Robot connection failed!")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()

        prepare_robot_posture(robot, 0, 0, args.waist_z, args.waist_pitch)
        arm_move_pre(
            robot,
            args.arm_origin, args.arm_origin_quat,
            args.arm_target, args.arm_quat,
        )

        print("\n[Done] Reached initial observation pose.")
        print_camera_pose(robot)
        if args.hold:
            print("[HOLD] Entering hold mode, press Ctrl+C to exit...")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C interrupt received, preparing safe shutdown...")
    except Exception as e:
        print(f"\n[!] Runtime exception: {e}")
    finally:
        print("[Cleanup] Releasing robot control...")
        if "robot" in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()


if __name__ == "__main__":
    main()
