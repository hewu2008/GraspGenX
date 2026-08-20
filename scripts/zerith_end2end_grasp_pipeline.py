import os
import sys
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# ================= SDK and environment setup =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode, EtherCAT_Motor_Index,
    ArmAction, ArmPose, ArmEndPose, Motor_Control
)
from camera_client import CameraClient
from zerith.zerith_client import (
    create_client,
    detect_parts,
    process_label,
    save_detection_debug,
)

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


# ================= 1. Robot posture preparation =================
def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[A] Adjusting robot to initial observation posture...")
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


def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[E] Moving arm from relative zero to target smoothly...")
    _move_arm(robot, TARGET_ARM, cur_xyz, cur_quat, dest_xyz, dest_quat, 2)
    time.sleep(0.5)
    print(" -> Arm reached target.")


def arm_move_rec(robot, dx, dy, dz):
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
    print(" -> Reached target.")


def arm_move_to_grasp(robot, target_pos, target_quat):
    """Three-stage approach: pre-grasp waypoint, orient, final approach."""
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    # Pre-grasp waypoint: pull back along X, lift along Y.
    temp_xyz = [arm_pos_rel[0] + target_pos[0] - 0.10,
                arm_pos_rel[1] + target_pos[1] + 0.02,
                arm_pos_rel[2] + target_pos[2]]
    temp_quat = [0.0, 0.0, 0.0, 1.0]
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat, 2)
    time.sleep(0.5)

    # Rotate to grasp orientation at the same position.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    temp_xyz = [arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]]
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, target_quat, 1)
    time.sleep(0.5)

    # Final approach toward the object.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    dest_xyz = [arm_pos_rel[0] + 0.05, arm_pos_rel[1], arm_pos_rel[2] - 0.01]
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, dest_xyz, arm_quat_rel, 1)
    time.sleep(0.5)
    print(" -> Reached grasp pose.")


# ================= 2. RGB-D capture =================
def capture_rgbd_data():
    print(f"\n[B] Connecting to camera service for RGB-D ({GRPC_TARGET})...")
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    client.start()

    rgb_path = "zerith_rgb.png"
    depth_path = "zerith_depth.npy"

    try:
        max_retries = 50
        for _ in range(max_retries):
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME)

            if depth_data is not None and color_data is not None:
                depth_raw_mm, _ = depth_data
                color_raw, _ = color_data
                # Convert to float32 meters for the algorithm.
                depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0

                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_raw_m)
                print(f" -> Captured. Saved {rgb_path} and {depth_path}")
                return rgb_path, depth_path

            time.sleep(0.1)
        raise TimeoutError("Image capture timed out. Check the gRPC node.")
    finally:
        client.stop()


# ================= 3. Perception client (all detected objects) =================
def build_pose_path(debug_dir, category_id, instance_index):
    """Path to the per-object pose file saved by zerith_client."""
    return os.path.join(
        debug_dir,
        f"{category_id}_{instance_index}",
        "ob_in_cam",
        "0.txt",
    )


def run_perception_client(rgb_path, depth_path, debug_dir=CLIENT_DEBUG_DIR):
    """Run Detection + Register for all detected objects via zerith_client.

    Each successful object's pose is saved to:
        <debug_dir>/<category_id>_<instance_index>/ob_in_cam/0.txt

    Returns:
        dict mapping (category_id, instance_index) -> pose_path.
    """
    print("\n[C] Requesting Zerith Detection + Register for all objects...")
    os.makedirs(debug_dir, exist_ok=True)

    client = create_client(ZMQ_SERVER_ADDR)
    if client is None:
        print(" -> [ERROR] Failed to connect to Zerith server.")
        return {}

    pose_files = {}

    try:
        # 1. Detection: keep all returned detections.
        color, boxes = detect_parts(client, rgb_path)
        if color is None or boxes is None:
            print(" -> [ERROR] Detection call failed.")
            return {}

        if len(boxes) == 0:
            print(" -> [WARN] No objects detected in current frame.")
            return {}

        save_detection_debug(debug_dir, color, boxes)
        print(f" -> Detection returned {len(boxes)} objects. Starting Register.")

        # 2. Load the same depth data used by zerith_client.main.
        depth = np.load(depth_path)

        # Same category may appear multiple times; index like zerith_client.main.
        category_counts = {}

        for detection_index, box_dict in enumerate(boxes):
            label = box_dict["label"]
            category_id = str(box_dict["category_id"])

            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1

            box = [
                int(box_dict["x1"]),
                int(box_dict["y1"]),
                int(box_dict["x2"]),
                int(box_dict["y2"]),
            ]

            label_output_dir = os.path.join(debug_dir, f"{category_id}_{instance_index}")

            print(f" -> [{detection_index + 1}/{len(boxes)}] "
                  f"processing {category_id}_{instance_index}: {label}")

            # process_label calls client.register and writes the result to
            # <label_output_dir>/ob_in_cam/0.txt.
            success = process_label(
                client=client,
                K=K_COLOR,
                color=color,
                depth=depth,
                label=label,
                category_id=category_id,
                box=box,
                mesh_bbox=None,
                to_origin=None,
                label_output_dir=label_output_dir,
                register_iterations=REGISTER_ITERATIONS,
                show=False,
            )

            if not success:
                print(f" -> [WARN] {category_id}_{instance_index} register failed; continuing.")
                continue

            pose_path = build_pose_path(debug_dir, category_id, instance_index)

            if not os.path.isfile(pose_path):
                print(f" -> [WARN] Register succeeded but pose file missing: {pose_path}")
                continue

            pose_files[(category_id, instance_index)] = pose_path
            print(f" -> Pose saved: {pose_path}")

        print(f"\n -> Successfully obtained {len(pose_files)} object poses:")
        for (category_id, instance_index), pose_path in pose_files.items():
            print(f"    {category_id}_{instance_index}: {pose_path}")

        return pose_files

    except Exception as e:
        print(f" -> [EXCEPTION] Perception client crashed: {e}")
        return {}

    finally:
        client.close()


# ================= 4. Grasp computation and execution =================
def load_pose_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    matrix_lines = [line.strip().split() for line in lines[:4]]
    pose = np.array(matrix_lines, dtype=np.float64).reshape(4, 4)
    angle = float(lines[4].strip())
    return pose, angle


def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam):
    """Coordinate transform: perception pose -> arm-relative target pose."""
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    # Left-arm mounting offset.
    T3 = np.eye(4)
    T3[:3, 3] = [-0.5743, -0.1800, -0.1208]

    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T4 = np.linalg.inv(T4_inv)

    T_obj_in_arm = T4 @ T3 @ T2 @ T1 @ T_obj_cam

    T_grasp_local = np.eye(4)
    T_final = T_obj_in_arm @ T_grasp_local

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()
    return target_pos, target_quat


def chassis_move(robot: H1Robot, dist):
    dt = 0.2
    speed = 0.2
    distance = abs(dist)
    direction = 1 if dist >= 0 else -1
    velocity = speed * direction
    duration = distance / speed

    print(f"[Chassis] Forward: speed={speed} m/s, distance={distance} m, duration={duration:.2f}s")
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            loop_start = time.perf_counter()
            robot.setChassis_high(velocity, 0.0)

            elapsed_time = time.time() - start_time
            remaining_distance = distance - (speed * elapsed_time)
            print(f"[Chassis] elapsed {elapsed_time:.2f}s, remaining {remaining_distance:.2f} m")

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] Forward {distance} m complete")

    except KeyboardInterrupt:
        print("\n[Chassis] Manual stop")
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

    print(f"[Waist] Z: {start_z:.2f} m -> {target_z:.2f} m")
    for i in range(1, steps + 1):
        ratio = i / steps
        waist_pose.z = start_z + (target_z - start_z) * ratio
        waist_end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(waist_end_pose)
        time.sleep(DT)

    time.sleep(0.5)


def select_arm(robot, pose_path):
    """Resolve the target pose for the single (left) arm."""
    print("\n[D] Reading pose state and solving target matrix...")
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None)
    cam_quat_rel = getattr(cam_state, "rotation", None)

    ok_arm, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    if not (ok_cam and ok_arm):
        print("Sensor pose retrieval failed!")
        return None, None

    T_obj_cam, angle = load_pose_matrix(pose_path)
    target_pos, _ = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam
    )
    return target_pos, angle


def grasp_object(robot, target_pos, target_quat):
    """Full single-arm grasp cycle: approach, close, lift, place, release, retract."""
    print(f" -> Target relative translation: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    print("\n[E] Moving arm to grasp target smoothly...")
    arm_move_to_grasp(robot, target_pos, target_quat)

    print("\n[F] Closing gripper to grasp...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(TARGET_GRIPPER_MOTOR, close_cmd)
    time.sleep(2.0)

    # Lift after grasp.
    arm_move_rec(robot, -0.2, 0, 0.05)
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # Move to placement position.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.17, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # Drop: lower the waist before releasing.
    print("[Waist] Lowering waist before release...")
    move_waist_z(robot, WAIST_NORMAL_Z, WAIST_RELEASE_Z)
    time.sleep(1.0)

    close_cmd.Position = 0.0
    robot.setGripper_high(TARGET_GRIPPER_MOTOR, close_cmd)
    time.sleep(GRIPPER_RELEASE_WAIT)

    # Restore the waist before retracting the arm.
    print("[Waist] Restoring waist after release...")
    move_waist_z(robot, WAIST_RELEASE_Z, WAIST_NORMAL_Z)

    # Retract to a safe waypoint.
    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    _, arm_state = robot.getHandRelative(TARGET_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, TARGET_ARM, arm_pos_rel, arm_quat_rel, [-0.1, 0.0, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)


# ================= Main control flow =================
def main():
    robot = H1Robot()
    try:
        print("[INIT] Instantiating robot and connecting...")
        if not robot.robot_connect():
            print("Failed to connect to robot!")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        chassis_move(robot, 0.8)
        time.sleep(1.0)
        prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
        arm_move_pre(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                    [-0.1, 0.0, 0.30], [0.0, 0.0, 0.0, 1.0])
        chassis_move(robot, 0.3)
        time.sleep(2.0)

        for attempt in range(RETRY_COUNT):
            print(f"\n===== Grasp attempt {attempt}/{RETRY_COUNT} =====")
            # Capture RGB-D.
            rgb_path, depth_path = capture_rgbd_data()

            # Run perception for all detected objects.
            pose_files = run_perception_client(rgb_path, depth_path)

            # Grasp each detected object sequentially with the single arm.
            for (category_id, instance_index), pose_path in pose_files.items():
                print(f"\n[Main] Object pose: {category_id}_{instance_index}")
                print(pose_path)
                target_pos, angle = select_arm(robot, pose_path)
                if target_pos is None:
                    print(" -> [WARN] Target pose resolution failed; skipping.")
                    continue
                grasp_quat = R.from_euler("xyz", [angle, 0, 0], degrees=True).as_quat()
                grasp_object(robot, target_pos, grasp_quat)
                time.sleep(1.0)

            print(f"Grasp flow completed on attempt {attempt}.")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        print(f"\n[!] Runtime exception: {e}")

    finally:
        print("[Cleanup] Releasing robot control...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()


if __name__ == "__main__":
    main()
