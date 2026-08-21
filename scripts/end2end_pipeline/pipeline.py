"""Top-level orchestration of the end-to-end grasp pipeline."""

import time

from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import H1Robot, MotorControlMode

from .config import RETRY_COUNT
from .robot_motion import chassis_move, prepare_robot_posture, arm_move_pre
from .perception import capture_rgbd_data, run_perception_client
from .grasp_executor import select_arm, grasp_object


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
