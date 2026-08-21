"""Top-level orchestration of the end-to-end grasp pipeline."""

import time

from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import H1Robot, MotorControlMode

from .config import RETRY_COUNT
from .robot_motion import move_chassis, prepare_robot_posture, move_arm_to_ready_pose
# from .perception import capture_rgbd_data, run_perception_client
from .grasp_executor import select_arm, grasp_object
from .logging_utils import get_logger

logger = get_logger(__name__)


def approach_workspace(robot, move_chassis=True):
    """Drive the chassis to the workspace and bring the arm to the ready pose.

    When ``move_chassis`` is False, the chassis is assumed to already be at the
    workspace; only the posture and arm ready pose are applied.
    """
    if move_chassis:
        move_chassis(robot, 0.8)
        time.sleep(1.0)

    prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
    move_arm_to_ready_pose(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                [-0.1, 0.0, 0.30], [0.0, 0.0, 0.0, 1.0])

    if move_chassis:
        move_chassis(robot, 0.3)
        time.sleep(2.0)


def grasp_detected_objects(robot, pose_files):
    """Grasp every detected object returned by the perception client."""
    for (category_id, instance_index), pose_path in pose_files.items():
        logger.info(f"[Main] Object pose: {category_id}_{instance_index}")
        logger.info(pose_path)
        target_pos, angle = select_arm(robot, pose_path)
        if target_pos is None:
            logger.warning(" -> Target pose resolution failed; skipping.")
            continue
        grasp_quat = R.from_euler("xyz", [angle, 0, 0], degrees=True).as_quat()
        grasp_object(robot, target_pos, grasp_quat)
        time.sleep(1.0)


def run_grasp_attempts(robot):
    """Repeat the perceive-then-grasp cycle for RETRY_COUNT attempts."""
    for attempt in range(RETRY_COUNT):
        logger.info(f"===== Grasp attempt {attempt}/{RETRY_COUNT} =====")
        # Capture RGB-D.
        rgb_path, depth_path = capture_rgbd_data()

        # Run perception for all detected objects.
        pose_files = run_perception_client(rgb_path, depth_path)

        # Grasp each detected object sequentially with the single arm.
        grasp_detected_objects(robot, pose_files)
        logger.info(f"Grasp flow completed on attempt {attempt}.")


def main(args=None):
    logger.info(f"[Main] Args: {args}")
    move_chassis = getattr(args, "move_chassis", True)
    robot = H1Robot()
    try:
        logger.info("[INIT] Instantiating robot and connecting...")
        if not robot.robot_connect():
            logger.info("Failed to connect to robot!")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()

        approach_workspace(robot, move_chassis=move_chassis)
        # run_grasp_attempts(robot)

        # while True:
        #     time.sleep(1.0)

    except KeyboardInterrupt:
        logger.warning("Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        logger.error(f"Runtime exception: {e}")

    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
