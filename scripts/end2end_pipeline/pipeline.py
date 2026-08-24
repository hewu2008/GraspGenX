"""Top-level orchestration of the end-to-end grasp pipeline."""

import time

from lib_h1_sdk_python import H1Robot, MotorControlMode

from .robot_motion import move_chassis, prepare_robot_posture, move_arm_to_ready_pose
from .perception import acquire_rgbd, detect_and_segment, write_meta_data, generate_and_save_grasps
from .grasp_visualization import visualize_saved_grasps
from .logging_utils import get_logger

logger = get_logger(__name__)


def approach_workspace(robot, drive_chassis=True):
    """Drive the chassis to the workspace and bring the arm to the ready pose.

    When ``drive_chassis`` is False, the chassis is assumed to already be at the
    workspace; only the posture and arm ready pose are applied.
    """
    if drive_chassis:
        move_chassis(robot, 0.8)
        time.sleep(1.0)

    prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
    move_arm_to_ready_pose(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                [-0.1, 0.0, 0.30], [0.0, 0.0, 0.0, 1.0])

    if drive_chassis:
        move_chassis(robot, 0.3)
        time.sleep(2.0)


def main(args=None):
    logger.info(f"[Main] Args: {args}")
    mode = getattr(args, "mode", "real")
    scene_dir = getattr(args, "scene_dir", None)
    yolo_model = getattr(args, "yolo_model", None)
    visualize = getattr(args, "visualize", True)

    if not scene_dir:
        logger.error("[Main] --scene-dir is required (e.g. assets/zerith/real_scene/00).")
        return

    # Real mode: drive the robot, capture RGB-D into the scene dir, then detect.
    drive_chassis = getattr(args, "move_chassis", True)
    robot = H1Robot()
    try:
        logger.info("[INIT] Instantiating robot and connecting...")
        if not robot.robot_connect():
            logger.info("Failed to connect to robot!")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()

        approach_workspace(robot, drive_chassis=drive_chassis)

        rgb, depth = acquire_rgbd(scene_dir, mode=mode)
        if depth is not None:
            logger.info(f"[Main] depth: shape={depth.shape} dtype={depth.dtype}")
        detections = detect_and_segment(rgb, yolo_model, scene_dir)
        write_meta_data(scene_dir, robot, len(detections))
        summary, viz_data = generate_and_save_grasps(scene_dir)
        
        if visualize and viz_data:
            logger.info("[Main] Generating grasp visualizations...")
            visualize_saved_grasps(scene_dir, viz_data=viz_data)
        elif not visualize:
            logger.info("[Main] Visualization disabled by --no-visualize flag.")
        else:
            logger.warning("[Main] No viz_data returned; skipping visualization.")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        logger.error(f"Runtime exception: {e}")

    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
