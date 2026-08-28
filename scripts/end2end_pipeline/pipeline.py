"""Top-level orchestration of the end-to-end grasp pipeline."""

import random
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import H1Robot, MotorControlMode, ArmAction

from .robot_motion import move_chassis, prepare_robot_posture, move_arm_to_ready_pose
from .perception import (
    acquire_rgbd,
    detect_and_segment,
    write_meta_data,
    generate_and_save_grasps,
)
from .grasp_visualization import visualize_saved_grasps
from .grasp_executor import (
    resolve_grasp_target_hand,
    grasp_object,
)
from .camera_pose import compute_hand_camera_pose
from .config import (
    LEFT_HAND_CAMERA_NAME,
    LEFT_HAND_CAM_SUFFIX,
    RIGHT_HAND_CAMERA_NAME,
    RIGHT_HAND_CAM_SUFFIX,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_GRIPPER_MOTOR,
    RIGHT_GRIPPER_MOTOR,
    LEFT_GRIPPER_NAME,
    RIGHT_GRIPPER_NAME,
    LEFT_TARGET_CLASSES,
    RIGHT_TARGET_CLASSES,
    LEFT_VIZ_PORT,
    RIGHT_VIZ_PORT,
    K_LEFT_HAND_COLOR,
    K_RIGHT_HAND_COLOR,
)
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


def execute_grasp_all_objects_wrist(
    robot,
    scene_dir,
    viz_data,
    gripper_name=LEFT_GRIPPER_NAME,
    arm=LEFT_ARM,
    gripper_motor=LEFT_GRIPPER_MOTOR,
    dry_run=False,
    target_description="object",
):
    """Grasp and place every detected object in sequence using the wrist camera
    result and the fixed-URDF-offset hand-eye transform.

    Args:
        robot: connected H1Robot instance (or None in sim mode).
        scene_dir: path to the wrist camera scene directory (contains meta_data.json).
        viz_data: visualization/grasp data returned by generate_and_save_grasps.
        gripper_name: 'zerith_left_gripper' or 'zerith_right_gripper'.
        arm: ArmAction.LEFT_ARM or ArmAction.RIGHT_ARM.
        gripper_motor: EtherCAT motor index for gripper.
        dry_run: if True, only resolve and log targets without moving robot.
        target_description: descriptive label for logging.
    """
    import json
    from pathlib import Path

    meta_path = Path(scene_dir) / "meta_data.json"
    if not meta_path.exists():
        logger.error(f"[Grasp] meta_data.json not found: {meta_path}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    cam_pose = np.asarray(meta.get("camera_pose"), dtype=np.float64)
    if cam_pose.size != 16:
        logger.error("[Grasp] camera_pose missing/invalid in meta_data.json")
        return
    cam_pose = cam_pose.reshape(4, 4)
    cam_pose_inv = np.linalg.inv(cam_pose)  # world frame -> camera frame

    grippers = viz_data.get("grippers", {})
    if gripper_name not in grippers:
        logger.error(f"[Grasp] gripper {gripper_name!r} not found in viz_data")
        return
    grasps_data = grippers[gripper_name].get("grasps", {})
    if not grasps_data:
        logger.info(f"[Grasp] No grasps to execute for gripper {gripper_name} ({target_description}).")
        return

    labels = list(grasps_data.keys())
    logger.info(f"[Grasp] Picking {len(labels)} {target_description}(s) in order: {labels} using {gripper_name}")
    for obj_label in labels:
        data = grasps_data[obj_label]
        grasps = data.get("grasps")
        conf = data.get("conf")
        if grasps is None or len(grasps) == 0:
            logger.warning(f"[Grasp] {obj_label}: no grasps, skipping")
            continue

        best_idx = int(np.argmax(conf))
        T_world = np.asarray(grasps[best_idx], dtype=np.float64)  # grasp pose (world)
        T_cam = cam_pose_inv @ T_world  # world -> camera frame

        logger.info(
            f"[Grasp] {obj_label}: pre(world)  pos={T_world[:3, 3].tolist()}, "
            f"euler_xyz(deg)={R.from_matrix(T_world[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
        )
        logger.info(
            f"[Grasp] {obj_label}: post(camera) pos={T_cam[:3, 3].tolist()}, "
            f"euler_xyz(deg)={R.from_matrix(T_cam[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
        )
        logger.info(f"[Grasp] {obj_label}: best conf={conf[best_idx]:.3f}")

        target_pos, target_quat = resolve_grasp_target_hand(robot, T_cam)
        if target_pos is None:
            logger.error(f"[Grasp] {obj_label}: failed to resolve target, skipping")
            continue

        _target_euler = R.from_quat(target_quat).as_euler("xyz", degrees=True)
        target_dist = float(np.linalg.norm(target_pos))
        logger.info(
            f"[Grasp] {obj_label}: target_pos={target_pos.tolist()}, "
            f"target_quat={target_quat.tolist()}, "
            f"target_euler_xyz(deg)={_target_euler.tolist()}, "
            f"target_dist={target_dist:.4f}"
        )

        if not dry_run:
            grasp_object(robot, target_pos, target_quat, arm=arm, gripper_motor=gripper_motor)
        logger.info(f"[Grasp] {obj_label} ({target_description}): grasped & placed.")

    logger.info(f"[Grasp] All {target_description}(s) grasped & placed.")

def connect_robot(mode, drive_chassis):
    """Connect, initialize and move the robot to the observation posture.

    Returns ``(robot, ok)``:
      - sim mode: (None, True) -- no robot is touched, scenes are processed offline.
      - real mode: (H1Robot, True) on success; (H1Robot, False) if connection failed
        (the instance is returned so the caller's cleanup can still deinit it).
    """
    if mode == "sim":
        logger.info("[Main] Sim mode: robot is NOT connected or operated; "
                    "processing the saved scenes offline.")
        return None, True

    robot = H1Robot()
    logger.info("[INIT] Instantiating robot and connecting...")
    if not robot.robot_connect():
        logger.info("Failed to connect to robot!")
        return robot, False

    robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
    robot.robot_init()

    approach_workspace(robot, drive_chassis=drive_chassis)
    logger.info("[Main] Workspace reached, please reset the environment!")
    
    import pdb; pdb.set_trace()
    return robot, True


def run_wrist_camera_pipeline(
    robot, mode, yolo_model, scene_dir, *,
    suffix, camera_name, allowed_classes, k_color, gripper_name, arm,
    target_description,
):
    """One wrist-camera arm pipeline: capture, detect, write meta, generate grasps.

    Returns ``(scene, viz_data)`` where ``scene = scene_dir + suffix``.
    """
    scene = scene_dir + suffix
    logger.info(f"[Main] === {target_description} ===")
    logger.info(f"[Main] Capturing {camera_name} scene into {scene} ...")
    rgb, depth = acquire_rgbd(scene, mode=mode, camera_name=camera_name)
    if depth is not None:
        logger.info(f"[Main] {camera_name} depth: shape={depth.shape} dtype={depth.dtype}")

    detections = detect_and_segment(rgb, yolo_model, scene, allowed_classes=allowed_classes)

    if mode == "real":
        cam_pose = compute_hand_camera_pose(robot, arm=arm)
        write_meta_data(
            scene, robot, len(detections),
            camera_pose=cam_pose, intrinsics=k_color,
        )
    else:
        logger.info(f"[Main] Sim mode: keeping existing {scene} meta_data.json.")

    _, viz_data = generate_and_save_grasps(scene, gripper_names=[gripper_name])
    return scene, viz_data


def _has_grasps(viz_data, gripper_name):
    """True when ``viz_data`` holds at least one grasp for ``gripper_name``."""
    return bool(
        viz_data
        and viz_data.get("grippers", {}).get(gripper_name, {}).get("grasps")
    )


def run_visualization(visualize, left_scene, viz_data_left, right_scene, viz_data_right):
    """Launch the (blocking) viser visualization for each scene that has grasps."""
    if not visualize:
        logger.info("[Main] Visualization disabled by --no-visualize flag.")
        return

    if _has_grasps(viz_data_left, LEFT_GRIPPER_NAME):
        logger.info(
            f"[Main] Running left-hand camera visualization on port {LEFT_VIZ_PORT} for {left_scene}."
        )
        visualize_saved_grasps(left_scene, viz_data=viz_data_left, port=LEFT_VIZ_PORT)

    if _has_grasps(viz_data_right, RIGHT_GRIPPER_NAME):
        logger.info(
            f"[Main] Running right-hand camera visualization on port {RIGHT_VIZ_PORT} for {right_scene}."
        )
        visualize_saved_grasps(right_scene, viz_data=viz_data_right, port=RIGHT_VIZ_PORT)

    if not _has_grasps(viz_data_left, LEFT_GRIPPER_NAME) and not _has_grasps(viz_data_right, RIGHT_GRIPPER_NAME):
        logger.warning("[Main] No viz_data returned for visualization; skipping.")


def execute_wrist_grasps(robot, mode, left_scene, viz_data_left, right_scene, viz_data_right):
    """Execute the grasp+place cycles for both arms from their wrist camera results."""
    if _has_grasps(viz_data_left, LEFT_GRIPPER_NAME):
        logger.info(f"[Main] Executing left-hand grasp(s) for {LEFT_TARGET_CLASSES}...")
        execute_grasp_all_objects_wrist(
            robot=robot,
            scene_dir=left_scene,
            viz_data=viz_data_left,
            gripper_name=LEFT_GRIPPER_NAME,
            arm=LEFT_ARM,
            gripper_motor=LEFT_GRIPPER_MOTOR,
            dry_run=(mode != "real"),
            target_description=f"{LEFT_TARGET_CLASSES}",
        )
    else:
        logger.info(f"[Main] No left-hand grasps for {LEFT_TARGET_CLASSES} to execute.")

    if _has_grasps(viz_data_right, RIGHT_GRIPPER_NAME):
        logger.info(f"[Main] Executing right-hand grasp(s) for {RIGHT_TARGET_CLASSES}...")
        execute_grasp_all_objects_wrist(
            robot=robot,
            scene_dir=right_scene,
            viz_data=viz_data_right,
            gripper_name=RIGHT_GRIPPER_NAME,
            arm=RIGHT_ARM,
            gripper_motor=RIGHT_GRIPPER_MOTOR,
            dry_run=(mode != "real"),
            target_description=f"{RIGHT_TARGET_CLASSES}",
        )
    else:
        logger.info(f"[Main] No right-hand grasps for {RIGHT_TARGET_CLASSES} to execute.")


def set_seed(seed: int = 42):
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args=None):
    seed = getattr(args, "seed", 42)
    if seed is not None and seed >= 0:
        set_seed(seed)
        logger.info(f"[Main] Fixed random seed: {seed}")
    logger.info(f"[Main] Args: {args}")
    mode = getattr(args, "mode", "real")
    scene_dir = getattr(args, "scene_dir", None)
    yolo_model = getattr(args, "yolo_model", None)
    visualize = getattr(args, "visualize", True)

    if not scene_dir:
        logger.error("[Main] --scene-dir is required (e.g. assets/zerith/real_scene/00).")
        return

    drive_chassis = getattr(args, "move_chassis", True)
    robot = None
    try:
        robot, ok = connect_robot(mode, drive_chassis)
        if not ok:
            return

        left_scene, viz_data_left = run_wrist_camera_pipeline(
            robot, mode, yolo_model, scene_dir,
            suffix=LEFT_HAND_CAM_SUFFIX,
            camera_name=LEFT_HAND_CAMERA_NAME,
            allowed_classes=LEFT_TARGET_CLASSES,
            k_color=K_LEFT_HAND_COLOR,
            gripper_name=LEFT_GRIPPER_NAME,
            arm=LEFT_ARM,
            target_description=(
                f"Left Camera Pipeline: Target = ({LEFT_TARGET_CLASSES})"
            ),
        )

        right_scene, viz_data_right = run_wrist_camera_pipeline(
            robot, mode, yolo_model, scene_dir,
            suffix=RIGHT_HAND_CAM_SUFFIX,
            camera_name=RIGHT_HAND_CAMERA_NAME,
            allowed_classes=RIGHT_TARGET_CLASSES,
            k_color=K_RIGHT_HAND_COLOR,
            gripper_name=RIGHT_GRIPPER_NAME,
            arm=RIGHT_ARM,
            target_description=(
                f"Right Camera Pipeline: Target = ({RIGHT_TARGET_CLASSES})"
            ),
        )

        # Optional visualization
        run_visualization(visualize, left_scene, viz_data_left, right_scene, viz_data_right)

        # Grasp & place every detected object for each arm.
        execute_wrist_grasps(robot, mode, left_scene, viz_data_left, right_scene, viz_data_right)

        logger.info("[Main] Pipeline finished successfully.")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        logger.error(f"Runtime exception: {e}")

    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if robot is not None and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
