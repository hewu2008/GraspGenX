"""Top-level orchestration of the end-to-end grasp pipeline."""

import time

import numpy as np
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


def execute_grasp_all_objects_hand(robot, scene_dir, viz_data, dry_run=False):
    """Backward compatibility wrapper for left hand grasp."""
    return execute_grasp_all_objects_wrist(
        robot=robot,
        scene_dir=scene_dir,
        viz_data=viz_data,
        gripper_name=LEFT_GRIPPER_NAME,
        arm=LEFT_ARM,
        gripper_motor=LEFT_GRIPPER_MOTOR,
        dry_run=dry_run,
        target_description="L-pipe (left arm)",
    )


def main(args=None):
    logger.info(f"[Main] Args: {args}")
    mode = getattr(args, "mode", "real")
    scene_dir = getattr(args, "scene_dir", None)
    yolo_model = getattr(args, "yolo_model", None)
    visualize = getattr(args, "visualize", True)

    if not scene_dir:
        logger.error("[Main] --scene-dir is required (e.g. assets/zerith/real_scene/00).")
        return

    robot = None
    try:
        if mode != "sim":
            drive_chassis = getattr(args, "move_chassis", True)
            robot = H1Robot()
            logger.info("[INIT] Instantiating robot and connecting...")
            if not robot.robot_connect():
                logger.info("Failed to connect to robot!")
                return

            robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
            robot.robot_init()

            approach_workspace(robot, drive_chassis=drive_chassis)
            logger.info("[Main] Workspace reached, please reset the environment!")
            import pdb; pdb.set_trace()
        else:
            logger.info("[Main] Sim mode: robot is NOT connected or operated; "
                        "processing the saved scenes offline.")

        # =========================================================================
        # 1. Left-hand (wrist) camera & Left arm: Grasp L-shaped elbow pipe
        # =========================================================================
        left_scene = scene_dir + LEFT_HAND_CAM_SUFFIX
        logger.info(f"[Main] === Left Camera Pipeline: Target = L-shaped elbow pipe ({LEFT_TARGET_CLASSES}) ===")
        logger.info(f"[Main] Capturing left-hand camera scene into {left_scene} ...")
        rgb_l, depth_l = acquire_rgbd(left_scene, mode=mode, camera_name=LEFT_HAND_CAMERA_NAME)
        if depth_l is not None:
            logger.info(f"[Main] LEFT HAND depth: shape={depth_l.shape} dtype={depth_l.dtype}")
        det_l = detect_and_segment(rgb_l, yolo_model, left_scene, allowed_classes=LEFT_TARGET_CLASSES)
        if mode == "real":
            left_cam_pose = compute_hand_camera_pose(robot, arm=LEFT_ARM)
            write_meta_data(
                left_scene, robot, len(det_l),
                camera_pose=left_cam_pose, intrinsics=K_LEFT_HAND_COLOR,
            )
        else:
            logger.info("[Main] Sim mode: keeping existing left-scene meta_data.json.")
        summary_left, viz_data_left = generate_and_save_grasps(left_scene, gripper_names=[LEFT_GRIPPER_NAME])

        # =========================================================================
        # 2. Right-hand (wrist) camera & Right arm: Grasp Door handle
        # =========================================================================
        right_scene = scene_dir + RIGHT_HAND_CAM_SUFFIX
        logger.info(f"[Main] === Right Camera Pipeline: Target = Door handle ({RIGHT_TARGET_CLASSES}) ===")
        logger.info(f"[Main] Capturing right-hand camera scene into {right_scene} ...")
        rgb_r, depth_r = acquire_rgbd(right_scene, mode=mode, camera_name=RIGHT_HAND_CAMERA_NAME)
        if depth_r is not None:
            logger.info(f"[Main] RIGHT HAND depth: shape={depth_r.shape} dtype={depth_r.dtype}")
        det_r = detect_and_segment(rgb_r, yolo_model, right_scene, allowed_classes=RIGHT_TARGET_CLASSES)
        if mode == "real":
            right_cam_pose = compute_hand_camera_pose(robot, arm=RIGHT_ARM)
            write_meta_data(
                right_scene, robot, len(det_r),
                camera_pose=right_cam_pose, intrinsics=K_RIGHT_HAND_COLOR,
            )
        else:
            logger.info("[Main] Sim mode: keeping existing right-scene meta_data.json.")
        summary_right, viz_data_right = generate_and_save_grasps(right_scene, gripper_names=[RIGHT_GRIPPER_NAME])

        # =========================================================================
        # 3. Optional Visualization
        # =========================================================================
        if visualize:
            if viz_data_left and viz_data_left.get("grippers", {}).get(LEFT_GRIPPER_NAME, {}).get("grasps"):
                logger.info(f"[Main] Running left-hand camera visualization in main thread for {left_scene}.")
                visualize_saved_grasps(left_scene, viz_data=viz_data_left, port=8080)
            elif viz_data_right and viz_data_right.get("grippers", {}).get(RIGHT_GRIPPER_NAME, {}).get("grasps"):
                logger.info(f"[Main] Running right-hand camera visualization in main thread for {right_scene}.")
                visualize_saved_grasps(right_scene, viz_data=viz_data_right, port=8080)
            else:
                logger.warning("[Main] No viz_data returned for visualization; skipping.")
        else:
            logger.info("[Main] Visualization disabled by --no-visualize flag.")

        # =========================================================================
        # 4. Grasp Execution
        # =========================================================================
        # (1) Left hand grasps L-shaped elbow pipes
        if viz_data_left and viz_data_left.get("grippers", {}).get(LEFT_GRIPPER_NAME, {}).get("grasps"):
            logger.info("[Main] Executing left-hand grasp(s) for L-shaped elbow pipe...")
            execute_grasp_all_objects_wrist(
                robot=robot,
                scene_dir=left_scene,
                viz_data=viz_data_left,
                gripper_name=LEFT_GRIPPER_NAME,
                arm=LEFT_ARM,
                gripper_motor=LEFT_GRIPPER_MOTOR,
                dry_run=(mode != "real"),
                target_description="L-shaped elbow pipe (left arm)",
            )
        else:
            logger.info("[Main] No left-hand grasps for L-shaped elbow pipe to execute.")

        # (2) Right hand grasps Door handles
        if viz_data_right and viz_data_right.get("grippers", {}).get(RIGHT_GRIPPER_NAME, {}).get("grasps"):
            logger.info("[Main] Executing right-hand grasp(s) for door handle...")
            execute_grasp_all_objects_wrist(
                robot=robot,
                scene_dir=right_scene,
                viz_data=viz_data_right,
                gripper_name=RIGHT_GRIPPER_NAME,
                arm=RIGHT_ARM,
                gripper_motor=RIGHT_GRIPPER_MOTOR,
                dry_run=(mode != "real"),
                target_description="door handle (right arm)",
            )
        else:
            logger.info("[Main] No right-hand grasps for door handle to execute.")

        logger.info("[Main] Pipeline finished successfully.")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        logger.error(f"Runtime exception: {e}")

    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if robot is not None and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
