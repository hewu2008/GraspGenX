"""Top-level orchestration of the end-to-end grasp pipeline."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from lib_h1_sdk_python import H1Robot, MotorControlMode

from .robot_motion import move_chassis, prepare_robot_posture, move_arm_to_ready_pose
from .perception import (
    acquire_rgbd,
    detect_and_segment,
    write_meta_data,
    generate_and_save_grasps,
)
from .grasp_visualization import visualize_saved_grasps
from .grasp_executor import (
    resolve_grasp_target,
    resolve_grasp_target_hand,
    grasp_object,
)
from .config import (
    CAMERA_NAME,
    HAND_CAMERA_NAME,
    HAND_CAM_SUFFIX,
    K_HAND_COLOR,
)
from .logging_utils import get_logger

logger = get_logger(__name__)

# Left-gripper used for the sequential pick-and-place execution.
EXEC_GRIPPER = "zerith_left_gripper"


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


def execute_grasp_all_objects(robot, scene_dir, viz_data, dry_run=False):
    """Grasp and place every detected object in sequence using the left gripper.

    For each object, the top-1 conf grasp pose (world frame) is pulled from
    ``viz_data``, transformed back to the camera frame with the ``camera_pose``
    stored in meta_data.json, converted to a left-arm relative target via
    ``resolve_grasp_target``, and executed with ``grasp_object``.

    Args:
        robot: connected H1Robot instance.
        scene_dir: path to the scene directory (contains meta_data.json).
        viz_data: visualization data returned by generate_and_save_grasps.
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
    if EXEC_GRIPPER not in grippers:
        logger.error(f"[Grasp] gripper {EXEC_GRIPPER!r} not found in viz_data")
        return

    grasps_data = grippers[EXEC_GRIPPER].get("grasps", {})
    if not grasps_data:
        logger.info("[Grasp] No grasps to execute for left gripper.")
        return

    labels = list(grasps_data.keys())
    logger.info(f"[Grasp] Picking {len(labels)} object(s) in order: {labels}")
    for obj_label in labels:
        data = grasps_data[obj_label]
        grasps = data.get("grasps")
        conf = data.get("conf")
        if grasps is None or len(grasps) == 0:
            logger.warning(f"[Grasp] {obj_label}: no grasps, skipping")
            continue

        best_idx = int(np.argmax(conf))
        T_world = np.asarray(grasps[best_idx], dtype=np.float64)  # grasp pose (world)
        T_cam = cam_pose_inv @ T_world  # world -> camera

        logger.info(
            f"[Grasp] {obj_label}: pre(world)  pos={T_world[:3, 3].tolist()}, "
            f"euler_xyz(deg)={R.from_matrix(T_world[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
        )
        logger.info(
            f"[Grasp] {obj_label}: post(camera) pos={T_cam[:3, 3].tolist()}, "
            f"euler_xyz(deg)={R.from_matrix(T_cam[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
        )

        logger.info(f"[Grasp] {obj_label}: best conf={conf[best_idx]:.3f}")
        _euler = R.from_matrix(T_cam[:3, :3]).as_euler("xyz", degrees=True)
        logger.info(
            f"[Grasp] {obj_label}: graspgen pos(cam)={T_cam[:3, 3].tolist()}, "
            f"euler_xyz(deg)={_euler.tolist()}"
        )

        target_pos, target_quat = resolve_grasp_target(robot, T_cam)
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
            grasp_object(robot, target_pos, target_quat)
        logger.info(f"[Grasp] {obj_label}: grasped & placed.")

    logger.info("[Grasp] All objects grasped & placed.")


def execute_grasp_all_objects_hand(robot, scene_dir, viz_data, dry_run=False):
    """Grasp and place every detected object in sequence using the LEFT-HAND
    camera result and the fixed-URDF-offset hand-eye transform.

    The head camera pipeline is NOT used here (see ``log_head_hand_comparison``);
    this path drives the robot exclusively from the wrist camera.
    """
    grippers = viz_data.get("grippers", {})
    if EXEC_GRIPPER not in grippers:
        logger.error(f"[HandGrasp] gripper {EXEC_GRIPPER!r} not found in viz_data")
        return
    grasps_data = grippers[EXEC_GRIPPER].get("grasps", {})
    if not grasps_data:
        logger.info("[HandGrasp] No grasps to execute for the left-hand camera.")
        return

    labels = list(grasps_data.keys())
    logger.info(f"[HandGrasp] Picking {len(labels)} object(s) in order: {labels}")
    for obj_label in labels:
        data = grasps_data[obj_label]
        grasps = data.get("grasps")
        conf = data.get("conf")
        if grasps is None or len(grasps) == 0:
            logger.warning(f"[HandGrasp] {obj_label}: no grasps, skipping")
            continue

        best_idx = int(np.argmax(conf))
        T_hand = np.asarray(grasps[best_idx], dtype=np.float64)  # grasp (hand-camera frame)
        logger.info(
            f"[HandGrasp] {obj_label}: grasp pos(cam)={T_hand[:3, 3].tolist()}, "
            f"quat(cam)={R.from_matrix(T_hand[:3, :3]).as_quat().tolist()}, "
            f"euler_xyz(deg)={R.from_matrix(T_hand[:3, :3]).as_euler('xyz', degrees=True).tolist()}"
        )

        target_pos, target_quat = resolve_grasp_target_hand(robot, T_hand)
        if target_pos is None:
            logger.error(f"[HandGrasp] {obj_label}: failed to resolve target, skipping")
            continue

        _target_euler = R.from_quat(target_quat).as_euler("xyz", degrees=True)
        logger.info(
            f"[HandGrasp] {obj_label}: best conf={conf[best_idx]:.3f}, "
            f"cam pos={T_hand[:3, 3].tolist()}, "
            f"target_pos={target_pos.tolist()}, "
            f"target_quat={target_quat.tolist()}, "
            f"target_euler_xyz(deg)={_target_euler.tolist()}"
        )

        if not dry_run:
            grasp_object(robot, target_pos, target_quat)
        logger.info(f"[HandGrasp] {obj_label}: grasped & placed.")

    logger.info("[HandGrasp] All objects grasped & placed.")


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

        logger.info(f"[Main] Capturing head-camera scene into {scene_dir} ...")
        rgb, depth = acquire_rgbd(scene_dir, mode=mode, camera_name=CAMERA_NAME)
        if depth is not None:
            logger.info(f"[Main] HEAD depth: shape={depth.shape} dtype={depth.dtype}")
        detections = detect_and_segment(rgb, yolo_model, scene_dir)
        if mode == "real":
            write_meta_data(scene_dir, robot, len(detections))
        else:
            logger.info("[Main] Sim mode: keeping existing meta_data.json.")
        summary_head, viz_data_head = generate_and_save_grasps(scene_dir)
        if mode == "real":
            execute_grasp_all_objects(robot, scene_dir, viz_data_head, dry_run=True)
        else:
            logger.info("[Main] Sim mode: skipping head-camera grasp resolution/execution.")

        # ---- Left-hand (wrist) camera: the scene that drives grasping ----
        # Sibling of --scene-dir, e.g. ".../real_scene/02_hand_camera".
        hand_scene = scene_dir + HAND_CAM_SUFFIX
        logger.info(f"[Main] Capturing left-hand camera scene into {hand_scene} ...")
        rgb_hd, depth_hd = acquire_rgbd(hand_scene, mode=mode, camera_name=HAND_CAMERA_NAME)
        if depth_hd is not None:
            logger.info(f"[Main] HAND depth: shape={depth_hd.shape} dtype={depth_hd.dtype}")
        det_hd = detect_and_segment(rgb_hd, yolo_model, hand_scene)
        # camera_pose=identity => the hand-camera frame IS the world frame; the
        # fixed-URDF-offset hand-eye chain in resolve_grasp_target_hand consumes
        # the grasps directly in that frame, so no FK is needed.
        if mode == "real":
            write_meta_data(
                hand_scene, robot, len(det_hd),
                camera_pose=np.eye(4), intrinsics=K_HAND_COLOR,
            )
        else:
            logger.info("[Main] Sim mode: keeping existing hand-scene meta_data.json.")
        summary_hand, viz_data_hand = generate_and_save_grasps(hand_scene)

        if visualize and viz_data_head:
            # Run visualization on the main thread. It blocks until the user
            # presses Ctrl+C, then returns so the rest of the flow can proceed.
            # This shows the head-camera scene used only as a comparison.
            logger.info("[Main] Running head-camera visualization in main thread.")
            visualize_saved_grasps(scene_dir, viz_data=viz_data_head, port=8080)
        elif not visualize:
            logger.info("[Main] Visualization disabled by --no-visualize flag.")
        else:
            logger.warning("[Main] No head viz_data returned; skipping visualization.")

        # Grasp & place every detected object in sequence using the left-hand
        # camera result (the head result was comparison-only, never executed).
        if viz_data_hand:
            if mode == "real":
                execute_grasp_all_objects_hand(robot, hand_scene, viz_data_hand)
            else:
                logger.info("[Main] Sim mode: skipping hand-camera grasp execution.")
        else:
            logger.warning("[Main] No hand-camera viz_data; nothing to execute.")
        logger.info("[Main] Pipeline finished.")

    except KeyboardInterrupt:
        logger.warning("Ctrl+C received; preparing safe shutdown...")
    except Exception as e:
        logger.error(f"Runtime exception: {e}")

    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if robot is not None and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()
