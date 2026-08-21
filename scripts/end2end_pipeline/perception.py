"""RGB-D capture and Zerith detection + register for all detected objects."""

import os
import time

import cv2
import numpy as np

from camera_client import CameraClient
from zerith.zerith_client import (
    create_client,
    detect_parts,
    process_label,
    save_detection_debug,
)

from .config import (
    GRPC_TARGET,
    CAMERA_NAME,
    ZMQ_SERVER_ADDR,
    CLIENT_DEBUG_DIR,
    REGISTER_ITERATIONS,
    K_COLOR,
)
from .logging_utils import get_logger

logger = get_logger(__name__)


# ================= 2. RGB-D capture =================
def capture_rgbd_data():
    logger.info(f"[B] Connecting to camera service for RGB-D ({GRPC_TARGET})...")
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
                logger.info(f" -> Captured. Saved {rgb_path} and {depth_path}")
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
    logger.info("[C] Requesting Zerith Detection + Register for all objects...")
    os.makedirs(debug_dir, exist_ok=True)

    client = create_client(ZMQ_SERVER_ADDR)
    if client is None:
        logger.error(" -> Failed to connect to Zerith server.")
        return {}

    pose_files = {}

    try:
        # 1. Detection: keep all returned detections.
        color, boxes = detect_parts(client, rgb_path)
        if color is None or boxes is None:
            logger.error(" -> Detection call failed.")
            return {}

        if len(boxes) == 0:
            logger.warning(" -> No objects detected in current frame.")
            return {}

        save_detection_debug(debug_dir, color, boxes)
        logger.info(f" -> Detection returned {len(boxes)} objects. Starting Register.")

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

            logger.info(f" -> [{detection_index + 1}/{len(boxes)}] "
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
                logger.warning(f" -> {category_id}_{instance_index} register failed; continuing.")
                continue

            pose_path = build_pose_path(debug_dir, category_id, instance_index)

            if not os.path.isfile(pose_path):
                logger.warning(f" -> Register succeeded but pose file missing: {pose_path}")
                continue

            pose_files[(category_id, instance_index)] = pose_path
            logger.info(f" -> Pose saved: {pose_path}")

        logger.info(f" -> Successfully obtained {len(pose_files)} object poses:")
        for (category_id, instance_index), pose_path in pose_files.items():
            logger.info(f"    {category_id}_{instance_index}: {pose_path}")

        return pose_files

    except Exception as e:
        logger.error(f" -> Perception client crashed: {e}")
        return {}

    finally:
        client.close()
