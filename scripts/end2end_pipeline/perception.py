"""RGB-D acquisition and YOLO detection + segmentation.

The pipeline runs the same detection + segmentation regardless of mode; the
only mode-dependent step is how the RGB-D frame is obtained:

  - sim:  read ``rgb.png`` and ``depth.npy`` already stored in the scene dir.
  - real: capture from the head camera and write ``rgb.png`` / ``depth.npy``
          into the scene dir so the on-disk scene mirrors a sim run.

Detection + segmentation runs YOLO instance segmentation, returns each
instance's bbox and mask, and writes the combined instance-label mask to
``<scene_dir>/seg.png`` (label_map convention: obj_1=101, obj_2=102, ...).
"""

import os
import time
import json

import cv2
import numpy as np

from .config import GRPC_TARGET, CAMERA_NAME, K_COLOR, SCENE_BOUNDS
from .camera_pose import compute_camera_pose
from .logging_utils import get_logger

logger = get_logger(__name__)

# YOLO inference hyperparameters (the model path itself is supplied by the caller).
YOLO_CONFIDENCE = 0.9
YOLO_IOU_THRESHOLD = 0.7
YOLO_IMAGE_SIZE = 640

RGB_FILENAME = "rgb.png"
DEPTH_FILENAME = "depth.npy"
SEG_FILENAME = "seg.png"
META_FILENAME = "meta_data.json"


def acquire_rgbd(scene_dir, mode="sim"):
    """Return (rgb, depth) for the scene, acquiring by mode.

    sim:  read rgb.png / depth.npy from scene_dir.
    real: capture from the head camera and persist rgb.png / depth.npy into
          scene_dir so a later sim run reads the same data.
    """
    os.makedirs(scene_dir, exist_ok=True)
    rgb_path = os.path.join(scene_dir, RGB_FILENAME)
    depth_path = os.path.join(scene_dir, DEPTH_FILENAME)

    if mode == "sim":
        logger.info(f"[Perc] Reading RGB-D from {rgb_path} and {depth_path}")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            logger.error(f"[Perc] Failed to read {rgb_path}")
            return None, None
        try:
            depth = np.load(depth_path)
            logger.info(
                f"[Perc] Loaded depth: shape={depth.shape} dtype={depth.dtype}"
            )
        except Exception as e:
            logger.warning(
                f"[Perc] Could not load {depth_path} ({e}); continuing with rgb only."
            )
            depth = None
        return rgb, depth

    # real: capture from camera and persist into the scene directory.
    logger.info(f"[Perc] Capturing RGB-D from camera ({GRPC_TARGET}) into {scene_dir}")
    from camera_client import CameraClient
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    client.start()
    try:
        for _ in range(50):
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME)
            if depth_data is not None and color_data is not None:
                depth_raw_mm, _ = depth_data
                color_raw, _ = color_data
                depth_m = depth_raw_mm.astype(np.float32) / 1000.0
                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_m)
                logger.info(f"[Perc] Captured and saved {rgb_path} and {depth_path}")
                return color_raw, depth_m
            time.sleep(0.1)
        logger.error("[Perc] Image capture timed out. Check the gRPC node.")
        return None, None
    finally:
        client.stop()


def detect_and_segment(rgb, yolo_model, scene_dir):
    """Run YOLO instance segmentation; return detections and save seg.png.

    Returns a list of dicts: ``{bbox, mask, class_id, class_name, conf}``.
    The combined instance-label mask is written to ``<scene_dir>/seg.png``.
    """
    if rgb is None:
        logger.error("[Perc] No RGB image to segment.")
        return []
    if not yolo_model:
        logger.error("[Perc] YOLO model path is required for detection.")
        return []
    if not os.path.isfile(yolo_model):
        logger.error(f"[Perc] YOLO model not found: {yolo_model}")
        return []

    from ultralytics import YOLO

    logger.info(f"[Perc] Loading YOLO model: {yolo_model}")
    model = YOLO(yolo_model)
    class_names = model.names

    results = model.predict(
        source=rgb,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU_THRESHOLD,
        verbose=False,
    )
    result = results[0]
    if result.masks is None:
        logger.info("[Perc] No instances detected.")
        _save_seg_png(scene_dir, np.zeros(rgb.shape[:2], dtype=np.uint8))
        return []

    h, w = rgb.shape[:2]
    masks_data = result.masks.data.cpu().numpy()
    boxes = result.boxes

    detections = []
    combined = np.zeros((h, w), dtype=np.uint8)
    for idx in range(len(masks_data)):
        mask_resized = cv2.resize(
            masks_data[idx].astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = (mask_resized > 0.5).astype(np.uint8)
        cls_id = int(boxes.cls[idx])
        conf = float(boxes.conf[idx])
        xyxy = boxes.xyxy[idx].cpu().numpy()
        bbox = [int(v) for v in xyxy]
        cls_name = class_names.get(cls_id, str(cls_id))

        # label_map convention: obj_1=101, obj_2=102, ...
        combined[mask == 1] = 101 + idx
        detections.append({
            "bbox": bbox,
            "mask": mask,
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": conf,
        })
        logger.info(
            f"[Perc] [{idx}] {cls_name} conf={conf:.2f} bbox={bbox}"
        )

    seg_path = _save_seg_png(scene_dir, combined)
    logger.info(
        f"[Perc] {len(detections)} instance(s); seg saved to {seg_path}"
    )
    return detections


def _save_seg_png(scene_dir, combined_mask):
    os.makedirs(scene_dir, exist_ok=True)
    seg_path = os.path.join(scene_dir, SEG_FILENAME)
    cv2.imwrite(seg_path, combined_mask)
    return seg_path


def write_meta_data(scene_dir, robot, num_objects):
    """Write meta_data.json for the just-captured scene.

    Fields match what graspgenx.utils.scene_loaders expects:
      intrinsics    : 3x3 K (from config.K_COLOR)
      camera_pose   : 4x4 camera-to-world (from IMU + motors via compute_camera_pose)
      label_map     : {"ground": 0, "obj_i": 100 + i}  (matches seg.png convention)
      scene_bounds  : workspace bbox (from config.SCENE_BOUNDS)

    Returns the written path, or None if the camera pose could not be computed.
    """
    T = compute_camera_pose(robot)
    if T is None:
        logger.error("[Perc] Failed to compute camera pose; skipping meta_data.json.")
        return None

    label_map = {"ground": 0}
    for i in range(1, num_objects + 1):
        label_map[f"obj_{i}"] = 100 + i

    meta = {
        "intrinsics": K_COLOR.tolist(),
        "camera_pose": T.tolist(),
        "label_map": label_map,
        "scene_bounds": list(SCENE_BOUNDS),
    }
    os.makedirs(scene_dir, exist_ok=True)
    path = os.path.join(scene_dir, META_FILENAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[Perc] meta_data.json written to {path}")
    return path
