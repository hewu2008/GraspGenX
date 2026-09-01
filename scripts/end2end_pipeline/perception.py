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

import cv2
import numpy as np

from .config import (
    GRPC_TARGET,
    CAMERA_NAME,
    SCENE_BOUNDS,
)
from .logging_utils import get_logger

logger = get_logger(__name__)

# YOLO inference hyperparameters (the model path itself is supplied by the caller).
YOLO_CONFIDENCE = 0.8
YOLO_IOU_THRESHOLD = 0.7
YOLO_IMAGE_SIZE = 640

RGB_FILENAME = "rgb.png"
DEPTH_FILENAME = "depth.npy"
SEG_FILENAME = "seg.png"


def acquire_rgbd(scene_dir, mode="sim", camera_name=CAMERA_NAME):
    """Return (rgb, depth) for the scene, acquiring by mode.

    sim:  read rgb.png / depth.npy from scene_dir.
    real: capture from ``camera_name`` and persist rgb.png / depth.npy into
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
    logger.info(f"[Perc] Capturing RGB-D from camera ({GRPC_TARGET}/{camera_name}) into {scene_dir}")
    from camera_client import CameraClient
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    client.start()
    try:
        for _ in range(50):
            depth_data = client.get_latest_depth(camera_name)
            color_data = client.get_latest_frame(camera_name)
            if depth_data is not None and color_data is not None:
                depth_raw, _ = depth_data
                color_raw, _ = color_data
                # Depth unit differs per camera: the head camera stream
                # reports millimetres (/1000 -> m), while the wrist camera
                # reports 0.1 mm (/10000 -> m).
                depth_scale = 1000.0 if camera_name == CAMERA_NAME else 10000.0
                depth_m = depth_raw.astype(np.float32) / depth_scale
                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_m)
                logger.info(f"[Perc] Captured and saved {rgb_path} and {depth_path}")
                return color_raw, depth_m
            time.sleep(0.1)
        logger.error("[Perc] Image capture timed out. Check the gRPC node.")
        return None, None
    finally:
        client.stop()


def is_class_allowed(cls_name, allowed_classes):
    """Check if cls_name matches any entry in allowed_classes."""
    if allowed_classes is None:
        return True
    cls_clean = str(cls_name).lower().replace("-", "_").replace(" ", "_")
    for allowed in allowed_classes:
        allowed_clean = str(allowed).lower().replace("-", "_").replace(" ", "_")
        if cls_clean == allowed_clean or allowed_clean in cls_clean or cls_clean in allowed_clean:
            return True
    return False


def detect_and_segment(rgb, yolo_model, scene_dir, allowed_classes=None):
    """Run YOLO instance segmentation; return detections and save seg.png.

    Optionally filters instances by ``allowed_classes`` (e.g. {'elbow_pipe'} or
    {'interior_door_handle'}).

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
        cls_id = int(boxes.cls[idx])
        conf = float(boxes.conf[idx])
        cls_name = class_names.get(cls_id, str(cls_id))

        if allowed_classes is not None and not is_class_allowed(cls_name, allowed_classes):
            logger.info(
                f"[Perc] [{idx}] {cls_name} (conf={conf:.2f}) skipped (not in allowed_classes: {allowed_classes})"
            )
            continue

        mask_resized = cv2.resize(
            masks_data[idx].astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = (mask_resized > 0.5).astype(np.uint8)
        xyxy = boxes.xyxy[idx].cpu().numpy()
        bbox = [int(v) for v in xyxy]

        kept_idx = len(detections)
        # label_map convention: obj_1=101, obj_2=102, ...
        combined[mask == 1] = 101 + kept_idx
        detections.append({
            "bbox": bbox,
            "mask": mask,
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": conf,
        })
        logger.info(
            f"[Perc] [kept obj_{kept_idx + 1}] {cls_name} conf={conf:.2f} bbox={bbox}"
        )

    seg_path = _save_seg_png(scene_dir, combined)
    logger.info(
        f"[Perc] {len(detections)} instance(s) matching allowed_classes={allowed_classes}; seg saved to {seg_path}"
    )
    return detections


def _save_seg_png(scene_dir, combined_mask):
    os.makedirs(scene_dir, exist_ok=True)
    seg_path = os.path.join(scene_dir, SEG_FILENAME)
    cv2.imwrite(seg_path, combined_mask)
    return seg_path
