"""Simulated perception for the end-to-end grasp pipeline.

In simulation mode the pipeline does not connect to the robot, the camera,
or the Zerith server. It iterates recorded scenes under
``assets/zerith/real_scene`` and runs YOLO instance segmentation (detection +
segmentation) on each scene's ``rgb.png``, mirroring
``scripts/instance_segment_mask.py``. Per-instance and combined masks are
written under ``<scene>/seg_masks/``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .config import REAL_SCENE_DIR
from .logging_utils import get_logger

logger = get_logger(__name__)

# YOLO inference parameters (detection/segmentation hyperparameters, not the
# model itself — the model path is supplied by the caller).
YOLO_CONFIDENCE = 0.6
YOLO_IOU_THRESHOLD = 0.7
YOLO_IMAGE_SIZE = 640


def _color_from_id(instance_id):
    rng = np.random.RandomState(instance_id * 137 + 42)
    return tuple(int(c) for c in rng.randint(0, 256, size=3))


def _segment_image(model, image, conf, iou, imgsz):
    """Run YOLO instance segmentation; return masks, class ids, confidences."""
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    result = results[0]
    if result.masks is None:
        return [], [], []

    h, w = image.shape[:2]
    masks_data = result.masks.data.cpu().numpy()
    boxes = result.boxes

    masks, classes, confs = [], [], []
    for idx in range(len(masks_data)):
        mask_resized = cv2.resize(
            masks_data[idx].astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        masks.append((mask_resized > 0.5).astype(np.uint8))
        classes.append(int(boxes.cls[idx]))
        confs.append(float(boxes.conf[idx]))
    return masks, classes, confs


def _save_masks(scene_dir, image, masks, classes, confs, class_names):
    """Write per-instance and combined masks under <scene_dir>/seg_masks/."""
    h, w = image.shape[:2]
    out_dir = os.path.join(scene_dir, "seg_masks")
    os.makedirs(out_dir, exist_ok=True)

    combined = np.zeros((h, w), dtype=np.uint8)
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, mask in enumerate(masks):
        cls_name = class_names.get(classes[idx], str(classes[idx]))
        # label_map convention in meta_data.json: obj_1=101, obj_2=102, ...
        combined[mask == 1] = 101 + idx
        color_mask[mask == 1] = _color_from_id(idx)

        mask_path = os.path.join(
            out_dir, f"mask_{idx:02d}_{cls_name}_conf{confs[idx]:.2f}.png"
        )
        cv2.imwrite(mask_path, mask * 255)

    cv2.imwrite(os.path.join(out_dir, "combined_mask.png"), combined)
    cv2.imwrite(os.path.join(out_dir, "color_mask.png"), color_mask)
    return out_dir


def run_simulation(scene_dir=REAL_SCENE_DIR, yolo_model=None):
    """Run YOLO detection + segmentation on every recorded scene."""
    from ultralytics import YOLO

    if not yolo_model:
        logger.error(
            "[Sim] YOLO model path is required in simulation mode "
            "(pass it via --yolo-model)."
        )
        return
    if not os.path.isfile(yolo_model):
        logger.error(f"[Sim] YOLO model not found: {yolo_model}")
        return
    if not os.path.isdir(scene_dir):
        logger.error(f"Simulation scene directory not found: {scene_dir}")
        return

    logger.info(f"[Sim] Loading YOLO model: {yolo_model}")
    model = YOLO(yolo_model)
    class_names = model.names

    scenes = sorted(
        d for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d))
    )
    if not scenes:
        logger.warning(f"[Sim] No scene subdirectories in {scene_dir}")
        return

    logger.info(f"[Sim] {len(scenes)} scene(s): {scenes}")

    for name in scenes:
        sub = os.path.join(scene_dir, name)
        rgb_path = os.path.join(sub, "rgb.png")
        depth_path = os.path.join(sub, "depth.npy")

        logger.info(f"[Sim] Scene {name}: reading {rgb_path} and {depth_path}")
        image = cv2.imread(rgb_path)
        if image is None:
            logger.error(f"[Sim] {name}: failed to read rgb.png; skipping.")
            continue
        try:
            depth = np.load(depth_path)
            logger.info(
                f"[Sim] {name}: depth shape={depth.shape} dtype={depth.dtype}"
            )
        except Exception as e:
            logger.warning(
                f"[Sim] {name}: could not load depth.npy ({e}); continuing with rgb only."
            )

        masks, classes, confs = _segment_image(
            model, image, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_IMAGE_SIZE
        )
        logger.info(f"[Sim] {name}: detected {len(masks)} instance(s)")
        if not masks:
            continue

        out_dir = _save_masks(sub, image, masks, classes, confs, class_names)
        for idx, (cls, c) in enumerate(zip(classes, confs)):
            cls_name = class_names.get(cls, str(cls))
            logger.info(f"[Sim] {name}: [{idx}] {cls_name} conf={c:.2f}")
        logger.info(f"[Sim] {name}: masks saved to {out_dir}")

    logger.info("[Sim] Done.")
