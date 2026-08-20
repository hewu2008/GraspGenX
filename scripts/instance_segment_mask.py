#!/usr/bin/env python3
"""Instance segmentation on an RGB image using YOLO and save mask predictions.

Runs YOLO instance segmentation on the given image, saves each detected instance
as a separate binary mask PNG, and produces a combined mask image where all
instances are represented together.

Usage
-----
    python scripts/instance_segment_mask.py assets/zerith/real_scene/00/rgb.png
    python scripts/instance_segment_mask.py assets/zerith/real_scene/00/rgb.png \\
        --model assets/zerith/yolo/last_20260807_v0.pt \\
        --conf 0.5 --iou 0.7 --output outputs/seg_results/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEVICE_CONFIG = 0
FALLBACK_TO_CPU = True
IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.6
IOU_THRESHOLD = 0.7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instance segmentation on an RGB image using YOLO and save masks."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input RGB image (e.g. rgb.png).",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="assets/zerith/yolo/last_20260807_v0.pt",
        help="Path to YOLO segmentation model weights (.pt file).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help="Inference image size.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for masks and visualization. Defaults to <image_dir>/seg_masks/",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cpu', '0', '0,1'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        default=True,
        help="Save individual binary mask PNGs for each detected instance.",
    )
    parser.add_argument(
        "--no_save_individual",
        dest="save_individual",
        action="store_false",
        help="Do not save individual mask PNGs.",
    )
    return parser.parse_args()


def resolve_device():
    try:
        import torch
        if torch.cuda.is_available():
            return DEVICE_CONFIG
        else:
            if FALLBACK_TO_CPU:
                print("[!] CUDA not detected, switching to CPU mode.")
                return "cpu"
            else:
                print("[!] CUDA not detected, and CPU fallback is disabled. Using CPU directly.")
                return "cpu"
    except ImportError:
        print("[!] torch is not installed. Using CPU mode.")
        return "cpu"


def color_from_id(instance_id):
    rng = np.random.RandomState(instance_id * 137 + 42)
    return tuple(int(c) for c in rng.randint(0, 256, size=3))


def main():
    args = parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    print(f"[*] Input image: {image_path}")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    print(f"[*] Model path: {model_path}")

    device = args.device if args.device is not None else resolve_device()
    print(f"[*] Device: {device}")

    from ultralytics import YOLO
    print("[*] Loading YOLO segmentation model ...")
    model = YOLO(str(model_path))
    print(f"[+] Model loaded. Task: {model.task}")
    print(f"[*] Classes: {model.names}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    print(f"[*] Image size: {w}x{h}")

    print(f"[*] Running inference (conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}) ...")
    results = model.predict(
        source=img,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    print("[+] Inference complete.")

    result = results[0]
    num_instances = 0
    class_counts = {}

    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        boxes_data = result.boxes
        num_instances = len(masks_data)

        output_dir = args.output
        if output_dir is None:
            output_dir = image_path.parent / "seg_masks"
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Output directory: {output_dir}")
        print(f"[*] Detected {num_instances} instance(s).")

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for idx in range(num_instances):
            cls_id = int(boxes_data.cls[idx])
            cls_name = model.names.get(cls_id, str(cls_id))
            conf_val = float(boxes_data.conf[idx])

            if cls_name not in class_counts:
                class_counts[cls_name] = 0
            class_counts[cls_name] += 1

            mask_resized = cv2.resize(
                masks_data[idx].astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            # Per-instance label matches the label_map convention in
            # meta_data.json (obj_1=101, obj_2=102, ...).
            combined_mask[binary_mask == 1] = 101 + idx

            color = color_from_id(idx)
            color_mask[binary_mask == 1] = color

            if args.save_individual:
                mask_filename = f"mask_{idx:02d}_{cls_name}_conf{conf_val:.2f}.png"
                mask_path = output_dir / mask_filename
                cv2.imwrite(str(mask_path), binary_mask * 255)

        combined_path = output_dir / f"combined_mask_{image_path.stem}.png"
        cv2.imwrite(str(combined_path), combined_mask)
        print(f"[+] Combined mask saved: {combined_path}")

        color_path = output_dir / f"color_mask_{image_path.stem}.png"
        cv2.imwrite(str(color_path), color_mask)
        print(f"[+] Color mask saved: {color_path}")

        summary_path = output_dir / f"summary_{image_path.stem}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Confidence threshold: {args.conf}\n")
            f.write(f"IoU threshold: {args.iou}\n")
            f.write(f"Total instances: {num_instances}\n")
            f.write(f"\nClass breakdown:\n")
            for cls_name, count in sorted(class_counts.items()):
                f.write(f"  {cls_name}: {count}\n")
        print(f"[+] Summary saved: {summary_path}")

    else:
        print("[!] No instances detected.")

    print("[+] Done.")


if __name__ == "__main__":
    main()