#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))
from camera_client import CameraClient

GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"

IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.6
IOU_THRESHOLD = 0.7

DEVICE_CONFIG = 0
FALLBACK_TO_CPU = True

INFERENCE_INTERVAL = 1
DISPLAY_FPS = True
CONFIDENCE_STEP = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime YOLO object detection via camera streaming")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="/path/to/your/model.pt",
        help="Path to YOLO model weights (.pt file)",
    )
    return parser.parse_args()


def clamp_confidence(value):
    return max(0.01, min(1.0, value))


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


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights file not found: {model_path}")
    print(f"[*] Model path: {model_path}")

    device = resolve_device()
    print(f"[*] Device: {device}")

    from ultralytics import YOLO
    print("[*] Loading YOLO model ...")
    model = YOLO(str(model_path))
    print("[+] Model loaded successfully.")

    client = CameraClient(grpc_target=GRPC_TARGET)
    print(f"[*] Connecting to service {GRPC_TARGET} ...")
    client.start()
    print(f"[+] Connected. Capturing frames from camera [{CAMERA_NAME}] ...")
    print(f"[*] Controls: +/-/= adjust confidence, r reset, q quit")

    confidence = DEFAULT_CONFIDENCE
    frame_count = 0
    last_results = None
    last_infer_time = 0.0
    fps = 0.0
    fps_display_interval = 0.5
    last_fps_update_time = time.time()
    fps_frame_count = 0

    try:
        while True:
            data = client.get_latest_frame(CAMERA_NAME)
            if data is None:
                continue

            img, ts = data
            frame_count += 1
            fps_frame_count += 1

            should_infer = (frame_count % INFERENCE_INTERVAL == 0) or (last_results is None)

            if should_infer:
                t0 = time.time()
                results = model.predict(
                    source=img,
                    imgsz=IMAGE_SIZE,
                    conf=confidence,
                    iou=IOU_THRESHOLD,
                    device=device,
                    verbose=False,
                )
                last_results = results
                last_infer_time = time.time() - t0

                now = time.time()
                if now - last_fps_update_time >= fps_display_interval:
                    elapsed = now - last_fps_update_time
                    fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    last_fps_update_time = now

            if last_results is not None and len(last_results) > 0:
                annotated = last_results[0].plot()
            else:
                annotated = img

            if DISPLAY_FPS:
                info_lines = []
                info_lines.append(f"FPS: {fps:.1f}")
                info_lines.append(f"Inference: {last_infer_time*1000:.0f}ms")
                info_lines.append(f"Conf: {confidence:.2f}  +/- adjust  r reset")

                y_offset = 25
                for line in info_lines:
                    cv2.putText(
                        annotated, line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )
                    y_offset += 22

            cv2.imshow("Realtime YOLO Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[+] User requested exit.")
                break
            elif key in (ord('+'), ord('=')):
                confidence = clamp_confidence(confidence + CONFIDENCE_STEP)
                print(f"[*] Confidence: {confidence:.2f}")
                last_results = None
            elif key in (ord('-'), ord('_')):
                confidence = clamp_confidence(confidence - CONFIDENCE_STEP)
                print(f"[*] Confidence: {confidence:.2f}")
                last_results = None
            elif key == ord('r'):
                confidence = DEFAULT_CONFIDENCE
                print(f"[*] Confidence reset to default: {confidence:.2f}")
                last_results = None

    except Exception as e:
        print(f"\n[!] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[*] Releasing resources and disconnecting ...")
        client.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
