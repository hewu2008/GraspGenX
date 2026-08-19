#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Local SDK bundles shipped under assets/zerith/sdk/
#   lib/   -> camera_client.cpython-310-x86_64-linux-gnu.so, lib_h1_sdk_python.so
#   proto/ -> robot_pb2.py, robot_pb2_grpc.py
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
_sdk_dir = os.path.join(_root, "assets", "zerith", "sdk")
for _sub in ("lib", "proto"):
    _p = os.path.join(_sdk_dir, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from camera_client import CameraClient

GRPC_TARGET = "localhost:50051"
# This server publishes wrist cameras (rs/cam_left_wrist, rs/cam_right_wrist),
# not rs/cam_high. Available tracks are logged at startup as "[PC] OnTrack: ... id=<name>".
CAMERA_NAME = "rs/cam_high"
SAVE_DIR = Path("captured_images")
RGB_FORMAT = ".png"
DEPTH_FORMAT = ".npy"  # float32 meters, matches graspgenx scene loader convention
# Depth is delivered as uint16 millimeters. Colorize for display only.
DEPTH_CMAP = cv2.COLORMAP_TURBO
DEPTH_MAX_M = 3.0  # upper bound (meters) for colormap scaling


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 mm depth -> 8-bit BGR colormap image (0 mm = invalid -> black)."""
    valid = depth_mm > 0
    depth_m = np.where(valid, depth_mm.astype(np.float32) / 1000.0, 0.0)
    vmax_m = max(float(depth_m[valid].max()) if valid.any() else DEPTH_MAX_M, 1e-3)
    vmax_m = min(vmax_m, DEPTH_MAX_M)
    norm = np.clip(depth_m / vmax_m, 0.0, 1.0)
    vis = (norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(vis, DEPTH_CMAP)
    vis[~valid] = 0  # mark invalid pixels black
    return vis


def main():
    parser = argparse.ArgumentParser(description="Live RGB+D capture from a Zerith camera service.")
    parser.add_argument("--save_dir", default=str(SAVE_DIR),
                        help=f"Directory for saved frames (default: {SAVE_DIR}).")
    parser.add_argument("--camera_name", default=CAMERA_NAME,
                        help=f"Camera track name (default: {CAMERA_NAME}). "
                             "Available tracks are logged at startup as '[PC] OnTrack: ... id=<name>'.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    print(f"[*] Connecting to service {GRPC_TARGET} ...")
    client.start()
    print(f"[+] Connected. Streaming RGB+D from camera [{args.camera_name}] ...")
    print(f"[*] Save directory: {save_dir.resolve()}")
    print("[*] Press 's' to save RGB+D, 'q' to quit")

    capture_count = 0

    try:
        while True:
            rgb_data = client.get_latest_frame(args.camera_name)
            depth_data = client.get_latest_depth(args.camera_name)

            if rgb_data is None and depth_data is None:
                print(f"[!] No RGB+D frame yet, continue. (ts={rgb_data[1] if rgb_data is not None else 'N/A'}, depth {depth_data[1] if depth_data is not None else 'N/A'})")
                continue

            if rgb_data is not None:
                img, ts = rgb_data
                cv2.imshow("RGB", img)

            if depth_data is not None:
                depth_mm, _ = depth_data
                depth_vis = colorize_depth(depth_mm)
                cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if rgb_data is None:
                    print("[!] No RGB frame yet, save skipped.")
                else:
                    img, ts = rgb_data
                    timestamp_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
                    rgb_path = save_dir / f"{timestamp_str}_rgb{RGB_FORMAT}"
                    cv2.imwrite(str(rgb_path), img, [cv2.IMWRITE_JPEG_QUALITY])
                    if depth_data is not None:
                        depth_path = save_dir / f"{timestamp_str}_depth{DEPTH_FORMAT}"
                        depth_m = depth_data[0].astype(np.float32) / 1000.0  # mm -> m
                        np.save(str(depth_path), depth_m)
                        print(f"[+] Saved: {rgb_path.name}, {depth_path.name}")
                    else:
                        print(f"[+] Saved: {rgb_path.name} (no depth)")
                    capture_count += 1
                    print(f"    total: {capture_count}")

            if key == ord('q'):
                print("\n[+] User requested exit.")
                break

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
