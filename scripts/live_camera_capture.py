#!/usr/bin/env python3
import os
import sys
import cv2
import time
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))
from camera_client import CameraClient

GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
SAVE_DIR = Path("captured_images")
CAPTURE_INTERVAL = 1.0
CAPTURE_FORMAT = ".jpg"
CAPTURE_QUALITY = 95


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    client = CameraClient(grpc_target=GRPC_TARGET)
    print(f"[*] Connecting to service {GRPC_TARGET} ...")
    client.start()
    print(f"[+] Connected. Capturing frames from camera [{CAMERA_NAME}] ...")
    print(f"[*] Save directory: {SAVE_DIR.resolve()}")
    print(f"[*] Capture interval: {CAPTURE_INTERVAL}s")
    print("[*] Press 's' to save manually, 'q' to quit")

    last_capture_time = 0.0
    capture_count = 0

    try:
        while True:
            data = client.get_latest_frame(CAMERA_NAME)
            if data is None:
                continue

            img, ts = data
            cv2.imshow("Live Camera", img)

            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            should_auto_save = (now - last_capture_time >= CAPTURE_INTERVAL)
            should_manual_save = (key == ord('s'))

            if should_auto_save or should_manual_save:
                last_capture_time = now
                timestamp_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
                if should_manual_save:
                    timestamp_str += "_manual"
                file_path = SAVE_DIR / f"{timestamp_str}{CAPTURE_FORMAT}"
                cv2.imwrite(str(file_path), img, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_QUALITY])
                capture_count += 1
                print(f"[+] Saved: {file_path.name} (total: {capture_count})")

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
