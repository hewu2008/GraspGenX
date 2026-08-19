#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
CAPTURE_DIR="${DETECTION_DIR}/captured_images"

if [ -d "$CAPTURE_DIR" ]; then
    file_count=$(find "$CAPTURE_DIR" -type f | wc -l)
    echo "[!] captured_images directory already exists (${file_count} files found)"
    read -r -p "Delete it before starting? [y/N]: " confirm
    case "$confirm" in
        [Yy]|[Yy][Ee][Ss])
            rm -rf "$CAPTURE_DIR"
            echo "[*] Directory removed."
            ;;
        *)
            echo "[*] Keeping existing directory. New captures will be added to it."
            ;;
    esac
fi

cd "$DETECTION_DIR" || exit 1
python live_camera_capture.py
