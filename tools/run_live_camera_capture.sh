#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CAPTURE_DIR="${PROJECT_DIR}/assets/zerith/real_scene/00"
CAMERA_NAME=${1:-rs/cam_left_wrist}

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

cd "$PROJECT_DIR" || exit 1
python scripts/live_camera_capture.py \
    --save_dir "$CAPTURE_DIR" \
    --camera_name "$CAMERA_NAME"
