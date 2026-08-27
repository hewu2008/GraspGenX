#!/bin/bash

DEPTH_PATH=${1:-assets/zerith/real_scene/02_cam_left_wrist/depth.npy}
RGB_PATH=${2:-assets/zerith/real_scene/02_cam_left_wrist/rgb.png}
MAX_DEPTH=${3:-}
CMAP=${4:-turbo}

ARGS=("$DEPTH_PATH")
if [ -n "$RGB_PATH" ] && [ -f "$RGB_PATH" ]; then
    ARGS+=(--rgb "$RGB_PATH")
fi
if [ -n "$MAX_DEPTH" ]; then
    ARGS+=(--max_depth "$MAX_DEPTH")
fi
if [ -n "$CMAP" ]; then
    ARGS+=(--cmap "$CMAP")
fi

python scripts/vis_depth.py "${ARGS[@]}"
