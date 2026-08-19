#!/bin/bash

IMAGE_PATH=${1:-assets/zerith/real_scene/00/rgb.png}
MODEL_PATH=${2:-assets/zerith/yolo/last_20260807_v0.pt}
CONF=${3:-0.9}
IOU=${4:-0.7}
OUTPUT_DIR=${5:-}

ARGS=("$IMAGE_PATH")
ARGS+=(--model "$MODEL_PATH")
ARGS+=(--conf "$CONF")
ARGS+=(--iou "$IOU")

if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output "$OUTPUT_DIR")
fi

python scripts/instance_segment_mask.py "${ARGS[@]}"