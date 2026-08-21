#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Required arguments:
#   $1 = scene directory, e.g. assets/zerith/real_scene/00
#   $2 = YOLO segmentation weights (.pt)
# Extra CLI args are forwarded and override the defaults below, e.g.
#   ./tools/run_end2end_grasp_pipeline.sh assets/zerith/real_scene/00 model.pt --mode real --move-chassis
SCENE_DIR="${1:?usage: $0 <scene_dir> <yolo_model> [extra args...]}"
YOLO_MODEL="${2:?missing yolo_model path}"
shift 2

# Defaults: sim mode, no chassis motion. Override by appending --mode / --move-chassis.
sudo /home/robot/miniconda3/envs/zerith/bin/python scripts/end2end_grasp_pipeline.py \
    --mode sim \
    --no-move-chassis \
    --scene-dir "$SCENE_DIR" \
    --yolo-model "$YOLO_MODEL" \
    "$@"
