#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Defaults: sim mode, no chassis motion. Override by appending --mode / --move-chassis.
sudo -E /home/robot/miniconda3/envs/zerith_graspgen/bin/python scripts/end2end_grasp_pipeline.py \
    --mode sim \
    --no-move-chassis \
    --scene-dir assets/zerith/real_scene/01 \
    --yolo-model assets/zerith/yolo/last_20260807_v0.pt \
    --visualize
