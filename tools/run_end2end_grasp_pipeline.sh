#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Forward any extra CLI args to the launcher, e.g. --no-move-chassis.
sudo /home/robot/miniconda3/envs/zerith/bin/python scripts/end2end_grasp_pipeline.py \
    --mode sim \
    --no-move-chassis \
    --yolo-model assets/zerith/yolo/last_20260807_v0.pt
