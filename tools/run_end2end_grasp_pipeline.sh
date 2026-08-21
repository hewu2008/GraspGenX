#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Forward CLI args to the launcher, e.g. --no-move-chassis to skip chassis motion.
sudo /home/robot/miniconda3/envs/zerith/bin/python scripts/end2end_grasp_pipeline.py \
    --no-move-chassis
