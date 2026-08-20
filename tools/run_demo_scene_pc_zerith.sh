#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

GRIPPER_NAME=${1:-zerith_left_gripper}

python scripts/demo_scene_pc.py \
    --sample_data_dir assets/zerith/real_scene \
    --gripper_name $GRIPPER_NAME
