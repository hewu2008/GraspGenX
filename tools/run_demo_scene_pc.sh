#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

GRIPPER_NAME=${1:-robotiq_2f_85}

python scripts/demo_scene_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name $GRIPPER_NAME
