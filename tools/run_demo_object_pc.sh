#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85 \
    --plot_top_mesh