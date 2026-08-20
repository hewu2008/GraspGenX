#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

GRIPPER_NAME=${1:-zerith_left_gripper}

python scripts/demo_scene_pc.py \
    --sample_data_dir assets/zerith/real_scene \
    --collision_threshold 0.002 \
    --moe_z_offsets_cm "0,2" \
    --moe_obb_density dense \
    --top_down_only \
    --top_down_dot_threshold 0.85 \
    --gripper_name $GRIPPER_NAME
