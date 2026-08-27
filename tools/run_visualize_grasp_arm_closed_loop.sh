#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

 python scripts/visualize_grasp_arm_closed_loop.py \
        --scene_dir assets/zerith/real_scene/02 \
        --log assets/zerith/real_scene/02/run.log \
        --out_port 9090