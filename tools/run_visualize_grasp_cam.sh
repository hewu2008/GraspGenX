#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

python scripts/visualize_grasp_cam.py \
    --scene_dir assets/zerith/real_scene/02 \
    --pos 0.08100385011015943 0.03297408496282614 0.46304010716121163 \
    --euler_deg 3.673195729920793 -8.50719798997383 -25.67341101266362 \
    --out assets/zerith/real_scene/02/grasp_vis.png