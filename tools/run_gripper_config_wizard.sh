#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

python scripts/gripper_config_wizard.py \
    --urdf ./assets/zerith/urdf/left_gripper.urdf \
    --name zerith_left_gripper \
    --port 8081

# python scripts/gripper_config_wizard.py \
#     --urdf $GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets/x_grippers/sharpa_wave/gripper.urdf \
#     --name sharpa_right \
#     --port 8081