#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# python scripts/gripper_config_wizard.py \
#     --urdf /home/robot/hewu/alg-product/Zerith_Model/ZERITH_H1_PRO_URDF/urdf/ZR_H1PRO-1.2.00.H.V4.3_URDF_2025.12.02.urdf \
#     --name zerith_left_gripper \
#     --port 8081

python scripts/gripper_config_wizard.py \
    --urdf $GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets/x_grippers/sharpa_wave/gripper.urdf \
    --name sharpa_right \
    --port 8081