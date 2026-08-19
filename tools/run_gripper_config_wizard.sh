#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

LEFT_URDF=${1:-./assets/zerith/urdf/left_gripper.urdf}
RIGHT_URDF=${2:-./assets/zerith/urdf/right_gripper.urdf}
PORT=${3:-8081}

LEFT_NAME=$(basename "${LEFT_URDF%.*}")
RIGHT_NAME=$(basename "${RIGHT_URDF%.*}")

python scripts/gripper_config_wizard.py \
    --urdf "$LEFT_URDF" \
    --name "$LEFT_NAME" \
    --port "$PORT"

python scripts/gripper_config_wizard.py \
    --urdf "$RIGHT_URDF" \
    --name "$RIGHT_NAME" \
    --port "$PORT"