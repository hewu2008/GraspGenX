#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

hf download \
    --repo-type model \
    adithyamurali/GraspGenXModel \
    --local-dir $GRASPGENX_CHECKPOINT_DIR

hf download \
    --repo-type dataset \
    adithyamurali/gripper_descriptions \
    --local-dir $GRASPGENX_GRIPPER_CFG_DIR