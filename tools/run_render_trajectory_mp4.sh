#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Low-res, textureless (fast — ~4–6x faster than textured on EGL):
export PYOPENGL_PLATFORM=egl

python end2end/render_trajectory_mp4.py \
  --trajectory end2end/runs/franka_single/trajectory.json \
  --output end2end/runs/franka_single/demo.mp4 \
  --resolution 320x240 \
  --no-texture