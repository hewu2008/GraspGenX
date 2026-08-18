#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=true 

python end2end/e2e_grasp_demo.py \
  --robot_config end2end/robots/franka_panda.yaml \
  --env_config end2end/envs/single_bin_demo.yaml \
  --task clutter_pick_and_drop \
  --playback_mode dynamic \
  --no-viser \
  --num_grasps 200 \
  --topk 80 \
  --grasp_threshold 0.7 \
  --planner graspmoe \
  --seed 0 \
  --export-trajectory end2end/runs/franka_single/trajectory.json
