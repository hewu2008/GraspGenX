#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Per-object grasps: green = collision-free, red = colliding; light-blue
# mesh = top-confidence grasp. Default is fully-observed (target included).
export PYOPENGL_PLATFORM=egl 

python end2end/visualize_scene_grasps.py \
  --env_config end2end/envs/franka_clutter3_demo.yaml \
  --robot_config end2end/robots/franka_panda.yaml \
  --threshold 0.7 \
  --moe_obb_density dense \
  --show_top_grasp_mesh 1 \
  --port 8090

# Debug the bin's collision shape vs its visual mesh:
python end2end/visualize_collision_vs_visual.py \
  --env_config end2end/envs/franka_clutter3_demo.yaml \
  --port 8090
