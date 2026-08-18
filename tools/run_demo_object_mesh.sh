#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/banana.obj \
    --mesh_scale 1.0 \
    --gripper_name inspire_hand \
    --grasp_threshold -1.0 --return_topk --topk_num_grasps 100 --plot_top_mesh