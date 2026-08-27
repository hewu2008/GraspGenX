#!/bin/bash

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

# Defaults: sim mode, no chassis motion. Override by appending --mode / --move-chassis.
# Tee stdout+stderr to <scene_dir>/run.log (sudo keeps root as owner, matching rgb/depth).
sudo -E /home/robot/miniconda3/envs/zerith_graspgen/bin/python scripts/end2end_grasp_pipeline.py \
    --mode real \
    --no-move-chassis \
    --scene-dir assets/zerith/real_scene/02 \
    --yolo-model assets/zerith/yolo/last_20260807_v0.pt \
    --visualize \
    2>&1 | tee assets/zerith/real_scene/02/run.log
# propagate the pipeline's exit code (not tee's)
exit ${PIPESTATUS[0]}
