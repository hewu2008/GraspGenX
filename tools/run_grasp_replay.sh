#!/bin/bash
# Replay GraspGenX grasp poses on the Zerith H1 robot.
#
# Usage:
#   ./tools/run_grasp_replay.sh [SCENE_DIR] [--extra flags...]
#
# Defaults to assets/zerith/real_scene/02_cam_left_wrist. Append extra flags for the underlying
# replay_grasps.py (e.g. --rounds 3, --top-grasps 2, --grasps-dir <npz>).
#
# Execution backend via GRASP_REPLAY_MODE (default: highlevel):
#   GRASP_REPLAY_MODE=curobo_lowlevel ./tools/run_grasp_replay.sh

export GRASPGENX_CHECKPOINT_DIR=/home/robot/hewu/model_zoo/GraspGenXModel
export GRASPGENX_GRIPPER_CFG_DIR=/home/robot/hewu/model_zoo/gripper_descriptions

SCENE_DIR=${1:-assets/zerith/real_scene/02_cam_left_wrist}
shift 2>/dev/null || true

# Tee stdout+stderr to the same run.log location the pipeline uses.
sudo -E /home/robot/miniconda3/envs/zerith_graspgen/bin/python scripts/replay/replay_grasps.py \
    --scene-dir "$SCENE_DIR" \
    --mode highlevel \
    --no-move-chassis \
    --rounds 10 \
    --top-grasps 1 \
    "$@" \
    2>&1 | tee "$SCENE_DIR/replay.log"
# propagate the pipeline's exit code (not tee's)
exit ${PIPESTATUS[0]}