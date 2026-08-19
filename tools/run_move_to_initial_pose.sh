#!/bin/bash

# Move H1 robot to its initial observation pose (waist + dual-arm pre-position).
# Positional args (all optional):
#   $1 waist_z         target waist Z (m), default 0.67
#   $2 waist_pitch     target waist pitch (rad), default 1.2
#   $3 hold            1=hold after arrival (default), 0=exit immediately
#
# Examples:
#   bash tools/run_move_to_initial_pose.sh
#   bash tools/run_move_to_initial_pose.sh 0.67 1.2 0

WAIST_Z=${1:-0.67}
WAIST_PITCH=${2:-1.2}
HOLD=${3:-1}

ARGS=(
    --waist_z "$WAIST_Z"
    --waist_pitch "$WAIST_PITCH"
)
if [ "$HOLD" = "1" ] || [ "$HOLD" = "true" ]; then
    ARGS+=(--hold)
else
    ARGS+=(--no-hold)
fi

python scripts/move_to_initial_pose.py "${ARGS[@]}"
