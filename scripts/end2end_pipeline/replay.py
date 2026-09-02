"""Replay grasp poses: repeatedly return to initial pose and execute grasps.

The replay targets a WRIST (hand) camera scene (e.g. assets/zerith/real_scene/
02_cam_left_wrist inferred from the grasp directory name), so it uses the
wrist-camera hand-eye chain -- NOT the head-camera IMU/FK chain.

    init robot                 -> see :func:`init_robot`
    move to initial pose       -> :func:`_return_to_initial_pose`
    input grasp poses          -> read GraspGenX ``grasps/*.npz`` (world frame)
    repeat grasp & go back     -> for each (gripper, label) top-K grasp:
                                      return to initial pose
                                      world grasp -> wrist-camera frame -> arm target
                                      grasp_object()  (approach/close/lift/place/release)
                                      return to initial pose

Coordinate chain (all 4x4, right-multiplied as composition):

    T_cam_world = compute_hand_camera_pose(robot, arm)      # wrist camera -> world
    T_obj_cam   = inv(T_cam_world) @ grasp4x4               # world grasp -> wrist-camera frame
    T_grasp_eef = CAM_TO_SDK_EEF_HAND @ T_obj_cam           # wrist cam -> current SDK eef
    (pos, quat) = _grasp_in_eef_to_sdk_target(T_grasp_eef)  # -> final SDK eef target

``compute_hand_camera_pose`` derives the wrist camera pose from IMU + waist +
the *current* arm pose (getHandRelative) plus the fixed URDF hand-eye offset
``CAM_TO_SDK_EEF_HAND``. Unlike the head path it does not rely on neck/waist
head joints -- the wrist camera is rigidly attached to the arm.

Because every loop iteration first returns the robot to the *same* initial pose
and does not move the chassis, the wrist camera-to-world transform is (nearly)
constant across iterations, so reusing a world-frame grasp is consistent. If the
chassis/waist/arm drift between iterations, re-plan instead.

NOTE: the hand-eye offset CAM_TO_SDK_EEF_HAND in grasp_executor is calibrated for
the LEFT wrist camera; right-wrist replays reuse it as-is unless a mirrored
calibration is added.

NOTE: grasp_executor.grasp_object() and robot_motion.move_arm_to_grasp() contain
``pdb.set_trace()`` breakpoints from the interactive tuning flow; when running this
replay unattended make sure they are removed or answered ('c') or the loop will pause.

Run from the project root, e.g.:
    sudo -E /home/robot/miniconda3/envs/zerith_graspgen/bin/python scripts/replay_grasps.py \
        --scene-dir assets/zerith/real_scene/02_cam_left_wrist
"""

import os

import numpy as np

from .config import (
    WAIST_NORMAL_Z,
    WAIST_PITCH,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_GRIPPER_MOTOR,
    RIGHT_GRIPPER_MOTOR,
    LEFT_GRIPPER_NAME,
    RIGHT_GRIPPER_NAME,
)
from .robot_motion import prepare_robot_posture, move_arm_to_ready_pose
from .grasp_executor import resolve_grasp_target_hand, grasp_object
from .camera_pose import compute_hand_camera_pose
from .logging_utils import get_logger

logger = get_logger(__name__)

# Initial-observation posture/app-ready targets, matching
# pipeline.approach_workspace / move_to_initial_pose.py defaults.
_READY_XYZ = [-0.1, 0.0, 0.30]
_READY_QUAT = [0.0, 0.0, 0.0, 1.0]

# Infer the arm + gripper motor from the ``zerith_<side>_gripper`` name used in the
# grasp sub-directory (grasps/<gripper>/<label>.npz).
_GRIPPER_SPECS = {
    LEFT_GRIPPER_NAME: (LEFT_ARM, LEFT_GRIPPER_MOTOR),
    RIGHT_GRIPPER_NAME: (RIGHT_ARM, RIGHT_GRIPPER_MOTOR),
}


def init_robot():
    """Connect to the robot, switch to high-level mode and init.

    Returns the connected H1Robot, or None on failure.
    """
    from lib_h1_sdk_python import H1Robot, MotorControlMode

    logger.info("[INIT] Instantiating robot and connecting...")
    robot = H1Robot()
    if not robot.robot_connect():
        logger.error("[INIT] Failed to connect to robot!")
        return None
    robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
    robot.robot_init()
    return robot


def move_chassis_forward(robot, dist, speed=0.2):
    """Drive the chassis straight by `dist` meters (optional, mirrors pipeline)."""
    import time

    dt = 0.2
    duration = abs(dist) / speed
    direction = 1 if dist >= 0 else -1
    velocity = speed * direction
    start_time = time.time()
    logger.info(f"[Chassis] Forward: {abs(dist):.2f} m @ {speed} m/s ({duration:.2f}s)")
    try:
        while time.time() - start_time < duration:
            loop_start = time.perf_counter()
            robot.setChassis_high(velocity, 0.0)
            elapsed = time.perf_counter() - loop_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
        robot.setChassis_high(0.0, 0.0)
    except KeyboardInterrupt:
        robot.setChassis_high(0.0, 0.0)
        raise
    logger.info("[Chassis] Move complete")


def _return_to_initial_pose(robot):
    """Restore waist Z/pitch + both arms to the ready pose (no chassis)."""
    prepare_robot_posture(robot, 0, 0, WAIST_NORMAL_Z, WAIST_PITCH)
    move_arm_to_ready_pose(robot, [0.0, 0.0, 0.0], _READY_QUAT, _READY_XYZ, _READY_QUAT)


def collect_grasp_plan(scene_dir, grasps_dir=None, top_grasps=1):
    """Scan ``<grasps_dir>/{gripper}/*.npz`` and order the grasps to execute.

    Each entry is ``(gripper_name, label, grasp_idx, grasp4x4)``, selected by top
    score within each file. ``grasps_dir`` defaults to ``<scene_dir>/grasps``. If
    ``grasps_dir`` points at a single ``.npz`` it is treated as a one-file plan.
    """
    grasps_dir = grasps_dir or os.path.join(scene_dir, "grasps")
    if os.path.isfile(grasps_dir):
        files = [grasps_dir]
        root = os.path.dirname(grasps_dir)
    else:
        if not os.path.isdir(grasps_dir):
            raise FileNotFoundError(f"Grasps dir not found: {grasps_dir}")
        files = [
            os.path.join(dp, f)
            for dp, _, fns in os.walk(grasps_dir)
            for f in fns
            if f.endswith(".npz")
        ]
        root = grasps_dir

    plan = []
    for npz in sorted(files):
        rel = os.path.relpath(npz, root)
        parts = rel.split(os.sep)
        gripper = parts[0] if len(parts) > 1 else "_"
        label = os.path.splitext(parts[-1])[0]
        data = np.load(npz)
        grasps = data["grasps"]
        conf = data.get("conf", None)
        idxs = np.argsort(-conf)[: max(1, int(top_grasps))] if conf is not None else [0]
        for i in idxs:
            plan.append((gripper, label, int(i), np.asarray(grasps[i], dtype=np.float64)))
    logger.info(f"[Replay] {len(plan)} grasp(s) to execute from {grasps_dir}")
    return plan


def world_grasp_to_hand_cam(robot, arm, grasp4x4):
    """Map a world-frame grasp pose into the `arm` wrist-camera frame.

    Uses compute_hand_camera_pose (IMU + waist + current arm pose + fixed hand-eye
    offset) as the single world <-> wrist-camera reference. Returns the 4x4
    wrist-camera-frame target, or None on failure.
    """
    T_cam_world = compute_hand_camera_pose(robot, arm=arm)
    if T_cam_world is None:
        logger.error("[Replay] Failed to compute hand camera pose; cannot map grasp.")
        return None
    T_world_cam = np.linalg.inv(T_cam_world)
    return T_world_cam @ grasp4x4


def run_replay(
    scene_dir,
    grasps_dir=None,
    top_grasps=1,
    drive_chassis=False,
    chassis_dist=0.8,
    rounds=1,
):
    """Initialize, then repeatedly: back-to-initial pose -> grasp -> back-to-initial."""
    robot = init_robot()
    if robot is None:
        return 1

    try:
        if drive_chassis:
            move_chassis_forward(robot, chassis_dist)
        _return_to_initial_pose(robot)

        plan = collect_grasp_plan(scene_dir, grasps_dir=grasps_dir, top_grasps=top_grasps)
        if not plan:
            logger.warning("[Replay] Empty grasp plan; nothing to execute.")
            return 0

        for r in range(max(1, int(rounds))):
            logger.info(f"[Replay] ======== round {r + 1}/{max(1, int(rounds))} ========")
            for entry_i, (gripper, label, gidx, grasp4x4) in enumerate(plan, start=1):
                if gripper not in _GRIPPER_SPECS:
                    logger.error(
                        f"[Replay] Unknown gripper '{gripper}'; skipping "
                        f"(grasp {entry_i})."
                    )
                    continue
                arm, motor = _GRIPPER_SPECS[gripper]
                logger.info(
                    f"[Replay] [{arm}] {label}[{gidx}] ({entry_i}/{len(plan)}): "
                    f"pos={grasp4x4[:3, 3].tolist()}"
                )

                _return_to_initial_pose(robot)
                T_obj_cam = world_grasp_to_hand_cam(robot, arm, grasp4x4)
                if T_obj_cam is None:
                    continue
                target_pos, target_quat = resolve_grasp_target_hand(robot, T_obj_cam)
                if target_pos is None:
                    logger.error("[Replay] resolve_grasp_target_hand failed; skipping grasp.")
                    continue

                grasp_object(robot, target_pos, target_quat, arm=arm, gripper_motor=motor)
                _return_to_initial_pose(robot)

        logger.info("[Replay] All rounds complete.")
        return 0
    except KeyboardInterrupt:
        logger.warning("[Replay] Ctrl+C received; safe shutdown.")
        return 130
    finally:
        logger.info("[Cleanup] Releasing robot control...")
        if "robot" in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()