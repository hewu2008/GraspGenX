"""Launcher for replaying GraspGenX grasp poses on the H1 robot.

Run from the project root:

    python scripts/replay_grasps.py --scene-dir assets/zerith/real_scene/02 \
        [--grasps-dir <dir|single.npz>] [--top-grasps N] [--rounds R] \
        [--move-chassis | --no-move-chassis]
"""

import argparse
import os
import sys


def _setup_sys_path():
    """Make `scripts/`, the project root, and the SDK dirs importable.

    When run as ``python replay/replay_grasps.py``, Python prepends the package
    directory (``scripts/replay``) to ``sys.path``; drop it so the ``replay``
    name resolves to the package rather than to ``replay/replay.py``.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # scripts/replay
    scripts_dir = os.path.abspath(os.path.join(here, ".."))  # scripts
    root = os.path.abspath(os.path.join(scripts_dir, ".."))  # repo root
    for entry in list(sys.path):
        if os.path.abspath(entry) == here:
            sys.path.remove(entry)
    for _dir in (
        scripts_dir,
        root,
        os.path.join(root, "assets", "zerith", "sdk", "lib"),
        os.path.join(root, "assets", "zerith", "sdk", "proto"),
    ):
        if _dir not in sys.path:
            sys.path.insert(0, _dir)


def _parse_args():
    p = argparse.ArgumentParser(description="Replay GraspGenX grasp poses on H1.")
    p.add_argument("--scene-dir", required=True,
                   help="Scene dir providing grasps/ (default) or meta_data.json. "
                        "e.g. assets/zerith/real_scene/02.")
    p.add_argument("--grasps-dir", default=None,
                   help="Path to a directory of <gripper>/*.npz grasp files, or a "
                        "single .npz. Defaults to <scene-dir>/grasps.")
    p.add_argument("--top-grasps", type=int, default=1,
                   help="Per .npz file, how many highest-scoring grasps to execute "
                        "(default 1 = best grasp only).")
    p.add_argument("--rounds", type=int, default=1,
                   help="Replay the whole grasp plan this many times (default 1).")
    p.add_argument("--move-chassis", action=argparse.BooleanOptionalAction, default=False,
                   help="Drive the chassis to the workspace first (default: off).")
    p.add_argument("--chassis-dist", type=float, default=0.8,
                   help="Chassis distance (m) to travel when --move-chassis is set.")
    p.add_argument("--mode", choices=["highlevel", "curobo_lowlevel"], default="highlevel",
                   help="Execution backend: 'highlevel' (SDK setArm_high) or "
                        "'curobo_lowlevel' (cuRobo plan + LOW_LEVEL SDK). Default: highlevel.")
    p.add_argument("--fake", action="store_true",
                   help="curobo_lowlevel: drive a FakeSDKRobot (no hardware); "
                        "planning still requires GPU/CUDA.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_sys_path()
    common = dict(
        grasps_dir=args.grasps_dir,
        top_grasps=args.top_grasps,
        rounds=args.rounds,
    )
    if args.mode == "curobo_lowlevel":
        from replay.replay_curobo_lowlevel import run_curobo_lowlevel_replay
        raise SystemExit(
            run_curobo_lowlevel_replay(args.scene_dir, fake=args.fake, **common)
        )
    from replay.replay import run_replay
    raise SystemExit(
        run_replay(
            args.scene_dir,
            drive_chassis=args.move_chassis,
            chassis_dist=args.chassis_dist,
            **common,
        )
    )