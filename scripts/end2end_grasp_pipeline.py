"""Thin launcher for the modular end-to-end grasp pipeline.

Run from the project root:

    python scripts/end2end_grasp_pipeline.py [--move-chassis|--no-move-chassis]
"""

import argparse
import os
import sys

def _setup_sys_path():
    """Make the project root and the compiled robot SDK directories importable.

    `scripts/` is automatically on sys.path[0] when this file is run directly,
    so the `end2end_pipeline` package resolves on its own. The project root and
    the SDK directories still need to be added for the runtime dependencies:
      assets/zerith/sdk/lib/   -> lib_h1_sdk_python, camera_client
      assets/zerith/sdk/proto/ -> robot_pb2, robot_pb2_grpc (camera_client imports robot_pb2)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    for _dir in (
        root,
        os.path.join(root, "assets", "zerith", "sdk", "lib"),
        os.path.join(root, "assets", "zerith", "sdk", "proto"),
    ):
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

def _parse_args():
    parser = argparse.ArgumentParser(description="End-to-end grasp pipeline.")
    parser.add_argument(
        "--move-chassis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive the chassis to the workspace before grasping (default: enabled). "
             "Pass --no-move-chassis when the robot is already positioned at the workspace.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_sys_path()
    from end2end_pipeline.pipeline import main
    main(args)
