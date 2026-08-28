"""Thin launcher for the modular end-to-end grasp pipeline.

Run from the project root:

    python scripts/end2end_grasp_pipeline.py --scene-dir <dir> --yolo-model <pt> \
        [--mode sim|real] [--move-chassis|--no-move-chassis]
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
        "--mode",
        choices=["sim", "real"],
        default="real",
        help="sim: read RGB-D from the scene dir. real: capture RGB-D from the camera "
             "into the scene dir (and drive the robot). Detection + segmentation run in "
             "both modes (default: real).",
    )
    parser.add_argument(
        "--move-chassis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive the chassis to the workspace before grasping (default: enabled). "
             "Pass --no-move-chassis when the robot is already positioned at the workspace.",
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory holding (sim) or receiving (real) rgb.png / depth.npy / seg.png, "
             "e.g. assets/zerith/real_scene/00.",
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to the YOLO segmentation weights (.pt) used for detection + segmentation.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render grasp visualizations to PNG after grasp generation (default: enabled). "
             "Pass --no-visualize to skip image output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible grasp generation (default: 42). Pass -1 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_sys_path()
    from end2end_pipeline.pipeline import main
    main(args)
