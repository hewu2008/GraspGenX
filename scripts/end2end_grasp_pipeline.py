"""Thin launcher for the modular end-to-end grasp pipeline.

Run from the project root:

    python scripts/end2end_grasp_pipeline.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# scripts/ is automatically on sys.path[0] when this file is run directly, so
# the `end2end_pipeline` package resolves. Make the project root and the
# compiled robot SDK directories importable as well:
#   assets/zerith/sdk/lib/  -> lib_h1_sdk_python, camera_client
#   assets/zerith/sdk/proto/ -> robot_pb2, robot_pb2_grpc (camera_client imports robot_pb2)
for _dir in (
    _ROOT,
    os.path.join(_ROOT, "assets", "zerith", "sdk", "lib"),
    os.path.join(_ROOT, "assets", "zerith", "sdk", "proto"),
):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from end2end_pipeline.pipeline import main

if __name__ == "__main__":
    main()
