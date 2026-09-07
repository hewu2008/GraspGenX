"""Shared helpers for the grasp replays (no end2end_pipeline dependency).

Hosts the pure-plan-reading code used by all three replay backends
(highlevel / curobo_lowlevel / pinocchio_lowlevel) so the low-level backends do
not need to import ``replay_sdk_highlevel`` (which pulls in the high-level
pipeline stack).
"""

from __future__ import annotations

import os

import numpy as np

from curobo_planning.logging_utils import get_logger

logger = get_logger(__name__)


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
