"""Persist reproducible plan metadata, trajectories, and a review plot.

Pure numpy; consumed by the public entry point and debugging workflows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .constants import ZERITH_CUROBO_YAML
from .trajectory import PlannedMotion, TrajectorySegment, save_trajectory_plot


def _segment_payload(prefix: str, segment: TrajectorySegment) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        f"{prefix}_joint_names": np.asarray(segment.joint_names, dtype="U64"),
        f"{prefix}_position": segment.position,
        f"{prefix}_dt_s": np.asarray(segment.dt_s),
    }
    for name in ("velocity", "acceleration", "jerk"):
        value = getattr(segment, name)
        if value is not None:
            payload[f"{prefix}_{name}"] = value
    return payload


def save_plan_artifacts(
    motion: PlannedMotion,
    output_root: str | Path,
    *,
    robot_yaml_path: str | Path = ZERITH_CUROBO_YAML,
) -> Path:
    """Persist reproducible plan metadata, trajectories, and review plot."""

    output_dir = Path(output_root) / motion.plan_id
    output_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, np.ndarray] = {
        "selected_tool_pose_base": motion.selected_tool_pose_base,
        "goalset_index": np.asarray(motion.goalset_index, dtype=np.int64),
        "source_candidate_index": np.asarray(
            motion.source_candidate_index, dtype=np.int64
        ),
        "candidate_confidence": np.asarray(motion.candidate_confidence),
    }
    payload.update(_segment_payload("grasp", motion.grasp))
    np.savez_compressed(output_dir / "trajectory.npz", **payload)

    yaml_path = Path(robot_yaml_path)
    yaml_digest = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    summary = {
        "plan_id": motion.plan_id,
        "arm": motion.arm,
        "object_label": motion.object_label,
        "goalset_index": motion.goalset_index,
        "source_candidate_index": motion.source_candidate_index,
        "candidate_confidence": motion.candidate_confidence,
        "status": motion.status,
        "planning_time_s": motion.planning_time_s,
        "scene_digest": motion.scene_digest,
        "selected_tool_pose_base": motion.selected_tool_pose_base.tolist(),
        "curobo_version": motion.curobo_version,
        "curobo_commit": motion.curobo_commit,
        "robot_yaml": str(yaml_path.resolve()),
        "robot_yaml_sha256": yaml_digest,
        "grasp": {
            "dt_s": motion.grasp.dt_s,
            "waypoints": motion.grasp.waypoint_count,
        },
        "metadata": motion.metadata,
    }
    with (output_dir / "plan.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    save_trajectory_plot(
        [motion.grasp], output_dir / "trajectory.png"
    )
    return output_dir