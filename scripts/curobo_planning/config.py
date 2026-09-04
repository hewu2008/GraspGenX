"""Planner tuning, grasp candidate file handling, and goal-set selection.

Pure numpy; builds on the frame helpers.  No cuRobo/torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import (
    filter_pose_workspace,
    grasps_world_to_tool_base,
    validate_grasp_poses,
)


@dataclass(frozen=True)
class CuroboPlannerConfig:
    device: str = "cuda:0"
    max_goalset: int = 16
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    collision_cache_mesh: int = 2
    collision_cache_cuboid: int = 8
    optimizer_collision_activation_distance: float = 0.01
    use_cuda_graph: bool = True
    warmup_iterations: int = 5
    workspace_bounds_base: tuple[float, float, float, float, float, float] | None = (
        -1.2,
        -1.2,
        -0.2,
        1.5,
        1.2,
        2.0,
    )
    random_seed: int = 123

    def __post_init__(self) -> None:
        if self.max_goalset < 1:
            raise ValueError("max_goalset must be positive")
        if self.num_ik_seeds < 1 or self.num_trajopt_seeds < 1:
            raise ValueError("CuRobo seed counts must be positive")


@dataclass(frozen=True)
class GraspCandidates:
    """Candidate grasp poses in the world frame, with scores and labels."""

    # (N, 4, 4) float64 homogeneous transforms from the world frame to each
    # candidate grasp/tool frame.
    poses_world: np.ndarray
    # (N,) float64 grasp scores; higher is better.  Candidate selection sorts by
    # these from highest to lowest confidence.
    confidence: np.ndarray
    # (N,) fixed-length strings tagging each candidate (e.g. an object label),
    # carried through to the selected motion and its artifacts.
    tags: np.ndarray
    # Path to the candidate file this batch was loaded from (verbatim, for
    # provenance/traceability).
    source_path: Path


def load_grasp_candidates(path: str | Path) -> GraspCandidates:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Grasp candidate file does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        if "grasps" not in data or "conf" not in data:
            raise ValueError(f"Grasp file must contain grasps and conf: {source}")
        poses = validate_grasp_poses(np.asarray(data["grasps"], dtype=np.float64))
        confidence = np.asarray(data["conf"], dtype=np.float64)
        tags = (
            np.asarray(data["tags"], dtype="U64")
            if "tags" in data
            else np.full(len(poses), "", dtype="U1")
        )
    if confidence.shape != (len(poses),) or tags.shape != (len(poses),):
        raise ValueError(
            f"Grasp conf/tags must each have shape ({len(poses)},): {source}"
        )
    if not np.isfinite(confidence).all():
        raise ValueError(f"Grasp confidence contains NaN or Inf: {source}")
    return GraspCandidates(poses, confidence, tags, source.resolve())


def select_goalset(
    candidates: GraspCandidates,
    *,
    max_goalset: int,
    world_T_base: np.ndarray,
    grasp_T_wrist: np.ndarray,
    workspace_bounds_base,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort top-K grasps and return poses, confidences, and original indices."""

    order = np.argsort(-candidates.confidence, kind="stable")[:max_goalset]
    # Convert to the planner's tool-at-base frame.
    # (N, 4, 4) float64 homogeneous transforms from the world frame to each
    # candidate grasp/tool frame.
    # (N, 4, 4) float64 homogeneous transforms from the robot base to the world frame.
    # (4, 4) float64 homogeneous transform from the planner's wrist frame to the grasp/tool base frame.
    poses_base = grasps_world_to_tool_base(
        candidates.poses_world[order], world_T_base, grasp_T_wrist
    )
    workspace_mask = filter_pose_workspace(poses_base, workspace_bounds_base)
    poses_base = poses_base[workspace_mask]
    confidence = candidates.confidence[order][workspace_mask]
    source_indices = order[workspace_mask]
    if len(poses_base) == 0:
        raise ValueError("No top-K grasp candidates remain inside the planning workspace")
    return poses_base, confidence, source_indices