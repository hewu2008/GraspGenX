"""Facade for the standalone cuRobo planning stack.

External callers should import from this package only::

    from end2end_pipeline.curobo_planning import CuroboPlanning

    planning = CuroboPlanning("left", full_joint_position_by_name)
    try:
        motion = planning.plan(
            candidates,
            world_T_base=world_T_base,
            grasp_T_wrist=grasp_T_wrist,
            scene_digest=scene_digest,
            object_label=object_label,
        )
        planning.save_artifacts(motion, output_root)
    finally:
        planning.close()

The GPU-backed planner is created lazily on first use, so constructing the
facade is safe on machines without CUDA.  The submodules (``model``,
``frames``, ``trajectory``, ``grasp_planner``) are implementation details;
their public names are re-exported below for advanced use but may be
reorganized without changing this facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from .frames import (
    WRIST_T_END_EFFECTOR,
    grasps_world_to_tool_base,
    grasp_world_to_tool_base,
    load_gripper_base_rotation,
    matrix_to_wxyz,
    poses_to_curobo_arrays,
    resolve_gripper_config_path,
    transform_points,
    validate_grasp_poses,
    validate_transform,
    wxyz_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from .grasp_planner import (
    EXPECTED_CUROBO_COMMIT,
    CuroboGraspPlanner,
    CuroboPlannerConfig,
    CuroboPlanningError,
    GraspCandidates,
    get_curobo_build_info,
    load_grasp_candidates,
    save_plan_artifacts,
    select_goalset,
)
from .model import (
    ACTIVE_JOINTS,
    DEFAULT_JOINT_POSITION,
    LOCKED_JOINTS,
    TOOL_FRAMES,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    ZERITH_ARM_TOOL_FRAME,
    ZERITH_CONTACT_LINKS,
    ZERITH_CUROBO_YAML,
    ZERITH_DEFAULT_JOINT_POSITION,
    ZERITH_LOCKED_JOINTS,
    ZERITH_PLANNING_URDF,
    ZERITH_SOFTWARE_POSITION_LIMITS,
    ZERITH_TOOL_FRAMES,
    build_single_arm_planning_config,
    get_repo_root,
    load_curobo_config,
)
from .trajectory import (
    PlannedMotion,
    TrajectorySegment,
    hermite_resample,
    save_trajectory_plot,
    stretch_trajectory_time,
    to_numpy,
    trim_curobo_trajectory,
    validate_trajectory_limits,
)

__all__ = [
    # Facade
    "CuroboPlanning",
    # Planner API
    "CuroboGraspPlanner",
    "CuroboPlannerConfig",
    "CuroboPlanningError",
    "EXPECTED_CUROBO_COMMIT",
    "GraspCandidates",
    "get_curobo_build_info",
    "load_grasp_candidates",
    "save_plan_artifacts",
    "select_goalset",
    # Model API
    "ACTIVE_JOINTS",
    "DEFAULT_JOINT_POSITION",
    "LOCKED_JOINTS",
    "TOOL_FRAMES",
    "ZERITH_ACTIVE_JOINTS",
    "ZERITH_ARM_JOINTS",
    "ZERITH_ARM_TOOL_FRAME",
    "ZERITH_CONTACT_LINKS",
    "ZERITH_CUROBO_YAML",
    "ZERITH_DEFAULT_JOINT_POSITION",
    "ZERITH_LOCKED_JOINTS",
    "ZERITH_PLANNING_URDF",
    "ZERITH_SOFTWARE_POSITION_LIMITS",
    "ZERITH_TOOL_FRAMES",
    "build_single_arm_planning_config",
    "get_repo_root",
    "load_curobo_config",
    # Frame API
    "WRIST_T_END_EFFECTOR",
    "grasps_world_to_tool_base",
    "grasp_world_to_tool_base",
    "load_gripper_base_rotation",
    "matrix_to_wxyz",
    "poses_to_curobo_arrays",
    "resolve_gripper_config_path",
    "transform_points",
    "validate_grasp_poses",
    "validate_transform",
    "wxyz_to_matrix",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
    # Trajectory API
    "PlannedMotion",
    "TrajectorySegment",
    "hermite_resample",
    "save_trajectory_plot",
    "stretch_trajectory_time",
    "to_numpy",
    "trim_curobo_trajectory",
    "validate_trajectory_limits",
]


class CuroboPlanning:
    """Single entry point for cuRobo single-arm grasp planning.

    Wraps planner lifecycle (lazy GPU initialization), frame conversion, and
    artifact persistence behind one object.  One instance plans for one arm
    with one captured joint snapshot, mirroring ``CuroboGraspPlanner``.
    """

    def __init__(
        self,
        arm: Literal["left", "right"],
        full_joint_position: Mapping[str, float] | np.ndarray,
        *,
        locked_joint_position: Mapping[str, float] | None = None,
        config: CuroboPlannerConfig | None = None,
        yaml_path: str | Path | None = None,
    ) -> None:
        self.arm = arm
        self.full_start_position = self._normalize_joint_position(
            full_joint_position
        )
        self.locked_joint_position = locked_joint_position
        self.config = config if config is not None else CuroboPlannerConfig()
        self.yaml_path = yaml_path
        self._planner: CuroboGraspPlanner | None = None

    @staticmethod
    def _normalize_joint_position(
        value: Mapping[str, float] | np.ndarray,
    ) -> np.ndarray:
        """Accept either a name->angle mapping or a 17-vector in model order."""

        if isinstance(value, Mapping):
            missing = [name for name in ZERITH_ACTIVE_JOINTS if name not in value]
            if missing:
                raise ValueError(f"full_joint_position missing joints: {missing}")
            return np.asarray(
                [float(value[name]) for name in ZERITH_ACTIVE_JOINTS],
                dtype=np.float64,
            )
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (len(ZERITH_ACTIVE_JOINTS),):
            raise ValueError(
                "full_joint_position sequence must contain all "
                f"{len(ZERITH_ACTIVE_JOINTS)} model joints in order"
            )
        return vector

    @property
    def planner(self) -> CuroboGraspPlanner:
        """GPU-backed planner, created on first access (requires CUDA)."""

        if self._planner is None:
            self._planner = CuroboGraspPlanner(
                self.arm,
                self.full_start_position,
                self.config,
                locked_joint_position=self.locked_joint_position,
            )
        return self._planner

    @property
    def planner_created(self) -> bool:
        return self._planner is not None

    def plan(
        self,
        candidates: GraspCandidates,
        *,
        world_T_base: np.ndarray,
        grasp_T_wrist: np.ndarray,
        scene_digest: str = "",
        object_label: str = "",
        metadata: dict[str, object] | None = None,
    ) -> PlannedMotion:
        """Plan the approach->grasp trajectory for the best grasp candidate."""

        return self.planner.plan_grasp(
            candidates,
            world_T_base=world_T_base,
            grasp_T_wrist=grasp_T_wrist,
            scene_digest=scene_digest,
            object_label=object_label,
            metadata=metadata,
        )

    def update_world(self, scene) -> None:
        """Replace the planner collision world (no-op until planner exists)."""

        if self._planner is not None:
            self._planner.update_world(scene)

    def grasp_to_tool_base(
        self,
        world_T_grasp: np.ndarray,
        world_T_base: np.ndarray,
        grasp_T_wrist: np.ndarray,
    ) -> np.ndarray:
        """Convert a world grasp pose into the planner's tool-at-base frame."""

        return grasp_world_to_tool_base(
            world_T_grasp,
            world_T_base,
            grasp_T_wrist,
        )

    def save_artifacts(
        self,
        motion: PlannedMotion,
        output_root: str | Path,
        *,
        robot_yaml_path: str | Path | None = None,
    ) -> Path:
        """Persist plan metadata, trajectories, and review plot."""

        return save_plan_artifacts(
            motion,
            output_root,
            robot_yaml_path=(
                robot_yaml_path
                if robot_yaml_path is not None
                else (self.yaml_path if self.yaml_path is not None else ZERITH_CUROBO_YAML)
            ),
        )

    def close(self) -> None:
        """Release the GPU planner, if one was created."""

        if self._planner is not None:
            self._planner.destroy()
            self._planner = None

    def __enter__(self) -> "CuroboPlanning":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
