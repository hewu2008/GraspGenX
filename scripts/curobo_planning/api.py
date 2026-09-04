"""Public API: single external interface for the cuRobo planning stack.

External callers should import :class:`CuroboPlanning` from here::

    from curobo_planning.api import CuroboPlanning

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
API is safe on machines without CUDA.  Importing this module does load cuRobo
(via ``planner``), so it still requires the ``zerith_graspgen`` environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from .artifacts import save_plan_artifacts
from .config import CuroboPlannerConfig, GraspCandidates
from .constants import ZERITH_ACTIVE_JOINTS, ZERITH_CUROBO_YAML
from .frames import grasp_world_to_tool_base
from .planner import CuroboGraspPlanner
from .trajectory import PlannedMotion


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
        """Plan the trajectory directly to the best grasp candidate."""

        return self.planner.plan(
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