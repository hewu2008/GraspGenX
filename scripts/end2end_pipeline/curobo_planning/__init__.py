"""cuRobo planning stack package.

The external interface lives in :mod:`.facade` (:class:`CuroboPlanning`)::

    from end2end_pipeline.curobo_planning import CuroboPlanning

This ``__init__`` re-exports the stack's public API.  The submodules
(``model``, ``frames``, ``trajectory``, ``grasp_planner``) are implementation
details and may be reorganized without changing the facade.
"""

from __future__ import annotations

from .facade import CuroboPlanning
from .frames import (
    WRIST_T_END_EFFECTOR,
    grasps_world_to_tool_base,
    grasp_world_to_tool_base,
    poses_to_curobo_arrays,
    validate_grasp_poses,
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
from .constants import (
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    ZERITH_CUROBO_YAML,
    ZERITH_PLANNING_URDF,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from .model import build_single_arm_planning_config, load_curobo_config
from .trajectory import PlannedMotion, TrajectorySegment

__all__ = [
    # Facade (implemented in facade.py)
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
    "ZERITH_ACTIVE_JOINTS",
    "ZERITH_ARM_JOINTS",
    "ZERITH_CUROBO_YAML",
    "ZERITH_PLANNING_URDF",
    "ZERITH_SOFTWARE_POSITION_LIMITS",
    "build_single_arm_planning_config",
    "load_curobo_config",
    # Frame API
    "WRIST_T_END_EFFECTOR",
    "grasps_world_to_tool_base",
    "grasp_world_to_tool_base",
    "poses_to_curobo_arrays",
    "validate_grasp_poses",
    # Trajectory contracts
    "PlannedMotion",
    "TrajectorySegment",
]
