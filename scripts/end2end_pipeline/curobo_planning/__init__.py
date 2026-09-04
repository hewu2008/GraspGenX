"""cuRobo planning stack package.

The external interface lives in :mod:`.facade` (:class:`CuroboPlanning`)::

    from end2end_pipeline.curobo_planning import CuroboPlanning

This ``__init__`` only re-exports the stack's public names.  The submodules
(``model``, ``frames``, ``trajectory``, ``grasp_planner``) are implementation
details and may be reorganized without changing the facade.
"""

from __future__ import annotations

from .facade import CuroboPlanning
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
