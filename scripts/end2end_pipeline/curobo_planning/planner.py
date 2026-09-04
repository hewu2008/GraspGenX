"""Single-arm cuRobo V2 grasp planner (approach -> grasp) for the Zerith H1.

Merged module: the robot-model factory, the CPU trajectory data contracts and
post-processing, and the GPU grasp planner.  Behavior is unchanged, including:

* the reduced-DOF (``lock_joints``) compatibility shim for the pinned cuRobo
  commit's ``plan_grasp`` (scalar/horizon ``knot`` handling);
* sequential per-candidate full-chain planning with JSON-serializable
  per-candidate/per-stage failure statistics;
* trajectory trimming, limit validation, and artifact persistence.

Requires the vendored cuRobo at ``ext/curobo`` importable in the active
environment and a CUDA device for planner construction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Mapping
import uuid

import numpy as np
import torch

import curobo
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

import yaml

from .constants import (
    REPO_ROOT,
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    ZERITH_ARM_TOOL_FRAME,
    ZERITH_CONTACT_LINKS,
    ZERITH_CUROBO_YAML,
    ZERITH_LOCKED_JOINTS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from .frames import (
    filter_pose_workspace,
    grasps_world_to_tool_base,
    poses_to_curobo_arrays,
    validate_grasp_poses,
)
from ..logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trajectory data contracts and CPU post-processing (pure numpy)
# ---------------------------------------------------------------------------

ArmName = Literal["left", "right"]


@dataclass(frozen=True)
class TrajectorySegment:
    """A trimmed, time-parameterized joint trajectory on the CPU."""

    name: str
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray | None
    acceleration: np.ndarray | None
    jerk: np.ndarray | None
    dt_s: float

    @property
    def waypoint_count(self) -> int:
        return int(self.position.shape[0])


@dataclass(frozen=True)
class PlannedMotion:
    """Serializable output of cuRobo approach/grasp planning."""

    plan_id: str
    arm: ArmName
    object_label: str
    goalset_index: int
    source_candidate_index: int
    candidate_confidence: float
    approach: TrajectorySegment
    grasp: TrajectorySegment
    status: str
    planning_time_s: float
    scene_digest: str
    selected_tool_pose_base: np.ndarray
    curobo_version: str
    curobo_commit: str | None
    metadata: dict[str, object] = field(default_factory=dict)


def to_numpy(value) -> np.ndarray | None:
    """Convert a torch tensor / array-like to numpy (None passes through)."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_dt(dt) -> float:
    value = to_numpy(dt)
    if value is None or value.size == 0:
        raise ValueError("CuRobo interpolated trajectory has no dt")
    flattened = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all() or np.any(flattened <= 0):
        raise ValueError(f"Invalid CuRobo trajectory dt: {flattened}")
    if not np.allclose(flattened, flattened[0], rtol=1e-5, atol=1e-8):
        raise ValueError("Planning expects a fixed dt within each CuRobo segment")
    return float(flattened[0])


def _trajectory_matrix(value, *, name: str, waypoint_count: int) -> np.ndarray | None:
    array = to_numpy(value)
    if array is None:
        return None
    matrix = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    if matrix.shape[0] < waypoint_count:
        raise ValueError(f"CuRobo {name} buffer shorter than its last timestep")
    return matrix[:waypoint_count]


def trim_curobo_trajectory(
    joint_state,
    interpolated_last_tstep,
    *,
    name: str,
) -> TrajectorySegment:
    """Trim CuRobo's preallocated tail using its exclusive last-timestep count."""

    if joint_state is None:
        raise ValueError(f"CuRobo returned no {name} trajectory")
    position_buffer = to_numpy(joint_state.position)
    if position_buffer is None or position_buffer.ndim < 2:
        raise ValueError(f"CuRobo returned an invalid {name} position buffer")
    buffer_count = int(position_buffer.reshape(-1, position_buffer.shape[-1]).shape[0])
    if interpolated_last_tstep is None:
        waypoint_count = buffer_count
    else:
        value = to_numpy(interpolated_last_tstep)
        if value is None or value.size == 0:
            raise ValueError(f"CuRobo returned an empty {name} last_tstep")
        waypoint_count = int(value.reshape(-1)[0])
    if waypoint_count < 1:
        raise ValueError(f"CuRobo returned an empty {name} trajectory")

    joint_names = tuple(str(x) for x in (joint_state.joint_names or ()))
    position = _trajectory_matrix(
        joint_state.position, name=f"{name}.position", waypoint_count=waypoint_count
    )
    assert position is not None
    if len(joint_names) != position.shape[1]:
        raise ValueError(
            f"CuRobo {name} joint_names has {len(joint_names)} entries but "
            f"trajectory has {position.shape[1]} columns"
        )
    return TrajectorySegment(
        name=name,
        joint_names=joint_names,
        position=position,
        velocity=_trajectory_matrix(
            joint_state.velocity,
            name=f"{name}.velocity",
            waypoint_count=waypoint_count,
        ),
        acceleration=_trajectory_matrix(
            joint_state.acceleration,
            name=f"{name}.acceleration",
            waypoint_count=waypoint_count,
        ),
        jerk=_trajectory_matrix(
            joint_state.jerk, name=f"{name}.jerk", waypoint_count=waypoint_count
        ),
        dt_s=_scalar_dt(joint_state.dt),
    )


def validate_trajectory_limits(
    segment: TrajectorySegment,
    *,
    position_limits: Mapping[str, tuple[float, float]],
    velocity_limits: Mapping[str, float] | None = None,
    acceleration_limits: Mapping[str, float] | None = None,
    jerk_limits: Mapping[str, float] | None = None,
) -> None:
    """Fail if a resampled trajectory crosses any supplied named limit."""

    arrays = (
        ("velocity", segment.velocity, velocity_limits),
        ("acceleration", segment.acceleration, acceleration_limits),
        ("jerk", segment.jerk, jerk_limits),
    )
    for column, joint_name in enumerate(segment.joint_names):
        if joint_name not in position_limits:
            raise ValueError(f"Missing position limit for {joint_name}")
        lower, upper = position_limits[joint_name]
        values = segment.position[:, column]
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"{segment.name}: {joint_name} crosses position limits")
        for derivative_name, derivative, limits in arrays:
            if limits is None:
                continue
            if derivative is None:
                raise ValueError(
                    f"{segment.name}: no {derivative_name} available for validation"
                )
            if joint_name not in limits:
                raise ValueError(f"Missing {derivative_name} limit for {joint_name}")
            if np.any(np.abs(derivative[:, column]) > limits[joint_name]):
                raise ValueError(
                    f"{segment.name}: {joint_name} crosses {derivative_name} limits"
                )


def save_trajectory_plot(
    segments: list[TrajectorySegment],
    output_path: str | Path,
) -> Path | None:
    """Save position/velocity/acceleration/jerk plots for plan review.

    Returns the written path, or None when matplotlib is unavailable.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not segments:
        raise ValueError("At least one trajectory segment is required")
    joint_names = segments[0].joint_names
    if any(segment.joint_names != joint_names for segment in segments):
        raise ValueError("All plotted trajectory segments must share joint order")
    derivatives = (
        ("position", "position"),
        ("velocity", "velocity"),
        ("acceleration", "acceleration"),
        ("jerk", "jerk"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    time_offset = 0.0
    for segment in segments:
        timestamps = time_offset + np.arange(segment.waypoint_count) * segment.dt_s
        for axis, (label, attribute) in zip(axes, derivatives):
            values = getattr(segment, attribute)
            if values is not None:
                axis.plot(timestamps, values)
            axis.set_ylabel(label)
            axis.axvline(time_offset, color="black", alpha=0.2)
        time_offset = float(timestamps[-1])
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(joint_names, loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


# ---------------------------------------------------------------------------
# cuRobo robot-model factory (17-DOF Zerith -> 7-DOF single arm)
# ---------------------------------------------------------------------------

def _resolve_repo_path(value: str, repo_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def load_curobo_config(
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a portable Zerith config ready to pass to cuRobo.

    Relative paths inside the YAML are interpreted relative to this repository,
    not the caller's current directory.  Returns a fresh copy of the inner
    ``robot_cfg`` mapping with absolute ``urdf_path`` and ``asset_root_path``.
    """

    config_path = Path(yaml_path) if yaml_path is not None else ZERITH_CUROBO_YAML
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    if not isinstance(document, dict) or not isinstance(document.get("robot_cfg"), dict):
        raise ValueError(f"Invalid cuRobo robot config (missing robot_cfg): {config_path}")

    robot_cfg = deepcopy(document["robot_cfg"])
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError(f"Invalid cuRobo robot config (missing kinematics): {config_path}")

    for key in ("urdf_path", "asset_root_path"):
        value = kinematics.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid cuRobo robot config ({key} is missing): {config_path}")
        kinematics[key] = _resolve_repo_path(value, REPO_ROOT)

    return robot_cfg


def build_single_arm_planning_config(
    arm: Literal["left", "right"],
    full_joint_position: Mapping[str, float],
    locked_joint_position: Mapping[str, float] | None = None,
    yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a seven-DOF planning config with every other joint locked.

    Locked values come from the synchronized robot snapshot.  Collision links,
    collision spheres, and self-collision configuration are intentionally left
    intact so the stationary waist and opposite arm remain collision geometry.
    """

    if arm not in ZERITH_ARM_JOINTS:
        raise ValueError(f"Unsupported target arm: {arm!r}")
    missing = set(ZERITH_ACTIVE_JOINTS) - set(full_joint_position)
    extra = set(full_joint_position) - set(ZERITH_ACTIVE_JOINTS)
    if missing or extra:
        raise ValueError(
            f"full_joint_position mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name, value in full_joint_position.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Joint {name} has a non-numeric lock value")

    robot_cfg = load_curobo_config(yaml_path)
    kinematics = robot_cfg["kinematics"]
    cspace = kinematics.get("cspace")
    if not isinstance(cspace, dict):
        raise ValueError("Zerith config has no cspace mapping")
    source_names = list(cspace.get("joint_names", ()))
    if source_names != list(ZERITH_ACTIVE_JOINTS):
        raise ValueError("Zerith cspace joint order differs from the public 17-joint order")

    active_names = list(ZERITH_ARM_JOINTS[arm])
    active_indices = [source_names.index(name) for name in active_names]
    for key in (
        "default_joint_position",
        "cspace_distance_weight",
        "null_space_weight",
    ):
        values = cspace.get(key)
        if not isinstance(values, list) or len(values) != len(source_names):
            raise ValueError(f"Zerith cspace {key} must have 17 values")
        if key == "default_joint_position":
            cspace[key] = [float(full_joint_position[name]) for name in active_names]
        else:
            cspace[key] = [values[index] for index in active_indices]
    cspace["joint_names"] = active_names

    lock_joints = deepcopy(kinematics.get("lock_joints", {}))
    if not isinstance(lock_joints, dict):
        raise ValueError("Zerith kinematics lock_joints must be a mapping")
    if locked_joint_position is not None:
        supplied_locked = {
            str(name): float(value) for name, value in locked_joint_position.items()
        }
        if set(supplied_locked) != set(ZERITH_LOCKED_JOINTS):
            raise ValueError("locked_joint_position keys differ from the Zerith model")
        if not all(math.isfinite(value) for value in supplied_locked.values()):
            raise ValueError("locked_joint_position contains NaN or Inf")
        lock_joints.update(supplied_locked)
    for name in ZERITH_ACTIVE_JOINTS:
        if name not in active_names:
            lock_joints[name] = float(full_joint_position[name])
    kinematics["lock_joints"] = lock_joints
    kinematics["tool_frames"] = [ZERITH_ARM_TOOL_FRAME[arm]]
    return robot_cfg


# ---------------------------------------------------------------------------
# Planner configuration, candidates, and goal-set selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CuroboPlannerConfig:
    device: str = "cuda:0"
    max_goalset: int = 16
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    collision_cache_mesh: int = 2
    collision_cache_cuboid: int = 8
    optimizer_collision_activation_distance: float = 0.01
    approach_axis: str = "x"
    approach_offset_m: float = 0.10
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
        if self.approach_offset_m <= 0:
            raise ValueError("approach_offset_m must be positive")
        if self.approach_axis not in ("x", "y", "z"):
            raise ValueError("approach_axis must be x, y, or z")


@dataclass(frozen=True)
class GraspCandidates:
    poses_world: np.ndarray
    confidence: np.ndarray
    tags: np.ndarray
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


def get_curobo_build_info() -> tuple[str, str]:
    module_path = Path(curobo.__file__).resolve()
    return str(curobo.__version__), str(module_path)


# ---------------------------------------------------------------------------
# Failure diagnostics (per-candidate / per-stage statistics)
# ---------------------------------------------------------------------------

def _tensor_any(value) -> bool:
    return value is not None and bool(value.any().item())


def _mask_summary(value) -> dict[str, object] | None:
    if value is None:
        return None
    flat = to_numpy(value).astype(bool, copy=False).reshape(-1)
    return {
        "total": int(flat.size),
        "true_count": int(np.count_nonzero(flat)),
    }


def _solver_result_summary(result) -> dict[str, object]:
    if result is None:
        return {"returned": False}
    summary: dict[str, object] = {
        "returned": True,
        "success": _mask_summary(getattr(result, "success", None)),
        "feasible": _mask_summary(getattr(result, "feasible", None)),
    }
    for name in ("solve_time", "total_time"):
        value = getattr(result, name, None)
        if isinstance(value, (int, float)):
            summary[name] = float(value)
    return summary


def _aggregate_ik_attempts(
    stage_diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    attempts = (
        list(stage_diagnostics.get("ik_attempts", ()))
        if isinstance(stage_diagnostics, Mapping)
        else []
    )
    success_counts = []
    feasible_counts = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        success = attempt.get("success")
        feasible = attempt.get("feasible")
        if isinstance(success, Mapping):
            success_counts.append(int(success.get("true_count", 0)))
        if isinstance(feasible, Mapping):
            feasible_counts.append(int(feasible.get("true_count", 0)))
    return {
        "attempt_count": len(attempts),
        "any_success": any(count > 0 for count in success_counts),
        "best_success_seed_count": max(success_counts, default=0),
        "best_feasible_seed_count": max(feasible_counts, default=0),
    }


def _failure_stage(result) -> str:
    if result is None:
        return "planner"
    status = str(getattr(result, "status", "") or "").lower()
    if "goalset" in status or "goal set" in status:
        return "grasp_goalset"
    if "approach" in status and "failed" in status:
        return "approach"
    if "grasp pose" in status and "failed" in status:
        return "grasp"
    if "lift" in status and "failed" in status:
        return "lift"
    if not _tensor_any(getattr(result, "approach_success", None)):
        return "approach"
    if not _tensor_any(getattr(result, "grasp_success", None)):
        return "grasp"
    return "planner"


def _failure_stage_counts(attempts: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.get("success") is True:
            continue
        stage = str(attempt.get("failure_stage", "unknown"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts


class CuroboPlanningError(RuntimeError):
    """Fail-closed planning error with JSON-serializable candidate diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


# ---------------------------------------------------------------------------
# GPU grasp planner
# ---------------------------------------------------------------------------

class CuroboGraspPlanner:
    """One warmed planner whose locked joints match one captured snapshot."""

    def __init__(
        self,
        arm,
        full_start_position: np.ndarray,
        config: CuroboPlannerConfig,
        *,
        locked_joint_position: Mapping[str, float] | None = None,
    ) -> None:
        if arm not in ZERITH_ARM_JOINTS:
            raise ValueError(f"Unsupported target arm: {arm!r}")
        full_start = np.asarray(full_start_position, dtype=np.float64)
        if full_start.shape != (17,):
            raise ValueError("full_start_position must contain all 17 model joints")
        if not np.isfinite(full_start).all():
            raise ValueError("full_start_position contains NaN or Inf")

        self.arm = arm
        self.config = config
        self.full_start_position = full_start.copy()
        self.full_start_by_name = dict(zip(ZERITH_ACTIVE_JOINTS, full_start.tolist()))
        self.robot_cfg = build_single_arm_planning_config(
            arm,
            self.full_start_by_name,
            locked_joint_position=locked_joint_position,
        )
        self.device_cfg = DeviceCfg(
            device=torch.device(config.device), dtype=torch.float32
        )
        planner_cfg = MotionPlannerCfg.create(
            robot=self.robot_cfg,
            scene_model=None,
            collision_cache={
                "mesh": config.collision_cache_mesh,
                "cuboid": config.collision_cache_cuboid,
            },
            self_collision_check=True,
            device_cfg=self.device_cfg,
            num_ik_seeds=config.num_ik_seeds,
            num_trajopt_seeds=config.num_trajopt_seeds,
            max_goalset=config.max_goalset,
            optimizer_collision_activation_distance=(
                config.optimizer_collision_activation_distance
            ),
            use_cuda_graph=config.use_cuda_graph,
            random_seed=config.random_seed,
        )
        self.planner = MotionPlanner(planner_cfg)
        self.version, self.module_path = get_curobo_build_info()
        warmup_ok = self.planner.warmup(
            enable_graph=config.use_cuda_graph,
            num_warmup_iterations=config.warmup_iterations,
        )
        if not warmup_ok:
            self.destroy()
            raise RuntimeError("CuRobo warmup failed")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.planner.joint_names)

    @property
    def tool_frame(self) -> str:
        return ZERITH_ARM_TOOL_FRAME[self.arm]

    def update_world(self, scene) -> None:
        self.planner.update_world(scene)

    def _plan_grasp_with_locked_joint_compat(
        self,
        *,
        diagnostics: dict[str, object] | None = None,
        **kwargs,
    ):
        """Call V2 plan_grasp with the reviewed commit's reduced-DOF fix.

        Commit 057a96f carries a scalar/horizon ``JointState.knot`` through the
        approach result, then asks ``Kinematics.get_active_js`` to reindex it as
        if it had one column per joint.  It also squeezes the only batch axis
        before feeding the state to the next ``plan_pose`` call.  Together these
        cause a CUDA index assertion or an always-failed final segment for a
        reduced-DOF/lock_joints planner.  ``knot`` is optimizer metadata, not a
        robot state derivative; drop it only when its final dimension cannot
        match joint_names and restore the required 2-D batch axis.
        """

        kinematics = self.planner.kinematics
        original_get_active_js = kinematics.get_active_js

        def get_active_js_without_invalid_knot(full_js):
            knot = getattr(full_js, "knot", None)
            joint_names = getattr(full_js, "joint_names", None)
            if (
                knot is not None
                and joint_names is not None
                and knot.shape[-1] != len(joint_names)
            ):
                full_js.knot = None
            active_js = original_get_active_js(full_js)
            if active_js.position.ndim == 1:
                active_js = active_js.unsqueeze(0)
            return active_js

        original_plan_pose = None
        current_stage = {"name": None}
        stage_names = ("grasp_goalset", "approach", "grasp", "lift")
        stage_index = 0

        if diagnostics is not None and hasattr(self.planner, "plan_pose"):
            diagnostics.setdefault("stages", {})
            original_plan_pose = self.planner.plan_pose

            def plan_pose_with_diagnostics(*args, **plan_kwargs):
                nonlocal stage_index
                stage = (
                    stage_names[stage_index]
                    if stage_index < len(stage_names)
                    else f"plan_pose_{stage_index}"
                )
                stage_index += 1
                stage_record: dict[str, object] = {"ik_attempts": []}
                diagnostics["stages"][stage] = stage_record
                previous_stage = current_stage["name"]
                current_stage["name"] = stage

                ik_solver = getattr(self.planner, "ik_solver", None)
                original_solve_pose = getattr(ik_solver, "solve_pose", None)
                if original_solve_pose is not None:

                    def solve_pose_with_diagnostics(*ik_args, **ik_kwargs):
                        try:
                            ik_result = original_solve_pose(*ik_args, **ik_kwargs)
                        except Exception as exc:
                            stage_record["ik_attempts"].append(
                                {
                                    "returned": False,
                                    "exception_type": type(exc).__name__,
                                    "exception": str(exc),
                                }
                            )
                            raise
                        stage_record["ik_attempts"].append(
                            _solver_result_summary(ik_result)
                        )
                        return ik_result

                    ik_solver.solve_pose = solve_pose_with_diagnostics
                try:
                    plan_result = original_plan_pose(*args, **plan_kwargs)
                    stage_record["trajectory_result"] = _solver_result_summary(
                        plan_result
                    )
                    return plan_result
                except Exception as exc:
                    stage_record["exception_type"] = type(exc).__name__
                    stage_record["exception"] = str(exc)
                    raise
                finally:
                    if original_solve_pose is not None:
                        ik_solver.solve_pose = original_solve_pose
                    current_stage["name"] = previous_stage

            self.planner.plan_pose = plan_pose_with_diagnostics

        kinematics.get_active_js = get_active_js_without_invalid_knot
        try:
            return self.planner.plan_grasp(**kwargs)
        finally:
            kinematics.get_active_js = original_get_active_js
            if original_plan_pose is not None:
                self.planner.plan_pose = original_plan_pose

    def _make_start_state(self):
        active_start = np.asarray(
            [self.full_start_by_name[name] for name in self.joint_names],
            dtype=np.float32,
        )
        return JointState.from_numpy(
            joint_names=list(self.joint_names),
            position=active_start[None, :],
            velocity=None,
            device_cfg=self.device_cfg,
        )

    def _make_single_candidate_goal(self, pose_base: np.ndarray):
        position, quaternion = poses_to_curobo_arrays(pose_base[None, ...])
        goal_pose = Pose.from_numpy(
            position=position,
            quaternion=quaternion,
            device_cfg=self.device_cfg,
        )
        return GoalToolPose.from_poses(
            {self.tool_frame: goal_pose},
            ordered_tool_frames=[self.tool_frame],
            num_goalset=1,
        )

    def _validate_segment_limits(self, segment: TrajectorySegment) -> None:
        limits = self.planner.kinematics.get_joint_limits()
        velocity = np.asarray(to_numpy(limits.velocity), dtype=np.float64)
        acceleration = np.asarray(to_numpy(limits.acceleration), dtype=np.float64)
        jerk = np.asarray(to_numpy(limits.jerk), dtype=np.float64)
        position_limits = {
            name: ZERITH_SOFTWARE_POSITION_LIMITS[name]
            for name in self.joint_names
        }
        velocity_limits = {
            name: float(np.max(np.abs(velocity[:, i])))
            for i, name in enumerate(self.joint_names)
        }
        acceleration_limits = {
            name: float(np.max(np.abs(acceleration[:, i])))
            for i, name in enumerate(self.joint_names)
        }
        jerk_limits = {
            name: float(np.max(np.abs(jerk[:, i])))
            for i, name in enumerate(self.joint_names)
        }
        validate_trajectory_limits(
            segment,
            position_limits=position_limits,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
        )

    def plan_grasp(
        self,
        candidates: GraspCandidates,
        *,
        world_T_base: np.ndarray,
        grasp_T_wrist: np.ndarray,
        scene_digest: str,
        object_label: str,
        metadata: dict[str, object] | None = None,
    ) -> PlannedMotion:
        poses_base, confidences, source_indices = select_goalset(
            candidates,
            max_goalset=self.config.max_goalset,
            world_T_base=world_T_base,
            grasp_T_wrist=grasp_T_wrist,
            workspace_bounds_base=self.config.workspace_bounds_base,
        )
        started = time.monotonic()
        attempts: list[dict[str, object]] = []
        result = None
        goal_index = None
        total_candidates = len(poses_base)
        for candidate_index, pose_base in enumerate(poses_base):
            source_index = int(source_indices[candidate_index])
            candidate_diagnostics: dict[str, object] = {
                "attempt_index": candidate_index,
                "goalset_index": candidate_index,
                "source_candidate_index": source_index,
                "confidence": float(confidences[candidate_index]),
                "tag": str(candidates.tags[source_index]),
            }
            logger.debug(
                "[Plan][Candidate %d/%d] %s (cand=%d, conf=%.3f, tag=%s); planning approach->grasp",
                candidate_index + 1,
                total_candidates,
                object_label,
                source_index,
                confidences[candidate_index],
                candidates.tags[source_index],
            )
            try:
                candidate_result = self._plan_grasp_with_locked_joint_compat(
                    diagnostics=candidate_diagnostics,
                    grasp_poses=self._make_single_candidate_goal(pose_base),
                    current_state=self._make_start_state(),
                    grasp_approach_axis=self.config.approach_axis,
                    grasp_approach_offset=-self.config.approach_offset_m,
                    grasp_approach_in_tool_frame=True,
                    plan_approach_to_grasp=True,
                    plan_grasp_to_lift=False,
                    disable_collision_links=list(ZERITH_CONTACT_LINKS[self.arm]),
                )
            except Exception as exc:
                candidate_diagnostics.update(
                    {
                        "success": False,
                        "failure_stage": "internal_error",
                        "status": f"{type(exc).__name__}: {exc}",
                    }
                )
                attempts.append(candidate_diagnostics)
                planning_diagnostics = {
                    "strategy": "sequential_full_chain",
                    "attempted_candidate_count": len(attempts),
                    "filtered_candidate_count": total_candidates,
                    "failure_stage_counts": _failure_stage_counts(attempts),
                    "candidate_attempts": attempts,
                }
                logger.error(
                    "[Plan][Candidate %d/%d] %s (cand=%d) "
                    "failed stage=internal_error reason=%s; aborting",
                    candidate_index + 1,
                    total_candidates,
                    object_label,
                    source_index,
                    candidate_diagnostics["status"],
                )
                raise CuroboPlanningError(
                    f"CuRobo internal error while planning candidate {source_index} "
                    f"for {object_label}: {type(exc).__name__}: {exc}",
                    planning_diagnostics,
                ) from exc

            candidate_ok = (
                candidate_result is not None
                and _tensor_any(candidate_result.success)
                and _tensor_any(candidate_result.approach_success)
                and _tensor_any(candidate_result.grasp_success)
            )
            candidate_status = (
                "planner returned no result"
                if candidate_result is None
                else str(candidate_result.status)
            )
            candidate_diagnostics["status"] = candidate_status
            candidate_diagnostics["result_flags"] = {
                "success": _tensor_any(
                    getattr(candidate_result, "success", None)
                ),
                "approach_success": _tensor_any(
                    getattr(candidate_result, "approach_success", None)
                ),
                "grasp_success": _tensor_any(
                    getattr(candidate_result, "grasp_success", None)
                ),
            }
            approach_ik = _aggregate_ik_attempts(
                candidate_diagnostics.get("stages", {}).get("approach")
                if isinstance(candidate_diagnostics.get("stages"), Mapping)
                else None
            )
            candidate_diagnostics["approach_ik"] = approach_ik
            if not candidate_ok:
                stage = _failure_stage(candidate_result)
                candidate_diagnostics["success"] = False
                candidate_diagnostics["failure_stage"] = stage
                attempts.append(candidate_diagnostics)
                logger.warning(
                    "[Plan] Candidate %d/%d (conf=%.3f) "
                    "failed at '%s' (%s); trying next candidate",
                    candidate_index + 1,
                    total_candidates,
                    confidences[candidate_index],
                    stage,
                    candidate_status,
                )
                continue

            candidate_diagnostics["success"] = True
            candidate_diagnostics["failure_stage"] = None
            attempts.append(candidate_diagnostics)
            result = candidate_result
            goal_index = candidate_index
            logger.info(
                "[Plan] Candidate %d/%d (conf=%.3f) "
                "full chain succeeded after %d attempt(s)",
                candidate_index + 1,
                total_candidates,
                confidences[candidate_index],
                len(attempts),
            )
            break

        wall_time = time.monotonic() - started
        if result is None or goal_index is None:
            stage_counts = _failure_stage_counts(attempts)
            planning_diagnostics = {
                "strategy": "sequential_full_chain",
                "attempted_candidate_count": len(attempts),
                "filtered_candidate_count": total_candidates,
                "failure_stage_counts": stage_counts,
                "candidate_attempts": attempts,
            }
            stage_text = ", ".join(
                f"{stage}={count}" for stage, count in sorted(stage_counts.items())
            ) or "unknown"
            raise CuroboPlanningError(
                f"CuRobo grasp planning failed for {object_label}: all "
                f"{len(attempts)} candidates failed full-chain planning "
                f"({stage_text})",
                planning_diagnostics,
            )
        if result.goalset_index is None:
            diagnostics = {
                "strategy": "sequential_full_chain",
                "attempted_candidate_count": len(attempts),
                "filtered_candidate_count": total_candidates,
                "failure_stage_counts": _failure_stage_counts(attempts),
                "candidate_attempts": attempts,
            }
            raise CuroboPlanningError(
                "CuRobo succeeded without returning goalset_index", diagnostics
            )
        internal_goal_index = int(result.goalset_index.reshape(-1)[0].item())
        if internal_goal_index != 0:
            diagnostics = {
                "strategy": "sequential_full_chain",
                "attempted_candidate_count": len(attempts),
                "filtered_candidate_count": total_candidates,
                "failure_stage_counts": _failure_stage_counts(attempts),
                "candidate_attempts": attempts,
            }
            raise CuroboPlanningError(
                "CuRobo returned a non-zero goalset index for a single-candidate "
                f"attempt: {internal_goal_index}",
                diagnostics,
            )

        raw_approach = result.approach_interpolated_trajectory
        raw_grasp = result.grasp_interpolated_trajectory
        if (
            raw_approach is not None
            and tuple(getattr(raw_approach, "joint_names", ())) != self.joint_names
        ):
            raw_approach = self.planner.kinematics.get_active_js(raw_approach)
        if (
            raw_grasp is not None
            and tuple(getattr(raw_grasp, "joint_names", ())) != self.joint_names
        ):
            raw_grasp = self.planner.kinematics.get_active_js(raw_grasp)

        approach = trim_curobo_trajectory(
            raw_approach,
            result.approach_interpolated_last_tstep,
            name="approach",
        )
        grasp = trim_curobo_trajectory(
            raw_grasp,
            result.grasp_interpolated_last_tstep,
            name="grasp",
        )
        if approach.joint_names != self.joint_names or grasp.joint_names != self.joint_names:
            raise RuntimeError("CuRobo trajectory joint order changed unexpectedly")
        self._validate_segment_limits(approach)
        self._validate_segment_limits(grasp)

        planner_time = wall_time
        output_metadata = dict(metadata or {})
        output_metadata.update(
            {
                "input_candidate_count": int(len(candidates.poses_world)),
                "filtered_goalset_count": int(len(poses_base)),
                "candidate_attempt_strategy": "sequential_full_chain",
                "candidate_attempts": attempts,
                "selected_candidate_tag": str(
                    candidates.tags[source_indices[goal_index]]
                ),
            }
        )
        return PlannedMotion(
            plan_id=uuid.uuid4().hex,
            arm=self.arm,
            object_label=object_label,
            goalset_index=goal_index,
            source_candidate_index=int(source_indices[goal_index]),
            candidate_confidence=float(confidences[goal_index]),
            approach=approach,
            grasp=grasp,
            status=str(result.status),
            planning_time_s=planner_time,
            scene_digest=scene_digest,
            selected_tool_pose_base=poses_base[goal_index],
            curobo_version=self.version,
            curobo_commit=None,
            metadata=output_metadata,
        )

    def destroy(self) -> None:
        planner = getattr(self, "planner", None)
        if planner is not None:
            planner.destroy()
            self.planner = None

    def __enter__(self) -> "CuroboGraspPlanner":
        return self

    def __exit__(self, *_exc) -> None:
        self.destroy()


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
    payload.update(_segment_payload("approach", motion.approach))
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
        "approach": {
            "dt_s": motion.approach.dt_s,
            "waypoints": motion.approach.waypoint_count,
        },
        "grasp": {
            "dt_s": motion.grasp.dt_s,
            "waypoints": motion.grasp.waypoint_count,
        },
        "metadata": motion.metadata,
    }
    with (output_dir / "plan.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    save_trajectory_plot(
        [motion.approach, motion.grasp], output_dir / "trajectory.png"
    )
    return output_dir