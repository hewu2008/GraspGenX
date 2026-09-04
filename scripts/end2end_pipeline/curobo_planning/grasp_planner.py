"""Single-arm cuRobo V2 grasp planner (approach -> grasp) for the Zerith H1.

Abstracted from the reviewed planning stack in
/home/robot/tanzhen/GraspGenX/scripts/end2end_pipeline/curobo_planner.py.
Behavior is preserved, including:

* the reduced-DOF (``lock_joints``) compatibility shim for the pinned cuRobo
  commit's ``plan_grasp`` (scalar/horizon ``knot`` handling);
* sequential per-candidate full-chain planning with JSON-serializable
  failure diagnostics;
* trajectory trimming, limit validation, and artifact persistence.

Requires the vendored cuRobo at ``ext/curobo`` (commit
``EXPECTED_CUROBO_COMMIT``) importable in the active environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Mapping
import uuid

import numpy as np

from .frames import (
    filter_pose_workspace,
    grasps_world_to_tool_base,
    poses_to_curobo_arrays,
    validate_grasp_poses,
)
from .model import (
    ZERITH_ARM_JOINTS,
    ZERITH_ARM_TOOL_FRAME,
    ZERITH_CONTACT_LINKS,
    ZERITH_CUROBO_YAML,
    ZERITH_SOFTWARE_POSITION_LIMITS,
    build_single_arm_planning_config,
)
from .trajectory import (
    PlannedMotion,
    TrajectorySegment,
    save_trajectory_plot,
    to_numpy,
    trim_curobo_trajectory,
    validate_trajectory_limits,
)
from ..logging_utils import get_logger


EXPECTED_CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"
logger = get_logger(__name__)


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


def _find_git_commit(module_path: Path) -> str | None:
    for parent in (module_path, *module_path.parents):
        if (parent / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip() or None
            except Exception:
                return None
    return None


def get_curobo_build_info() -> tuple[str, str | None, str]:
    import curobo

    module_path = Path(curobo.__file__).resolve()
    return str(curobo.__version__), _find_git_commit(module_path), str(module_path)


def _tensor_any(value) -> bool:
    return value is not None and bool(value.any().item())


def _mask_summary(value) -> dict[str, object] | None:
    if value is None:
        return None
    flat = to_numpy(value).astype(bool, copy=False).reshape(-1)
    false_indices = np.flatnonzero(~flat).tolist()
    return {
        "total": int(flat.size),
        "true_count": int(np.count_nonzero(flat)),
        "false_indices": [int(index) for index in false_indices[:64]],
        "false_indices_truncated": len(false_indices) > 64,
    }


def _numeric_summary(value) -> dict[str, float | int] | None:
    if value is None:
        return None
    flat = to_numpy(value).astype(np.float64, copy=False).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"total": int(flat.size), "finite_count": 0}
    return {
        "total": int(flat.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _debug_info_summary(value) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {"keys": sorted(str(key) for key in value)}
    return {"type": type(value).__name__}


def _solver_result_summary(result) -> dict[str, object]:
    if result is None:
        return {"returned": False}
    summary: dict[str, object] = {
        "returned": True,
        "success": _mask_summary(getattr(result, "success", None)),
        "feasible": _mask_summary(getattr(result, "feasible", None)),
        "position_error": _numeric_summary(
            getattr(result, "position_error", None)
        ),
        "rotation_error": _numeric_summary(
            getattr(result, "rotation_error", None)
        ),
    }
    for name in ("solve_time", "total_time"):
        value = getattr(result, name, None)
        if isinstance(value, (int, float)):
            summary[name] = float(value)
    debug_info = _debug_info_summary(getattr(result, "debug_info", None))
    if debug_info is not None:
        summary["debug_info"] = debug_info
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

        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        self.arm = arm
        self.config = config
        self.full_start_position = full_start.copy()
        from .model import ZERITH_ACTIVE_JOINTS

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
        self.version, self.commit, self.module_path = get_curobo_build_info()
        if self.commit is not None and self.commit != EXPECTED_CUROBO_COMMIT:
            self.destroy()
            raise RuntimeError(
                "Installed CuRobo commit does not match the reviewed pin: "
                f"installed={self.commit}, expected={EXPECTED_CUROBO_COMMIT}"
            )
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
        original_graph_find_path = None
        graph_planner = None
        current_stage = {"name": None}
        stage_names = ("grasp_goalset", "approach", "grasp", "lift")
        stage_index = 0

        if diagnostics is not None and hasattr(self.planner, "plan_pose"):
            diagnostics.setdefault("stages", {})
            diagnostics.setdefault("graph_queries", [])
            diagnostics.setdefault("collision_endpoints", [])
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

            graph_planner = getattr(self.planner, "graph_planner", None)
            original_graph_find_path = getattr(graph_planner, "find_path", None)
            if original_graph_find_path is not None:

                def find_path_with_diagnostics(*args, **graph_kwargs):
                    captured_feasibility = []
                    original_feasibility = getattr(
                        graph_planner, "check_samples_feasibility", None
                    )
                    if original_feasibility is not None:

                        def capture_feasibility(samples):
                            mask = original_feasibility(samples)
                            captured_feasibility.append(
                                to_numpy(mask).astype(bool, copy=True).reshape(-1)
                            )
                            return mask

                        graph_planner.check_samples_feasibility = capture_feasibility
                    try:
                        graph_result = original_graph_find_path(*args, **graph_kwargs)
                    finally:
                        if original_feasibility is not None:
                            graph_planner.check_samples_feasibility = (
                                original_feasibility
                            )
                    record: dict[str, object] = {
                        "stage": current_stage["name"],
                        "success": _mask_summary(
                            getattr(graph_result, "success", None)
                        ),
                        "valid_query": bool(
                            getattr(graph_result, "valid_query", True)
                        ),
                        "debug_info": _debug_info_summary(
                            getattr(graph_result, "debug_info", "")
                            if hasattr(graph_result, "debug_info")
                            else None
                        ),
                    }
                    debug_text = str(getattr(graph_result, "debug_info", "") or "")
                    if (
                        "Start or End state in collision" in debug_text
                        and len(args) >= 2
                        and captured_feasibility
                    ):
                        endpoint_labels = []
                        try:
                            combined = captured_feasibility[0]
                            query_count = int(args[0].shape[0])
                            if combined.size != query_count * 2:
                                raise ValueError(
                                    "graph endpoint feasibility size mismatch: "
                                    f"got {combined.size}, expected {query_count * 2}"
                                )
                            paired = combined.reshape(query_count, 2)
                            start_mask = paired[:, 0]
                            end_mask = paired[:, 1]
                            start_summary = _mask_summary(start_mask)
                            end_summary = _mask_summary(end_mask)
                            record["start_endpoint_feasibility"] = start_summary
                            record["end_endpoint_feasibility"] = end_summary
                            if (
                                start_summary is not None
                                and start_summary["true_count"] < start_summary["total"]
                            ):
                                endpoint_labels.append("current_start")
                            if (
                                end_summary is not None
                                and end_summary["true_count"] < end_summary["total"]
                            ):
                                endpoint_labels.append(
                                    f"{current_stage['name'] or 'unknown'}_ik_goal"
                                )
                        except Exception as exc:
                            record["endpoint_diagnostic_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        record["collision_endpoints"] = endpoint_labels
                        known_endpoints = diagnostics["collision_endpoints"]
                        for endpoint in endpoint_labels:
                            if endpoint not in known_endpoints:
                                known_endpoints.append(endpoint)
                    diagnostics["graph_queries"].append(record)
                    return graph_result

                graph_planner.find_path = find_path_with_diagnostics

        kinematics.get_active_js = get_active_js_without_invalid_knot
        try:
            return self.planner.plan_grasp(**kwargs)
        finally:
            kinematics.get_active_js = original_get_active_js
            if original_plan_pose is not None:
                self.planner.plan_pose = original_plan_pose
            if original_graph_find_path is not None:
                graph_planner.find_path = original_graph_find_path

    def _make_start_state(self):
        from curobo.types import JointState

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
        from curobo.types import GoalToolPose, Pose

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
            collision_endpoints = sorted(
                {
                    str(endpoint)
                    for attempt in attempts
                    for endpoint in attempt.get("collision_endpoints", ())
                }
            )
            planning_diagnostics = {
                "strategy": "sequential_full_chain",
                "attempted_candidate_count": len(attempts),
                "filtered_candidate_count": total_candidates,
                "failure_stage_counts": stage_counts,
                "collision_endpoints": collision_endpoints,
                "candidate_attempts": attempts,
            }
            stage_text = ", ".join(
                f"{stage}={count}" for stage, count in sorted(stage_counts.items())
            ) or "unknown"
            endpoint_text = ",".join(collision_endpoints) or "none reported"
            raise CuroboPlanningError(
                f"CuRobo grasp planning failed for {object_label}: all "
                f"{len(attempts)} candidates failed full-chain planning "
                f"({stage_text}); collision_endpoints={endpoint_text}",
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
            curobo_commit=self.commit,
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
        "expected_curobo_commit": EXPECTED_CUROBO_COMMIT,
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


__all__ = [
    "CuroboGraspPlanner",
    "CuroboPlanningError",
    "CuroboPlannerConfig",
    "EXPECTED_CUROBO_COMMIT",
    "GraspCandidates",
    "get_curobo_build_info",
    "load_grasp_candidates",
    "save_plan_artifacts",
    "select_goalset",
]
