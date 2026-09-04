"""Single-arm cuRobo V2 grasp planner (approach -> grasp) for the Zerith H1.

The GPU-backed :class:`CuroboGraspPlanner` runs per-candidate full-chain
planning with the reduced-DOF (``lock_joints``) compatibility shim for the
pinned cuRobo commit's ``plan_grasp``.  Robot-model config building lives in
``model``, tuning/candidates in ``config``, CPU post-processing in
``trajectory``, and failure statistics in ``diagnostics``.

Requires the vendored cuRobo at ``ext/curobo`` importable in the active
environment and a CUDA device for planner construction.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping
import uuid

import numpy as np
import torch

import curobo
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from .config import (
    CuroboPlannerConfig,
    GraspCandidates,
    select_goalset,
)
from .constants import (
    ZERITH_ACTIVE_JOINTS,
    ZERITH_ARM_JOINTS,
    ZERITH_ARM_TOOL_FRAME,
    ZERITH_CONTACT_LINKS,
    ZERITH_SOFTWARE_POSITION_LIMITS,
)
from .diagnostics import (
    CuroboPlanningError,
    _aggregate_ik_attempts,
    _failure_stage,
    _failure_stage_counts,
    _solver_result_summary,
    _tensor_any,
)
from .frames import poses_to_curobo_arrays
from .model import build_single_arm_planning_config
from .trajectory import (
    PlannedMotion,
    TrajectorySegment,
    to_numpy,
    trim_curobo_trajectory,
    validate_trajectory_limits,
)

from end2end_pipeline.logging_utils import get_logger


logger = get_logger(__name__)


def get_curobo_build_info() -> tuple[str, str]:
    module_path = Path(curobo.__file__).resolve()
    return str(curobo.__version__), str(module_path)


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