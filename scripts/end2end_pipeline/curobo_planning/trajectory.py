"""cuRobo trajectory data contracts and CPU-side post-processing.

Abstracted from the reviewed planning stack in
/home/robot/tanzhen/GraspGenX/scripts/end2end_pipeline/
(planning_types.py + the trajectory helpers of trajectory_executor.py),
keeping only what is needed to consume cuRobo ``plan_grasp`` output:
trajectory trimming, limit validation, and review plots.
Pure numpy; no cuRobo/torch import required at module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np


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


__all__ = [
    "ArmName",
    "PlannedMotion",
    "TrajectorySegment",
    "save_trajectory_plot",
    "to_numpy",
    "trim_curobo_trajectory",
    "validate_trajectory_limits",
]
