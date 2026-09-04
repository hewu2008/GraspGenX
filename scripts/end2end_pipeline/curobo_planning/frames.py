"""Frame and quaternion conversions at the GraspGenX/cuRobo boundary.

Abstracted from the reviewed planning stack in
/home/robot/tanzhen/GraspGenX/scripts/end2end_pipeline/planning_frames.py,
keeping only the pieces the grasp planner needs (world->base->tool transform
chain, quaternion conventions, workspace filtering, and the gripper ``G_T_U``
base-rotation loader).  Pure numpy/scipy; no robot SDK dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .model import get_repo_root

ArmName = Literal["left", "right"]


# Fixed offset from the wrist frame controlled by ``setArm_high`` to the
# gripper's end-effector tool frame in the planning URDF.
WRIST_T_END_EFFECTOR = np.array(
    [
        [1.0, 0.0, 0.0, 0.1435],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def validate_transform(value: np.ndarray, *, name: str = "transform") -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return transform


def invert_transform(value: np.ndarray) -> np.ndarray:
    transform = validate_transform(value)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    return points @ transform[:3, :3].T + transform[:3, 3]


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4 or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must be finite with final dimension 4")
    return quaternion[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4 or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must be finite with final dimension 4")
    return quaternion[..., [1, 2, 3, 0]]


def matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotation_matrix, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("Rotation matrices must be finite with shape (..., 3, 3)")
    xyzw = Rotation.from_matrix(matrices).as_quat()
    return xyzw_to_wxyz(xyzw)


def wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(wxyz_to_xyzw(quaternion)).as_matrix()


def validate_grasp_poses(grasps: np.ndarray) -> np.ndarray:
    poses = np.asarray(grasps, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"grasps must have shape (K, 4, 4), got {poses.shape}")
    if len(poses) == 0:
        raise ValueError("At least one grasp pose is required")
    for index, pose in enumerate(poses):
        validate_transform(pose, name=f"grasps[{index}]")
    return poses


def grasp_world_to_tool_base(
    world_T_grasp: np.ndarray,
    world_T_base: np.ndarray,
    grasp_T_wrist: np.ndarray,
    wrist_T_end_effector: np.ndarray = WRIST_T_END_EFFECTOR,
) -> np.ndarray:
    """Compute ``B_T_E = inv(W_T_B) @ W_T_G @ G_T_U @ U_T_E``."""

    return (
        invert_transform(world_T_base)
        @ validate_transform(world_T_grasp, name="world_T_grasp")
        @ validate_transform(grasp_T_wrist, name="grasp_T_wrist")
        @ validate_transform(wrist_T_end_effector, name="wrist_T_end_effector")
    )


def grasps_world_to_tool_base(
    grasp_poses_world: np.ndarray,
    world_T_base: np.ndarray,
    grasp_T_wrist: np.ndarray,
    wrist_T_end_effector: np.ndarray = WRIST_T_END_EFFECTOR,
) -> np.ndarray:
    poses = validate_grasp_poses(grasp_poses_world)
    return np.stack(
        [
            grasp_world_to_tool_base(
                pose, world_T_base, grasp_T_wrist, wrist_T_end_effector
            )
            for pose in poses
        ],
        axis=0,
    )


def poses_to_curobo_arrays(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split (K,4,4) poses into cuRobo position (K,3) and wxyz quaternion (K,4)."""

    poses = validate_grasp_poses(poses)
    positions = poses[:, :3, 3].astype(np.float32)
    quaternions = matrix_to_wxyz(poses[:, :3, :3]).astype(np.float32)
    return positions, quaternions


def filter_pose_workspace(
    poses: np.ndarray,
    bounds: Iterable[float] | None,
) -> np.ndarray:
    """Return a boolean mask for end-effector origins inside base-frame bounds."""

    poses = validate_grasp_poses(poses)
    if bounds is None:
        return np.ones(len(poses), dtype=bool)
    bounds_array = np.asarray(tuple(bounds), dtype=np.float64)
    if bounds_array.shape != (6,):
        raise ValueError("Workspace bounds must be [xmin,ymin,zmin,xmax,ymax,zmax]")
    xyz = poses[:, :3, 3]
    return np.all(xyz >= bounds_array[:3], axis=1) & np.all(
        xyz <= bounds_array[3:], axis=1
    )


def _candidate_config_paths(root: Path, gripper_name: str) -> list[Path]:
    return [
        root
        / "gripper_descriptions"
        / "assets"
        / "x_grippers"
        / gripper_name
        / "config.json",
        root / "assets" / "x_grippers" / gripper_name / "config.json",
        root / "x_grippers" / gripper_name / "config.json",
        root / gripper_name / "config.json",
    ]


def resolve_gripper_config_path(
    arm: ArmName,
    gripper_config_dir: str | Path | None = None,
) -> Path:
    gripper_name = f"zerith_{arm}_gripper"
    roots: list[Path] = []
    if gripper_config_dir is not None:
        roots.append(Path(gripper_config_dir).expanduser())
    env_root = os.environ.get("GRASPGENX_GRIPPER_CFG_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(get_repo_root() / "ext" / "gripper_descriptions")

    attempted: list[Path] = []
    for root in roots:
        for path in _candidate_config_paths(root, gripper_name):
            attempted.append(path)
            if path.is_file():
                return path.resolve()
    attempted_text = "\n  ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        f"Cannot find {gripper_name}/config.json. Set "
        f"GRASPGENX_GRIPPER_CFG_DIR or --gripper-config-dir. Tried:\n  "
        f"{attempted_text}"
    )


def load_gripper_base_rotation(
    arm: ArmName,
    gripper_config_dir: str | Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Load the configured ``G_T_U`` base rotation for the selected gripper."""

    path = resolve_gripper_config_path(arm, gripper_config_dir)
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if "base_rotation" not in config:
        raise ValueError(f"Gripper config has no base_rotation: {path}")
    transform = validate_transform(
        np.asarray(config["base_rotation"], dtype=np.float64),
        name=f"base_rotation in {path}",
    )
    return transform, path


__all__ = [
    "WRIST_T_END_EFFECTOR",
    "filter_pose_workspace",
    "grasp_world_to_tool_base",
    "grasps_world_to_tool_base",
    "invert_transform",
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
]
