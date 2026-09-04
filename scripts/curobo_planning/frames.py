"""Frame and quaternion conversions at the GraspGenX/cuRobo boundary.

Provides the world->base->tool transform chain, quaternion conventions, and
workspace filtering used by the grasp planner.  Pure numpy/scipy; no robot SDK
or cuRobo/torch dependency.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import WRIST_T_END_EFFECTOR


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


def matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotation_matrix, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("Rotation matrices must be finite with shape (..., 3, 3)")
    xyzw = Rotation.from_matrix(matrices).as_quat()
    # scipy returns xyzw; cuRobo expects wxyz.
    return xyzw[..., [3, 0, 1, 2]]


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
    """Compute ``B_T_E = inv(W_T_B) @ W_T_G @ G_T_U @ U_T_E``.

    Walks the grasp pose back up the frame chain to the robot base:
    world -> grasp -> gripper tool -> wrist -> end-effector, each ``X_T_Y``
    mapping a point in frame Y to frame X.  Inner indices cancel, leaving
    ``B_T_E``: the end-effector pose expressed in the base frame.

    Chain, left to right:
      inv(world_T_base) = B_T_W       -> world points to base frame
      world_T_grasp     = W_T_G       -> world frame to the grasp pose
      grasp_T_wrist     = G_T_U       -> grasp/gripper tool to the wrist
      wrist_T_end_effector = U_T_E    -> wrist to the end-effector tool frame
    """

    # B_T_E = B_T_W @ W_T_G @ G_T_U @ U_T_E, applied right to left:
    return (
        # B_T_W: base <- world (inverse of base-in-world).
        invert_transform(world_T_base)
        # W_T_G: world <- grasp pose (candidate's pose in the world).
        @ validate_transform(world_T_grasp, name="world_T_grasp")
        # G_T_U: grasp <- gripper tool frame (per-candidate GraspGenX pose).
        @ validate_transform(grasp_T_wrist, name="grasp_T_wrist")
        # U_T_E: wrist <- end-effector tool origin (default +X 0.1435 m).
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
