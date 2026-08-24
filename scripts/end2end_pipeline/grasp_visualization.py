# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save grasp visualization results to PNG images using pyrender headless renderer.

Provides functions to render 3D grasp scenes and save as images,
as an alternative to interactive viser visualization.
"""

import os
from typing import Dict, List, Optional, Tuple

# Must be set before importing OpenGL/pyrender: PyOpenGL binds to a platform
# (GLX vs EGL) the first time it is imported. Setting it later (e.g. at render
# time) has no effect, which would cause EGL platform init to fail with
# "'GLXPlatform' object has no attribute 'EGL'". Force EGL and headless for
# this module since it only renders offscreen.
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("PYGLET_HEADLESS", "true")

import numpy as np
import pyrender
import trimesh
from PIL import Image


def _setup_headless():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "true")


def _filter_scene_points(
    xyz: np.ndarray, rgb: np.ndarray, bounds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter scene points to a reasonable world-coordinate range.

    Args:
        xyz: (N, 3) points in world frame
        rgb: (N, 3) colors
        bounds: optional [[xmin,ymin,zmin],[xmax,ymax,zmax]] filter

    Returns:
        Filtered (xyz, rgb)
    """
    if bounds is not None:
        mask = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=1)
        return xyz[mask], rgb[mask]
    return xyz, rgb


def _make_point_sphere_mesh(
    points: np.ndarray, colors: np.ndarray, radius: float = 0.003
) -> trimesh.Trimesh:
    """Create a single merged mesh of colored spheres at each point."""
    if len(points) == 0:
        return trimesh.Trimesh()

    sphere_geo = trimesh.creation.icosphere(radius=radius, subdivisions=1)
    verts_per_sphere = len(sphere_geo.vertices)
    faces_per_sphere = len(sphere_geo.faces)

    total_verts = len(points) * verts_per_sphere
    total_faces = len(points) * faces_per_sphere

    all_verts = np.zeros((total_verts, 3), dtype=np.float32)
    all_faces = np.zeros((total_faces, 3), dtype=np.uint32)
    all_colors = np.zeros((total_verts, 3), dtype=np.uint8)

    for i, (pt, color) in enumerate(zip(points, colors)):
        v_start = i * verts_per_sphere
        v_end = v_start + verts_per_sphere
        f_start = i * faces_per_sphere
        f_end = f_start + faces_per_sphere

        all_verts[v_start:v_end] = sphere_geo.vertices + pt
        all_faces[f_start:f_end] = sphere_geo.faces + v_start
        all_colors[v_start:v_end] = color

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=all_colors)
    return mesh


def _make_bbox_wireframe(
    center: np.ndarray, half_extent: np.ndarray, rotation: np.ndarray
) -> List[trimesh.Trimesh]:
    """Create 12 thin cylinders forming a wireframe bounding box."""
    hx, hy, hz = half_extent
    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center

    lines = []
    for i, j in edges:
        p1 = transform[:3, :3] @ corners_local[i] + transform[:3, 3]
        p2 = transform[:3, :3] @ corners_local[j] + transform[:3, 3]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=0.003, height=length)
        z_axis = np.array([0, 0, 1.0])
        direction_norm = direction / length
        axis = np.cross(z_axis, direction_norm)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
            rot_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            cyl.apply_transform(rot_matrix)
        cyl.apply_translation((p1 + p2) / 2)
        lines.append(cyl)

    return lines


def _scene_to_origin(
    scene_xyz: Optional[np.ndarray] = None,
    obj_pcs: Optional[Dict[str, np.ndarray]] = None,
    grasps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Compute scene center and characteristic size."""
    all_pts = []
    if scene_xyz is not None and len(scene_xyz) > 0:
        all_pts.append(scene_xyz)
    if obj_pcs is not None:
        for pc in obj_pcs.values():
            if len(pc) > 0:
                all_pts.append(pc)
    if grasps is not None and len(grasps) > 0:
        all_pts.append(grasps[:, :3, 3])

    if not all_pts:
        return np.zeros(3), 0.5

    all_pts = np.vstack(all_pts)
    center = np.median(all_pts, axis=0)
    extent = np.percentile(all_pts.max(axis=0) - all_pts.min(axis=0), 80)
    char_size = max(float(extent), 0.3)
    return center, char_size


def _compute_camera_pose(
    scene_center: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Compute a camera pose looking at scene_center.

    pyrender convention: camera looks down -Z in camera frame.
    """
    x = scene_center[0] + distance * np.cos(elevation) * np.cos(azimuth)
    y = scene_center[1] + distance * np.cos(elevation) * np.sin(azimuth)
    z = scene_center[2] + distance * np.sin(elevation)

    pos = np.array([x, y, z])
    forward = scene_center - pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = pos
    return pose


def _create_pyrender_scene(
    scene_xyz: Optional[np.ndarray] = None,
    scene_rgb: Optional[np.ndarray] = None,
    obj_pcs: Optional[Dict[str, np.ndarray]] = None,
    obj_rgbs: Optional[Dict[str, np.ndarray]] = None,
    grasps: Optional[np.ndarray] = None,
    grasp_colors: Optional[np.ndarray] = None,
    grasp_mesh: Optional[trimesh.Trimesh] = None,
    best_grasp_idx: Optional[int] = None,
    obb_dict: Optional[Dict] = None,
    point_radius: float = 0.002,
) -> pyrender.Scene:
    """Build a pyrender scene with all visualization elements.
    
    Renders grasps as line-based gripper outlines (like viser) instead of
    full 3D meshes, for cleaner and more readable visualization.
    """
    scene = pyrender.Scene(
        ambient_light=np.array([0.4, 0.4, 0.4]),
        bg_color=np.array([0.12, 0.12, 0.18]),
    )

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))
    
    # Add fill light from opposite direction
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = np.array([-1, -1, 1])
    scene.add(fill_light, pose=fill_pose)

    # Render scene point cloud
    if scene_xyz is not None and len(scene_xyz) > 0:
        pc_mesh = _make_point_sphere_mesh(scene_xyz, scene_rgb, radius=point_radius)
        if len(pc_mesh.vertices) > 0:
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.8,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            scene.add(pyrender.Mesh.from_trimesh(pc_mesh, material=mat, smooth=False))

    # Render object point clouds with distinct colors
    if obj_pcs is not None:
        obj_colors_list = [
            [255, 180, 80],   # orange
            [80, 200, 255],   # cyan
            [255, 100, 150],  # pink
            [150, 255, 100],  # light green
            [255, 255, 100],  # yellow
            [200, 150, 255],  # purple
        ]
        for idx, (label, pc) in enumerate(obj_pcs.items()):
            if len(pc) == 0:
                continue
            cols = obj_rgbs.get(label) if obj_rgbs else None
            if cols is None:
                color_idx = idx % len(obj_colors_list)
                cols = np.tile(obj_colors_list[color_idx], (len(pc), 1)).astype(np.uint8)
            obj_mesh = _make_point_sphere_mesh(pc, cols, radius=point_radius * 1.5)
            if len(obj_mesh.vertices) > 0:
                mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, roughnessFactor=0.8,
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                )
                scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=mat, smooth=False))

    # Render grasps as line-based gripper outlines (similar to viser)
    if grasps is not None and len(grasps) > 0:
        # Use default gripper dimensions if no mesh available
        # These dimensions match a typical parallel jaw gripper
        gripper_width = 0.08  # w
        gripper_depth = 0.05  # d
        gripper_height = 0.04  # h
        z_offset = -0.01  # f (offset from grasp pose origin)
        
        # Try to get dimensions from mesh if available
        if grasp_mesh is not None:
            bounds = grasp_mesh.bounds
            if bounds is not None:
                extent = bounds[1] - bounds[0]
                gripper_width = max(float(extent[0]), 0.04)
                gripper_depth = max(float(extent[1]), 0.03)
                gripper_height = max(float(extent[2]), 0.02)

        for i, grasp in enumerate(grasps):
            color = grasp_colors[i] if grasp_colors is not None else [0, 100, 255]
            is_best = best_grasp_idx is not None and i == best_grasp_idx
            alpha = 1.0 if is_best else 0.7

            # Generate gripper outline lines in local frame
            local_points = np.array([
                [gripper_width / 2, 0, gripper_height / 2 + z_offset],
                [gripper_width / 2, 0, -gripper_height / 2 + z_offset],
                [0, 0, -gripper_height / 2 + z_offset],
                [0, 0, z_offset],
                [0, 0, -gripper_height / 2 + z_offset],
                [-gripper_width / 2, 0, -gripper_height / 2 + z_offset],
                [-gripper_width / 2, 0, gripper_height / 2 + z_offset],
            ])

            # Transform to world frame
            world_points = (grasp[:3, :3] @ local_points.T).T + grasp[:3, 3]

            # Create line segments (connecting the control points like viser)
            line_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
                (0, 2), (3, 6),
            ]

            for pi, pj in line_pairs:
                p1 = world_points[pi]
                p2 = world_points[pj]
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-5:
                    continue
                
                line_radius = 0.004 if is_best else 0.002
                cyl = trimesh.creation.cylinder(radius=line_radius, height=length)
                z_axis = np.array([0, 0, 1.0])
                direction_norm = direction / length
                axis = np.cross(z_axis, direction_norm)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis /= axis_norm
                    angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
                    rot_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                    cyl.apply_transform(rot_matrix)
                cyl.apply_translation((p1 + p2) / 2)

                mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    roughnessFactor=0.6,
                    baseColorFactor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha],
                )
                scene.add(pyrender.Mesh.from_trimesh(cyl, material=mat, smooth=False))

            # Add a small sphere at the grasp center for clarity
            if is_best:
                center_sphere = trimesh.creation.icosphere(radius=0.008)
                center_sphere.apply_translation(grasp[:3, 3])
                mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.5,
                    roughnessFactor=0.3,
                    baseColorFactor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0],
                )
                scene.add(pyrender.Mesh.from_trimesh(center_sphere, material=mat))

    return scene


def _score_to_colors(scores: np.ndarray) -> np.ndarray:
    """Convert grasp scores to RGB colors (red=low, green=high)."""
    colors = np.zeros((len(scores), 3), dtype=np.uint8)
    for i, s in enumerate(scores):
        r = int(255 * (1 - s))
        g = int(255 * s)
        colors[i] = [r, g, 0]
    return colors


def _downsample_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud to max_points."""
    if len(xyz) <= max_points:
        return xyz, rgb
    idx = np.random.choice(len(xyz), max_points, replace=False)
    return xyz[idx], rgb[idx]


def render_grasp_scene(
    scene_xyz: Optional[np.ndarray] = None,
    scene_rgb: Optional[np.ndarray] = None,
    obj_pcs: Optional[Dict[str, np.ndarray]] = None,
    obj_rgbs: Optional[Dict[str, np.ndarray]] = None,
    grasps: Optional[np.ndarray] = None,
    grasp_conf: Optional[np.ndarray] = None,
    grasp_mesh: Optional[trimesh.Trimesh] = None,
    obb_dict: Optional[Dict] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_pose: Optional[np.ndarray] = None,
    output_dir: str = ".",
    prefix: str = "grasp",
    resolution: Tuple[int, int] = (1600, 1200),
    max_scene_points: int = 2000,
    max_obj_points: int = 1000,
    num_views: int = 1,
    scene_bounds: Optional[np.ndarray] = None,
) -> List[str]:
    """Render grasp scene and save PNG images.

    Args:
        scene_xyz: (N, 3) full scene point cloud (world frame)
        scene_rgb: (N, 3) colors for scene points (0-255)
        obj_pcs: {label: (M, 3)} per-object point clouds
        obj_rgbs: {label: (M, 3)} per-object colors
        grasps: (K, 4, 4) grasp poses
        grasp_conf: (K,) grasp confidence scores
        grasp_mesh: gripper mesh to render at each grasp pose
        obb_dict: OBB info dict with 'center', 'R', 'half_extent'
        camera_intrinsics: 3x3 camera intrinsics
        camera_pose: 4x4 camera-to-world transform
        output_dir: directory to save images
        prefix: filename prefix
        resolution: (width, height) of output images
        max_scene_points: max scene points to render
        max_obj_points: max points per object
        num_views: number of camera views
        scene_bounds: optional [[xmin,ymin,zmin],[xmax,ymax,zmax]] to filter scene

    Returns:
        List of saved image file paths
    """
    _setup_headless()
    os.makedirs(output_dir, exist_ok=True)

    # Save original scene points count for debug
    orig_scene_pts = len(scene_xyz) if scene_xyz is not None else 0
    print(f"[Viz] Input: scene_xyz={orig_scene_pts}, objects={len(obj_pcs) if obj_pcs else 0}, grasps={len(grasps) if grasps is not None else 0}")

    # Filter scene points with bounds, but skip if it would leave nothing
    if scene_xyz is not None and len(scene_xyz) > 0 and scene_bounds is not None:
        filtered_xyz, filtered_rgb = _filter_scene_points(scene_xyz, scene_rgb, scene_bounds)
        if len(filtered_xyz) > 10:  # Only use filtered if we still have enough points
            scene_xyz, scene_rgb = filtered_xyz, filtered_rgb
        else:
            print(f"[Viz] Warning: scene_bounds filtered too many points ({len(filtered_xyz)} remaining), using original")

    if scene_xyz is not None:
        scene_xyz, scene_rgb = _downsample_points(scene_xyz, scene_rgb, max_scene_points)
        print(f"[Viz] After downsample: scene_xyz={len(scene_xyz)}")

    if obj_pcs is not None:
        for label in list(obj_pcs.keys()):
            pc = obj_pcs[label]
            if len(pc) > max_obj_points:
                idx = np.random.choice(len(pc), max_obj_points, replace=False)
                obj_pcs[label] = pc[idx]
                if obj_rgbs and label in obj_rgbs:
                    obj_rgbs[label] = obj_rgbs[label][idx]
            print(f"[Viz] Object {label}: {len(obj_pcs[label])} points")

    if grasp_conf is not None and len(grasp_conf) > 0:
        colors = _score_to_colors(grasp_conf)
        best_idx = int(np.argmax(grasp_conf))
    else:
        colors = None
        best_idx = None

    scene_center, char_size = _scene_to_origin(scene_xyz, obj_pcs, grasps)
    print(f"[Viz] Scene center: {scene_center}, char_size: {char_size:.3f}")

    views = []

    # Always add a clear view of the scene center
    # Use a reasonable default camera pose looking at scene center
    dist = max(char_size * 2.0, 0.5)
    default_pose = _compute_camera_pose(scene_center, dist, 0.8, 0.4)
    views.append(("view_center", default_pose))
    print(f"[Viz] Default camera pose: {default_pose[:3, 3]}")

    # Also add camera_view if camera_pose is provided
    if camera_pose is not None:
        views.append(("camera_view", camera_pose.copy()))
        print(f"[Viz] Camera pose: {camera_pose[:3, 3]}")

    # Add additional views if requested
    if num_views >= 3:
        elev = 0.35
        for i, az in enumerate(np.linspace(0.15, 2 * np.pi + 0.15, max(num_views - 2, 2))):
            dist = char_size * 2.5
            pose = _compute_camera_pose(scene_center, dist, az, elev)
            views.append((f"view_{i:02d}", pose))
            if len(views) >= num_views:
                break

    if num_views >= 4:
        for i, az in enumerate(np.linspace(0, 2 * np.pi, max(num_views - len(views), 1))):
            dist = char_size * 2.0
            pose = _compute_camera_pose(scene_center, dist, az, 0.15)
            views.append((f"top_{i:02d}", pose))
            if len(views) >= num_views:
                break

    print(f"[Viz] Total views to render: {len(views)}")

    yfov = np.pi / 4
    if camera_intrinsics is not None:
        fy = camera_intrinsics[1, 1]
        yfov_from_intr = 2 * np.arctan(resolution[1] / (2 * fy))
        yfov = min(yfov_from_intr, np.pi / 3)
        print(f"[Viz] Using yfov from intrinsics: {yfov:.3f}")

    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    camera = pyrender.PerspectiveCamera(yfov=yfov)

    saved_paths = []
    for name, cam_pose in views[:num_views]:
        print(f"[Viz] Rendering view: {name}")
        pyr_scene = _create_pyrender_scene(
            scene_xyz=scene_xyz,
            scene_rgb=scene_rgb,
            obj_pcs=obj_pcs,
            obj_rgbs=obj_rgbs,
            grasps=grasps,
            grasp_colors=colors,
            grasp_mesh=grasp_mesh,
            best_grasp_idx=best_idx,
            obb_dict=obb_dict,
            point_radius=max(char_size * 0.008, 0.002),
        )
        pyr_scene.add(camera, pose=cam_pose)
        color, _ = renderer.render(pyr_scene)
        print(f"[Viz] Render result: shape={color.shape}, mean={color.mean():.2f}")

        filepath = os.path.join(output_dir, f"{prefix}_{name}.png")
        Image.fromarray(color).save(filepath)
        saved_paths.append(filepath)
        print(f"[Viz] Saved: {filepath}")

    renderer.delete()
    return saved_paths


def save_grasp_visualization(
    scene: Dict,
    grasps: np.ndarray,
    grasp_conf: np.ndarray,
    grasp_mesh: trimesh.Trimesh,
    obb_dict: Optional[Dict] = None,
    output_dir: str = "grasp_visualizations",
    prefix: str = "scene",
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_pose: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = (1600, 1200),
    num_views: int = 1,
    scene_bounds: Optional[np.ndarray] = None,
) -> List[str]:
    """High-level function to save grasp visualization for a scene.

    Args:
        scene: dict with keys:
            - 'scene_xyz': (N, 3) full scene point cloud
            - 'scene_rgb': (N, 3) scene colors (0-255)
            - 'objects': {label: {'pc': (M,3), 'rgb': (M,3)}} per-object data
        grasps: (K, 4, 4) grasp poses
        grasp_conf: (K,) grasp confidence scores
        grasp_mesh: gripper collision/visual mesh
        obb_dict: OBB info
        output_dir: directory to save images
        prefix: filename prefix
        camera_intrinsics: 3x3 intrinsics (optional)
        camera_pose: 4x4 camera pose (optional)
        resolution: (width, height)
        num_views: number of camera views
        scene_bounds: optional point cloud bounds filter

    Returns:
        List of saved file paths
    """
    scene_xyz = scene.get("scene_xyz")
    scene_rgb = scene.get("scene_rgb")
    objects = scene.get("objects", {})

    obj_pcs = {}
    obj_rgbs = {}
    for label, obj_data in objects.items():
        obj_pcs[label] = obj_data["pc"]
        if "rgb" in obj_data:
            obj_rgbs[label] = obj_data["rgb"]

    return render_grasp_scene(
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        obj_pcs=obj_pcs,
        obj_rgbs=obj_rgbs,
        grasps=grasps,
        grasp_conf=grasp_conf,
        grasp_mesh=grasp_mesh,
        obb_dict=obb_dict,
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose,
        output_dir=output_dir,
        prefix=prefix,
        resolution=resolution,
        num_views=num_views,
        scene_bounds=scene_bounds,
    )


def visualize_saved_grasps(
    scene_dir: str,
    assets_dir: Optional[str] = None,
    num_views: int = 1,
    resolution: Tuple[int, int] = (1600, 1200),
    viz_data: Optional[Dict] = None,
):
    """Visualize saved grasp .npz files and save as PNG images.
    One image per gripper showing all objects together.

    Args:
        scene_dir: path to the scene directory (containing grasps/ subdir)
        assets_dir: path to assets directory (for gripper configs)
        num_views: number of camera views
        resolution: image resolution
        viz_data: pre-loaded visualization data from generate_and_save_grasps
    """
    import json
    from pathlib import Path

    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        print(f"[Viz] Scene directory not found: {scene_dir}")
        return

    meta_path = scene_dir / "meta_data.json"
    if not meta_path.exists():
        print(f"[Viz] meta_data.json not found in {scene_dir}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    camera_intrinsics = np.array(meta.get("intrinsics", []))
    camera_pose = np.array(meta.get("camera_pose", []))
    scene_bounds = np.array(meta.get("scene_bounds", []))

    if viz_data is None:
        print("[Viz] No pre-loaded viz_data, loading from disk...")
        return visualize_saved_grasps_from_disk(
            scene_dir, assets_dir, num_views, resolution,
            camera_intrinsics=camera_intrinsics if len(camera_intrinsics) > 0 else None,
            camera_pose=camera_pose if len(camera_pose) > 0 else None,
            scene_bounds=scene_bounds if len(scene_bounds) > 0 else None,
        )

    scene_data = viz_data.get("scene")
    grippers_data = viz_data.get("grippers", {})

    for gripper_name, gripper_info in grippers_data.items():
        print(f"[Viz] Processing gripper: {gripper_name}")
        collision_mesh = gripper_info.get("collision_mesh")
        grasps_data = gripper_info.get("grasps", {})

        if collision_mesh is None:
            print(f"[Viz] No collision mesh for {gripper_name}")
            continue

        # Collect all objects' data for this gripper
        all_grasps = []
        all_conf = []
        all_objects = {}
        
        for obj_label, data in grasps_data.items():
            grasps = data.get("grasps")
            conf = data.get("conf")
            if grasps is None or len(grasps) == 0:
                print(f"[Viz] No grasps for {obj_label}")
                continue

            if obj_label not in scene_data.get("objects", {}):
                print(f"[Viz] Object {obj_label} not found in scene data")
                continue

            all_grasps.append(grasps)
            all_conf.append(conf)
            all_objects[obj_label] = scene_data["objects"][obj_label]

        if not all_grasps:
            print(f"[Viz] No valid grasps for {gripper_name}, skipping")
            continue

        # Merge all grasps and confs
        merged_grasps = np.concatenate(all_grasps, axis=0)
        merged_conf = np.concatenate(all_conf, axis=0)
        print(f"[Viz] {gripper_name}: merged {len(merged_grasps)} grasps from {len(all_grasps)} objects")

        # Build scene dict with all objects
        scene_dict = {
            "scene_xyz": scene_data.get("scene_xyz"),
            "scene_rgb": scene_data.get("scene_rgb"),
            "objects": all_objects,
        }
        print(f"[Viz] scene_xyz: {len(scene_dict['scene_xyz']) if scene_dict['scene_xyz'] is not None else 0} points")
        print(f"[Viz] objects: {list(all_objects.keys())}")

        output_dir = scene_dir / "visualizations" / gripper_name
        print(f"[Viz] Saving to: {output_dir}")
        saved_paths = save_grasp_visualization(
            scene=scene_dict,
            grasps=merged_grasps,
            grasp_conf=merged_conf,
            grasp_mesh=collision_mesh,
            obb_dict=None,
            output_dir=str(output_dir),
            prefix="all_objects",
            camera_intrinsics=camera_intrinsics if len(camera_intrinsics) > 0 else None,
            camera_pose=camera_pose if len(camera_pose) > 0 else None,
            resolution=resolution,
            num_views=num_views,
            scene_bounds=scene_bounds if len(scene_bounds) > 0 else None,
        )
        print(f"[Viz] Saved {len(saved_paths)} images for {gripper_name}: {saved_paths}")


def visualize_saved_grasps_from_disk(
    scene_dir: str,
    assets_dir: Optional[str] = None,
    num_views: int = 1,
    resolution: Tuple[int, int] = (1600, 1200),
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_pose: Optional[np.ndarray] = None,
    scene_bounds: Optional[np.ndarray] = None,
):
    """Fallback: Load .npz files from disk and visualize.
    One image per gripper showing all objects together.

    Args:
        scene_dir: path to the scene directory
        assets_dir: path to assets directory
        num_views: number of camera views
        resolution: image resolution
        camera_intrinsics: 3x3 camera intrinsics
        camera_pose: 4x4 camera pose
        scene_bounds: scene bounds filter
    """
    from pathlib import Path
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx._setup_dependencies import get_checkpoints_version_dir

    scene_dir = Path(scene_dir)
    if assets_dir is None:
        repo_root = scene_dir.parent.parent.parent
        assets_dir = str(repo_root / "assets")

    grasps_dir = scene_dir / "grasps"
    if not grasps_dir.exists():
        print(f"[Viz] Grasps directory not found: {grasps_dir}")
        return

    checkpoint_root = str(get_checkpoints_version_dir())
    from graspgenx.utils.checkpoint_io import load_model_cfg
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
    )

    for gripper_dir in grasps_dir.iterdir():
        if not gripper_dir.is_dir():
            continue
        gripper_name = gripper_dir.name
        print(f"[Viz] Loading gripper: {gripper_name}")

        try:
            sampler = GraspGenXSampler(
                model_cfg, gripper_name, assets_dir=assets_dir
            )
            gripper = sampler.get_gripper_info()
        except Exception as e:
            print(f"[Viz] Failed to load gripper {gripper_name}: {e}")
            continue

        scene_data = load_scene_data(str(scene_dir))
        if scene_data is None:
            return

        # Collect all objects' data for this gripper
        all_grasps = []
        all_conf = []
        all_objects = {}

        for npz_file in gripper_dir.glob("*.npz"):
            obj_label = npz_file.stem
            print(f"[Viz] Loading {obj_label} for {gripper_name}")

            try:
                data = np.load(npz_file)
                grasps = data["grasps"]
                conf = data["conf"]
                if len(grasps) == 0:
                    continue

                if obj_label not in scene_data.get("objects", {}):
                    continue

                all_grasps.append(grasps)
                all_conf.append(conf)
                all_objects[obj_label] = scene_data["objects"][obj_label]
            except Exception as e:
                import traceback
                print(f"[Viz] Failed to load {npz_file.name}: {e}")
                traceback.print_exc()

        if not all_grasps:
            print(f"[Viz] No valid grasps for {gripper_name}, skipping")
            continue

        # Merge all grasps and confs
        merged_grasps = np.concatenate(all_grasps, axis=0)
        merged_conf = np.concatenate(all_conf, axis=0)

        # Build scene dict with all objects
        scene_dict = {
            "scene_xyz": scene_data.get("scene_xyz"),
            "scene_rgb": scene_data.get("scene_rgb"),
            "objects": all_objects,
        }

        output_dir = scene_dir / "visualizations" / gripper_name
        saved_paths = save_grasp_visualization(
            scene=scene_dict,
            grasps=merged_grasps,
            grasp_conf=merged_conf,
            grasp_mesh=gripper.collision_mesh,
            obb_dict=None,
            output_dir=str(output_dir),
            prefix="all_objects",
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            resolution=resolution,
            num_views=num_views,
            scene_bounds=scene_bounds,
        )
        print(f"[Viz] Saved {len(saved_paths)} images for {gripper_name}: {saved_paths}")


def load_scene_data(scene_dir: str) -> Optional[Dict]:
    """Load scene data from disk as fallback."""
    try:
        from graspgenx.utils.scene_loaders import load_realworld_scene
        scene_data = load_realworld_scene(scene_dir, min_obj_points=100)
        return scene_data
    except Exception as e:
        print(f"[Viz] Failed to load scene data: {e}")
        return None