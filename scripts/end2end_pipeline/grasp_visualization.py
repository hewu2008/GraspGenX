# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive grasp visualization using viser HTTP server.

Provides functions to visualize grasp scenes interactively in a web browser,
matching the visualization style of demo_scene_pc.py.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from graspgenx.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_x_grasp,
    visualize_mesh,
    visualize_pointcloud,
    is_rotation_matrix,
)


def visualize_saved_grasps(
    scene_dir: str,
    assets_dir: Optional[str] = None,
    num_views: int = 1,
    resolution: Tuple[int, int] = (1600, 1200),
    viz_data: Optional[Dict] = None,
    port: int = 8080,
):
    """Visualize saved grasp .npz files using viser interactive server.

    Starts a viser HTTP server and renders all grasps for each gripper
    in the same scene view. User can interact with the scene in browser.

    Args:
        scene_dir: path to the scene directory (containing grasps/ subdir)
        assets_dir: path to assets directory (for gripper configs)
        num_views: not used (kept for API compatibility)
        resolution: not used (kept for API compatibility)
        viz_data: pre-loaded visualization data from generate_and_save_grasps
        port: port for viser server
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

    scene_bounds = np.array(meta.get("scene_bounds", []))

    if viz_data is None:
        print("[Viz] No pre-loaded viz_data, cannot visualize via viser")
        print("[Viz] Please run generate_and_save_grasps first to get viz_data")
        return

    scene_data = viz_data.get("scene")
    grippers_data = viz_data.get("grippers", {})

    if not scene_data or not grippers_data:
        print("[Viz] No scene or gripper data to visualize")
        return

    # Create viser server
    vis = create_visualizer(port=port)

    # Render full scene point cloud
    scene_xyz = scene_data.get("scene_xyz")
    scene_rgb = scene_data.get("scene_rgb")
    print(f"[Viz] scene_xyz: shape={scene_xyz.shape if scene_xyz is not None else 'None'}")
    if scene_xyz is not None and len(scene_xyz) > 0:
        # Apply scene_bounds filter only if it would keep some points
        if len(scene_bounds) > 0:
            mask = np.all(
                (scene_xyz >= scene_bounds[0]) & (scene_xyz <= scene_bounds[1]),
                axis=1,
            )
            filtered_count = np.sum(mask)
            print(f"[Viz] scene_bounds filter: {filtered_count}/{len(scene_xyz)} points kept")
            if filtered_count > 0:
                scene_xyz = scene_xyz[mask]
                scene_rgb = scene_rgb[mask] if scene_rgb is not None else None

        if len(scene_xyz) > 0:
            visualize_pointcloud(
                vis, "pc_scene", scene_xyz, scene_rgb, size=0.0025
            )
            print(f"[Viz] Rendered scene point cloud: {len(scene_xyz)} points")
        else:
            print("[Viz] Warning: scene point cloud is empty after filtering, skipping")
    else:
        print("[Viz] Warning: scene_xyz is None or empty")

    # Render all object point clouds
    objects_data = scene_data.get("objects", {})
    print(f"[Viz] objects_data keys: {list(objects_data.keys())}")
    obj_colors_list = [
        [255, 180, 80],   # orange
        [80, 200, 255],   # cyan
        [255, 100, 150],  # pink
        [150, 255, 100],  # light green
        [255, 255, 100],  # yellow
        [200, 150, 255],  # purple
    ]

    for idx, (label, obj_data) in enumerate(objects_data.items()):
        obj_pc = obj_data.get("pc")
        obj_rgb = obj_data.get("rgb")
        print(f"[Viz] Object {label}: pc shape={obj_pc.shape if obj_pc is not None else 'None'}")
        if obj_pc is not None and len(obj_pc) > 0:
            # If no RGB, assign a distinctive color
            if obj_rgb is None or len(obj_rgb) == 0:
                color = obj_colors_list[idx % len(obj_colors_list)]
                obj_rgb = np.tile(color, (len(obj_pc), 1)).astype(np.uint8)

            visualize_pointcloud(
                vis, f"obj/{label}/pc", obj_pc, obj_rgb, size=0.004
            )
            print(f"[Viz] Rendered object {label}: {len(obj_pc)} points")

    # Render all grasps for each gripper
    for gripper_name, gripper_data in grippers_data.items():
        print(f"[Viz] Processing gripper: {gripper_name}")
        collision_mesh = gripper_data.get("collision_mesh")
        gripper_info_obj = gripper_data.get("gripper_info")
        grasps_data = gripper_data.get("grasps", {})

        if collision_mesh is None:
            print(f"[Viz] No collision mesh for {gripper_name}")
            continue

        # Collect all objects' data for this gripper
        all_grasps = []
        all_conf = []
        all_labels = []

        for obj_label, data in grasps_data.items():
            grasps = data.get("grasps")
            conf = data.get("conf")
            if grasps is None or len(grasps) == 0:
                continue

            all_grasps.append(grasps)
            all_conf.append(conf)
            all_labels.append(obj_label)

        if not all_grasps:
            print(f"[Viz] No valid grasps for {gripper_name}")
            continue

        # Merge all grasps and confs
        merged_grasps = np.concatenate(all_grasps, axis=0)
        merged_conf = np.concatenate(all_conf, axis=0)
        print(f"[Viz] {gripper_name}: {len(merged_grasps)} grasps from {len(all_labels)} objects")

        # Render grasps using visualize_x_grasp (same as demo)
        scores = get_color_from_score(merged_conf, use_255_scale=True)
        best_idx = int(merged_conf.argmax())

        for j, grasp in enumerate(merged_grasps):
            color = [0, 100, 255] if j == best_idx else scores[j]
            linewidth = 5.0 if j == best_idx else 1.5

            ns_prefix = f"grasps/{gripper_name}"
            ns = f"{ns_prefix}/grasp_{j:03d}"
            visualize_x_grasp(
                vis,
                ns,
                grasp,
                color=color,
                gripper_info=gripper_info_obj,
                linewidth=linewidth,
            )

        # Render top grasp mesh
        visualize_mesh(
            vis,
            f"grasps/{gripper_name}/top_mesh",
            collision_mesh,
            color=[0, 100, 255],
            transform=merged_grasps[best_idx],
        )

    print("\n" + "=" * 60)
    print("[Viz] Visualization is ready!")
    print(f"[Viz] Open browser and visit: http://localhost:{port}")
    print("[Viz] Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Keep server running and wait for user to press Enter
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Viz] Stopping visualization server...")
        print("[Viz] Done!")


def visualize_saved_grasps_from_disk(
    scene_dir: str,
    assets_dir: Optional[str] = None,
    num_views: int = 1,
    resolution: Tuple[int, int] = (1600, 1200),
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_pose: Optional[np.ndarray] = None,
    scene_bounds: Optional[np.ndarray] = None,
    port: int = 8080,
):
    """Fallback: Load .npz files from disk and visualize via viser.

    Args:
        scene_dir: path to the scene directory
        assets_dir: path to assets directory
        num_views: not used (kept for API compatibility)
        resolution: not used (kept for API compatibility)
        camera_intrinsics: not used (kept for API compatibility)
        camera_pose: not used (kept for API compatibility)
        scene_bounds: scene bounds filter
        port: port for viser server
    """
    from pathlib import Path
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx._setup_dependencies import get_checkpoints_version_dir
    from graspgenx.utils.scene_loaders import load_realworld_scene

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

    # Load scene data
    scene = load_realworld_scene(str(scene_dir), min_obj_points=100)
    if scene is None:
        print("[Viz] Failed to load scene data")
        return

    # Create viser server
    vis = create_visualizer(port=port)

    # Render scene point cloud
    scene_xyz = scene.get("scene_xyz")
    scene_rgb = scene.get("scene_rgb")
    if scene_xyz is not None and len(scene_xyz) > 0:
        visualize_pointcloud(
            vis, "pc_scene", scene_xyz, scene_rgb, size=0.0025
        )

    # Render all object point clouds
    objects = scene.get("objects", {})
    for label, obj_data in objects.items():
        obj_pc = obj_data.get("pc")
        obj_rgb = obj_data.get("rgb")
        if obj_pc is not None and len(obj_pc) > 0:
            visualize_pointcloud(
                vis, f"obj/{label}/pc", obj_pc, obj_rgb, size=0.004
            )

    # Process each gripper
    for gripper_dir in grasps_dir.iterdir():
        if not gripper_dir.is_dir():
            continue
        gripper_name = gripper_dir.name
        print(f"[Viz] Loading gripper: {gripper_name}")

        try:
            sampler = GraspGenXSampler(
                model_cfg, gripper_name, assets_dir=assets_dir
            )
            gripper_info = sampler.get_gripper_info()
        except Exception as e:
            print(f"[Viz] Failed to load gripper {gripper_name}: {e}")
            continue

        collision_mesh = gripper_info.collision_mesh

        # Collect all objects' grasps
        all_grasps = []
        all_conf = []

        for npz_file in gripper_dir.glob("*.npz"):
            obj_label = npz_file.stem
            try:
                data = np.load(npz_file)
                grasps = data["grasps"]
                conf = data["conf"]
                if len(grasps) == 0:
                    continue
                all_grasps.append(grasps)
                all_conf.append(conf)
            except Exception as e:
                print(f"[Viz] Failed to load {npz_file.name}: {e}")
                continue

        if not all_grasps:
            print(f"[Viz] No valid grasps for {gripper_name}")
            continue

        # Merge all grasps and confs
        merged_grasps = np.concatenate(all_grasps, axis=0)
        merged_conf = np.concatenate(all_conf, axis=0)
        print(f"[Viz] {gripper_name}: {len(merged_grasps)} total grasps")

        # Render grasps
        scores = get_color_from_score(merged_conf, use_255_scale=True)
        best_idx = int(merged_conf.argmax())

        for j, grasp in enumerate(merged_grasps):
            color = [0, 100, 255] if j == best_idx else scores[j]
            linewidth = 5.0 if j == best_idx else 1.5

            ns = f"grasps/{gripper_name}/grasp_{j:03d}"
            visualize_x_grasp(
                vis,
                ns,
                grasp,
                color=color,
                gripper_info=gripper_info,
                linewidth=linewidth,
            )

        # Render top grasp mesh
        visualize_mesh(
            vis,
            f"grasps/{gripper_name}/top_mesh",
            collision_mesh,
            color=[0, 100, 255],
            transform=merged_grasps[best_idx],
        )

    print("\n" + "=" * 60)
    print("[Viz] Visualization is ready!")
    print(f"[Viz] Open browser and visit: http://localhost:{port}")
    print("[Viz] Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Viz] Stopping visualization server...")
        print("[Viz] Done!")