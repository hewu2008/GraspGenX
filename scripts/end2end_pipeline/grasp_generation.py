"""Grasp pose generation (planner + collision + orientation filters).

Mirrors scripts/demo_scene_pc.py with the zerith tuning from
tools/run_demo_scene_pc_zerith.sh.  Requires the graspgenx stack: run in the
zerith_graspgen env with $GRASPGENX_CHECKPOINT_DIR and
$GRASPGENX_GRIPPER_CFG_DIR set.
"""

import json
import os

import numpy as np

from .camera_pose import compute_camera_pose
from .config import K_COLOR, K_HAND_COLOR, SCENE_BOUNDS
from .logging_utils import get_logger

logger = get_logger(__name__)

META_FILENAME = "meta_data.json"

# ---- Grasp pose generation (mirrors scripts/demo_scene_pc.py, zerith tuning) ----
GRASP_GRIPPERS = ("zerith_left_gripper", "zerith_right_gripper")

# Planner + filter parameters matching tools/run_demo_scene_pc_zerith.sh.
GRASP_PLANNER = "graspmoe"
GRASP_THRESHOLD = 0.7
GRASP_NUM_GRASPS = 200
GRASP_MOE_NUM_YAWS = 36
GRASP_MOE_Z_OFFSETS_CM = (0.0, 0.8, 1.6, 2.4)
GRASP_MOE_OUTLIER_THRESHOLD = 0.014
GRASP_MOE_OUTLIER_K = 20
GRASP_MOE_OBB_MODE = "advanced"
GRASP_MOE_SKIP_OBB_RULE = "auto"
GRASP_MOE_OBB_DENSITY = "dense"
GRASP_MOE_OBB_POSITION_SPACING_CM = 1.0
GRASP_MIN_OBJ_POINTS = 100
GRASP_COLLISION_THRESHOLD = 0.001
# Camera-frame orientation filter: keep only grasps whose roll stays within
# +/-GRASP_MAX_ROLL_DEG, whose pitch stays within
# [GRASP_PITCH_MIN_DEG, GRASP_PITCH_MAX_DEG] and whose yaw stays within
# +/-GRASP_MAX_YAW_DEG. Disable with False to keep all poses.
#
# The pitch range is intentionally ASYMMETRIC: real-scene data shows positive
# camera-frame pitch correlates with IK solutions sitting at/against the wrist
# joint limits (and with bad execution), while negative/small pitch grasps have
# comfortable margins. Positive-pitch grasps are therefore capped at +5 deg.
GRASP_FILTER_ORIENTATION = True
GRASP_MAX_ROLL_DEG = 20.0
GRASP_PITCH_MIN_DEG = -20.0
GRASP_PITCH_MAX_DEG = 5.0
GRASP_MAX_YAW_DEG = 135.0
GRASP_MAX_SCENE_POINTS = 8192
GRASP_NUM_COLLISION_SAMPLES = 4000
GRASPS_SUBDIR = "grasps"


def write_meta_data(scene_dir, robot, num_objects, camera_pose=None, intrinsics=None):
    """Write meta_data.json for the just-captured scene.

    Fields match what graspgenx.utils.scene_loaders expects:
      intrinsics    : 3x3 K (from config.K_COLOR, or ``intrinsics`` if given)
      camera_pose   : 4x4 camera-to-world (from IMU + motors via compute_camera_pose,
                      or ``camera_pose`` if given)
      label_map     : {"ground": 0, "obj_i": 100 + i}  (matches seg.png convention)
      scene_bounds  : workspace bbox (from config.SCENE_BOUNDS)

    Passing ``camera_pose=np.eye(4)`` makes the camera frame the world frame; this
    is used for the hand camera whose wrist-mounted (fixed URDF offset) hand-eye
    transform operates directly in the camera frame and needs no world FK.

    Returns the written path, or None if the camera pose could not be computed.
    """
    if camera_pose is None:
        T = compute_camera_pose(robot)
        if T is None:
            logger.error("[Perc] Failed to compute camera pose; skipping meta_data.json.")
            return None
    else:
        T = np.asarray(camera_pose, dtype=np.float64)

    K = K_HAND_COLOR if intrinsics is None and camera_pose is not None else K_COLOR
    if intrinsics is not None:
        K = np.asarray(intrinsics, dtype=np.float64)

    label_map = {"ground": 0}
    for i in range(1, num_objects + 1):
        label_map[f"obj_{i}"] = 100 + i

    meta = {
        "intrinsics": K.tolist(),
        "camera_pose": T.tolist(),
        "label_map": label_map,
        "scene_bounds": list(SCENE_BOUNDS),
    }
    os.makedirs(scene_dir, exist_ok=True)
    path = os.path.join(scene_dir, META_FILENAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[Perc] meta_data.json written to {path}")
    return path

def generate_and_save_grasps(scene_dir, gripper_names=GRASP_GRIPPERS, assets_dir=None):
    """Generate top-down grasp poses per object for each gripper and save them
    under ``<scene_dir>/grasps/<gripper>/<label>.npz``.

    Mirrors scripts/demo_scene_pc.py (zerith tuning from
    tools/run_demo_scene_pc_zerith.sh): loads the realworld scene, runs the
    GraspMoE planner per gripper (the diffusion model is gripper-independent
    and shared across grippers), applies collision + top-down filtering, and
    writes one .npz per (gripper, object) with keys ``grasps`` (K,4,4),
    ``conf`` (K,), ``tags`` (K, str).

    Requires the graspgenx stack: run in the zerith_graspgen env with
    $GRASPGENX_CHECKPOINT_DIR and $GRASPGENX_GRIPPER_CFG_DIR set.

    Returns a tuple: (summary, viz_data)
        - summary: {gripper_name: {label: num_grasps_saved}}
        - viz_data: dict containing scene data and gripper meshes for visualization
    """
    import trimesh
    from scipy.spatial.transform import Rotation as R
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.samplers.planner import run_planner_on_batch
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.utils.collision_filter import filter_colliding_grasps
    from graspgenx.utils.scene_loaders import (
        build_scene_pc_excluding_object,
        load_realworld_scene,
    )
    from graspgenx._setup_dependencies import get_checkpoints_version_dir

    scene = load_realworld_scene(scene_dir, min_obj_points=GRASP_MIN_OBJ_POINTS)
    labels = list(scene["objects"].keys())
    if not labels:
        logger.info("[Perc] No segmented objects; skipping grasp generation.")
        return {}, None
    obj_pcs = [scene["objects"][lab]["pc"] for lab in labels]
    logger.info(f"[Perc] Generating grasps for {len(labels)} object(s): {labels}")

    if assets_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        assets_dir = os.path.join(repo_root, "assets")

    checkpoint_root = str(get_checkpoints_version_dir())
    logger.info(f"[Perc] Checkpoints: {checkpoint_root}")
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
    )

    z_offsets = tuple(float(x) for x in GRASP_MOE_Z_OFFSETS_CM)

    summary = {}
    viz_data = {"scene": scene, "grippers": {}}
    model = None  # shared across grippers (model is gripper-independent)
    for gi, gripper_name in enumerate(gripper_names):
        logger.info(f"[Perc] Gripper {gi + 1}/{len(gripper_names)}: {gripper_name}")
        sampler = GraspGenXSampler(
            model_cfg, gripper_name, assets_dir=assets_dir, model=model
        )
        if model is None:
            model = sampler.model
        gripper = sampler.get_gripper_info()
        viz_data["grippers"][gripper_name] = {
            "gripper_info": gripper,
            "collision_mesh": gripper.collision_mesh,
            "sweep_volume": gripper.sweep_volume if hasattr(gripper, "sweep_volume") else None,
        }

        sampled_pts, _ = trimesh.sample.sample_surface(
            gripper.collision_mesh, GRASP_NUM_COLLISION_SAMPLES
        )
        gripper_surface_points = np.asarray(sampled_pts, dtype=np.float32)

        batch_results = run_planner_on_batch(
            obj_pcs,
            sampler,
            planner=GRASP_PLANNER,
            grasp_threshold=GRASP_THRESHOLD,
            num_grasps=GRASP_NUM_GRASPS,
            moe_num_yaws=GRASP_MOE_NUM_YAWS,
            moe_z_offsets_cm=z_offsets,
            moe_outlier_threshold=GRASP_MOE_OUTLIER_THRESHOLD,
            moe_outlier_k=GRASP_MOE_OUTLIER_K,
            moe_obb_mode=GRASP_MOE_OBB_MODE,
            moe_skip_obb_rule=GRASP_MOE_SKIP_OBB_RULE,
            moe_obb_density=GRASP_MOE_OBB_DENSITY,
            moe_obb_position_spacing_cm=GRASP_MOE_OBB_POSITION_SPACING_CM,
        )

        out_dir = os.path.join(scene_dir, GRASPS_SUBDIR, gripper_name)
        os.makedirs(out_dir, exist_ok=True)
        per_gripper = {}
        grasps_for_viz = {}
        for label, (grasps, conf, tags, _obb) in zip(labels, batch_results):
            if len(grasps) == 0:
                logger.info(f"[Perc] [{gripper_name}/{label}] no grasps")
                continue

            # Collision filter (target object's own pixels excluded).
            scene_pc_raw = build_scene_pc_excluding_object(scene, label)
            if len(scene_pc_raw) > GRASP_MAX_SCENE_POINTS:
                idx = np.random.choice(
                    len(scene_pc_raw), GRASP_MAX_SCENE_POINTS, replace=False
                )
                scene_pc = scene_pc_raw[idx]
            else:
                scene_pc = scene_pc_raw
            logger.info(
                f"[Perc] [{gripper_name}/{label}] scene_pc excluding target: "
                f"{len(scene_pc_raw)} pts -> using {len(scene_pc)} for collision check"
            )
            cf_mask = filter_colliding_grasps(
                scene_pc=scene_pc,
                grasp_poses=grasps,
                collision_threshold=GRASP_COLLISION_THRESHOLD,
                gripper_surface_points=gripper_surface_points,
            )
            dropped_grasps = grasps[~cf_mask]
            _log_collision_source_distribution(
                labels=labels,
                obj_pcs=obj_pcs,
                collided_grasps=dropped_grasps,
                scene_pc=scene_pc,
                gripper_surface_points=gripper_surface_points,
                threshold=GRASP_COLLISION_THRESHOLD,
                gripper_name=gripper_name,
                label=label,
            )
            grasps = grasps[cf_mask]
            conf = conf[cf_mask]
            tags = [t for t, keep in zip(tags, cf_mask) if keep]

            # Pitch/roll/yaw filter in the CAMERA frame. A top-down grasp is
            # ~vertical, so its euler_xyz first angle (roll) is ~180 deg (or 0
            # deg) about the approach axis -- an equivalent pose. We keep only
            # grasps whose roll stays within +/-20 deg, whose pitch stays within
            # [-20, +5] deg and whose yaw stays within +/-90 deg. The pitch
            # range is asymmetric: positive-pitch (camera-frame) grasps tend to
            # sit against the wrist joint limits and execute poorly, so they are
            # capped at +5 deg.
            if GRASP_FILTER_ORIENTATION and len(grasps) > 0:
                T_world_cam = np.linalg.inv(scene["camera_pose"])
                grasps_cam = T_world_cam @ grasps  # (K,4,4) world -> camera
                eul = R.from_matrix(grasps_cam[:, :3, :3]).as_euler(
                    "xyz", degrees=True
                )
                roll, pitch, yaw = eul[:, 0], eul[:, 1], eul[:, 2]
                mask = (
                    (np.abs(roll) <= GRASP_MAX_ROLL_DEG)
                    & (pitch >= GRASP_PITCH_MIN_DEG)
                    & (pitch <= GRASP_PITCH_MAX_DEG)
                    & (np.abs(yaw) <= GRASP_MAX_YAW_DEG)
                )
                grasps = grasps[mask]
                conf = conf[mask]
                tags = [t for t, keep in zip(tags, mask) if keep]

            if len(grasps) == 0:
                logger.info(
                    f"[Perc] [{gripper_name}/{label}] no grasps after filtering"
                )
                continue

            out_path = os.path.join(out_dir, f"{label}.npz")
            np.savez(
                out_path,
                grasps=grasps.astype(np.float32),
                conf=conf.astype(np.float32),
                tags=np.array(tags, dtype="<U8"),
            )
            per_gripper[label] = len(grasps)
            grasps_for_viz[label] = {
                "grasps": grasps.astype(np.float32),
                "conf": conf.astype(np.float32),
                "tags": tags,
            }
            logger.info(
                f"[Perc] [{gripper_name}/{label}] saved {len(grasps)} grasps -> {out_path}"
            )
        summary[gripper_name] = per_gripper
        viz_data["grippers"][gripper_name]["grasps"] = grasps_for_viz
    return summary, viz_data


def _log_collision_source_distribution(
    labels,
    obj_pcs,
    collided_grasps,
    scene_pc,
    gripper_surface_points,
    threshold,
    gripper_name,
    label,
    num_samples=800,
    contact_tol_m=0.010,
    bg_tol_m=0.050,
):
    """Diagnose where the collision-filter rejections come from.

    For each collision-rejected grasp, sample ``num_samples`` points on the
    gripper collision mesh, transform them to world, and find which ones land
    within ``threshold`` of the (target-excluded) scene point cloud. The
    nearest-target-object distance of each such collision point classifies the
    rejection source:

      * target-surface / through-body  : distance < contact_tol_m  (build_scene_pc_excluding_object
                                          left target pixels behind, or grasp pierces the object)
      * target-nearby edge              : contact_tol_m .. bg_tol_m (gripper skirts the object with no margin)
      * background / tray / other       : distance >= bg_tol_m

    Also logs how many scene points are actually used so the 8192 cap question
    can be answered from data.
    """
    from scipy.spatial import cKDTree

    if len(collided_grasps) == 0:
        return

    target_pc = np.asarray(obj_pcs[labels.index(label)], dtype=np.float64)
    target_tree = cKDTree(target_pc)
    scene_tree = cKDTree(np.asarray(scene_pc, dtype=np.float64))

    rng = np.random.default_rng(0)
    n_samp = min(num_samples, len(gripper_surface_points))
    gidx = rng.choice(len(gripper_surface_points), size=n_samp, replace=False)
    gpts = np.asarray(gripper_surface_points, dtype=np.float64)[gidx]  # (S, 3)

    coll_tgt_dist = []
    for p in np.asarray(collided_grasps, dtype=np.float64):
        world = (p[:3, :3] @ gpts.T + p[:3, 3:4]).T  # (S, 3)
        d_scene = scene_tree.query(world)[0]
        hit = world[d_scene < threshold]
        if len(hit) > 0:
            coll_tgt_dist.extend(target_tree.query(hit)[0].tolist())

    arr = np.asarray(coll_tgt_dist, dtype=np.float64)
    if len(arr) == 0:
        logger.warning(
            f"[Perc] [{gripper_name}/{label}] {len(collided_grasps)} grasps collided "
            "but no collision points found (scene_pc check mismatch)."
        )
        return

    n_contact = int((arr < contact_tol_m).sum())
    n_edge = int(((arr >= contact_tol_m) & (arr < bg_tol_m)).sum())
    n_bg = int((arr >= bg_tol_m).sum())
    tot = len(arr)
    logger.info(
        f"[Perc] [{gripper_name}/{label}] collision-source: {len(collided_grasps)} "
        f"rejected grasps, {tot} collision pts vs scene_pc({len(scene_pc)} pts); "
        f"target-dist pts: "
        f"contact-obj(<{int(contact_tol_m*1000)}mm)={n_contact}({100*n_contact/tot:.0f}%), "
        f"edge({int(contact_tol_m*1000)}-{int(bg_tol_m*1000)}mm)={n_edge}({100*n_edge/tot:.0f}%), "
        f"background/tray(>={int(bg_tol_m*1000)}mm)={n_bg}({100*n_bg/tot:.0f}%); "
        f"target-dist med={1000*np.median(arr):.0f}mm, max={1000*arr.max():.0f}mm"
    )
