"""Closed-loop arm-target visualization (offline, reads run.log, no robot).

Reconstructs the SAME camera->arm chain the pipeline uses (T1..T4) so the
perceived object point cloud and the commanded SDK end-effector target live in
ONE coordinate frame (the current-arm / SDK relative frame). You can then see
whether the gripper actually reaches the object.

Dependencies: run.log must contain the new runtime-input line added to
grasp_executor.calculate_target_relative_pose:
    [Arm] runtime inputs: cam_pos_rel=[...], cam_quat_rel=[...], arm_pos_rel=[...], arm_quat_rel=[...]
so run the pipeline once after that change to generate the needed data.

Usage:
    python scripts/visualize_grasp_arm_closed_loop.py \
        --scene_dir assets/zerith/real_scene/02 --log assets/zerith/real_scene/02/run.log \
        --out_port 9090
"""

import argparse
import json
import os
import re

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from graspgenx.utils.viser_utils import (
    create_visualizer,
    visualize_pointcloud,
    visualize_grasp,
    make_frame,
)

FINGERTIP_DEPTH_M = 0.16  # zerith_left_gripper "fingertip"[2]

# Same constants as grasp_executor.calculate_target_relative_pose
_T2_rot = R.from_euler("xyz", [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
_T2_t = np.array([0.2194, 0.0325, 0.6075])
_T3_t = np.array([-0.5743, -0.1800, -0.1208])


def _T(mat, t):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = t
    return T


def apply(pc, T):
    """Apply 4x4 T to (N,3) cloud; returns (N,3)."""
    return pc @ T[:3, :3].T + T[:3, 3]


def parse_log(log_path):
    txt = open(log_path).read()
    m = re.search(
        r"runtime inputs: cam_pos_rel=\[([^\]]*)\], cam_quat_rel=\[([^\]]*)\], "
        r"arm_pos_rel=\[([^\]]*)\], arm_quat_rel=\[([^\]]*)\]",
        txt,
    )
    if not m:
        raise SystemExit(
            "run.log has no '[Arm] runtime inputs:' line. Re-run the pipeline once "
            "after the grasp_executor.py change to log them."
        )
    cam_pos = np.array([float(v) for v in m.group(1).split(",")])
    cam_quat = np.array([float(v) for v in m.group(2).split(",")])
    arm_pos = np.array([float(v) for v in m.group(3).split(",")])
    arm_quat = np.array([float(v) for v in m.group(4).split(",")])

    def arm_frame_label(name):
        pat = re.compile(
            r"\[Transform\] " + re.escape(name) + r":\s*\n"
            r"\[Transform\]\s+translation=\[([^\]]*)\]\s*\n"
            r"\[Transform\]\s+euler_xyz\(deg\)=\[([^\]]*)\]",
            re.M,
        )
        mm = pat.search(txt)
        if not mm:
            raise SystemExit(f"cannot find [Transform] {name} in log")
        t = np.array([float(v) for v in mm.group(1).split(",")])
        e = np.array([float(v) for v in mm.group(2).split(",")])
        return _T(R.from_euler("xyz", e, degrees=True).as_matrix(), t)

    grasp_in_arm = arm_frame_label("before T_grasp_local (T_obj_in_arm)")
    sdk_target = arm_frame_label("after T4 / SDK EEF target (arm-relative)")
    return cam_pos, cam_quat, arm_pos, arm_quat, grasp_in_arm, sdk_target


def build_object_cam(scene_dir):
    seg = np.asarray(Image.open(os.path.join(scene_dir, "seg.png"))).astype(np.int32)
    depth = np.load(os.path.join(scene_dir, "depth.npy")).astype(np.float32)
    with open(os.path.join(scene_dir, "meta_data.json")) as f:
        meta = json.load(f)
    K = np.asarray(meta["intrinsics"]).reshape(3, 3)
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    z = depth
    xyz = np.stack([x, y, z], -1)
    obj_ids = [val for k, val in meta["label_map"].items() if k != "ground"]
    mask = np.zeros_like(seg, bool)
    for oid in obj_ids:
        mask |= seg == oid
    mask &= (depth > 0) & np.isfinite(depth)
    pc = xyz[mask]
    rgb = np.asarray(Image.open(os.path.join(scene_dir, "rgb.png")).convert("RGB"))[mask] \
        if os.path.exists(os.path.join(scene_dir, "rgb.png")) else None
    return pc, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_port", type=int, default=8091)
    args = ap.parse_args()

    cam_pos, cam_quat, arm_pos, arm_quat, grasp_in_arm, sdk_target = parse_log(args.log)
    print(f"cam_pos_rel={np.round(cam_pos,4)}  cam_quat_rel={np.round(cam_quat,4)}")
    print(f"arm_pos_rel={np.round(arm_pos,4)}  arm_quat_rel={np.round(arm_quat,4)}")

    # Rebuild the pipeline chain with the logged runtime inputs.
    T1 = _T(R.from_quat(cam_quat).as_matrix(), cam_pos)
    T2 = _T(_T2_rot, _T2_t)
    T3 = _T(np.eye(3), _T3_t)
    T4 = np.linalg.inv(_T(R.from_quat(arm_quat).as_matrix(), arm_pos))
    T_chain = T4 @ T3 @ T2 @ T1  # camera -> arm-relative
    print(f"recomputed T_obj_in_arm = T_chain @ T_cam => {np.round(grasp_in_arm[:3,3],4)}")

    # Perceived object cloud in the SAME arm-relative frame.
    pc_cam, rgb = build_object_cam(args.scene_dir)
    pc_arm = apply(pc_cam, T_chain)
    print(f"object cloud (arm frame): {len(pc_arm)} pts")

    # grasp tip/contact in arm frame (GraspGenX +Z = approach)
    GZ = grasp_in_arm[:3, :3][:, 2]
    grasp_tip_arm = grasp_in_arm[:3, 3] + FINGERTIP_DEPTH_M * GZ

    # closed-loop metric: distance from commanded target & contact to object cloud
    def nn(pt):
        d = np.sqrt(np.sum((pc_arm - pt) ** 2, axis=1))
        return float(np.min(d)) * 1000, pc_arm[int(np.argmin(d))]

    d_contact, _ = nn(grasp_tip_arm)
    d_target, _ = nn(sdk_target[:3, 3])
    print(f"NN contact->object  : {d_contact:.0f} mm   (grasp tip {np.round(grasp_tip_arm,3)})")
    print(f"NN EEF-target->object: {d_target:.0f} mm   (SDK EEF  {np.round(sdk_target[:3,3],3)})")

    if d_contact < 30:
        print("=> closed-loop OK: contact point is on the perceived object.")
    else:
        print("=> closed-loop MISMATCH: gripper does not reach the perceived object.")

    # ---------------- viser render (single unified arm-relative frame) --------
    vis = create_visualizer(clear=True, port=args.out_port)
    make_frame(vis, "arm/origin", h=0.2, T=np.eye(4))
    visualize_pointcloud(vis, "arm/object_cloud", pc_arm, rgb, size=0.004)

    vis.scene.add_icosphere("arm/grasp_base", radius=0.012,
                            position=grasp_in_arm[:3, 3], color=(255, 0, 255))
    vis.scene.add_icosphere("arm/grasp_tip", radius=0.012,
                            position=grasp_tip_arm, color=(0, 255, 0))
    approach_ep = grasp_in_arm[:3, 3] + 0.08 * GZ
    vis.scene.add_line_segments(
        "arm/approach",
        points=np.array([[grasp_in_arm[:3, 3], approach_ep]]),
        colors=(0, 255, 255), line_width=3.0,
    )
    make_frame(vis, "arm/grasp_base_frame", h=0.12, T=grasp_in_arm)
    # commanded SDK EEF pose + gripper
    T_target = sdk_target.copy()
    visualize_grasp(vis, "arm/eef_target_gripper", T_target, color=[0, 200, 255])
    make_frame(vis, "arm/sdk_target_frame", h=0.14, T=T_target)

    print("\n" + "=" * 60)
    print(f"Open http://localhost:{args.out_port} — single arm-relative scene:")
    print("  orange  = perceived object cloud (from depth)")
    print("  magenta = GraspGenX base, green = contact/tip, cyan = approach")
    print("  blue    = commanded SDK end-effector target + gripper")
    print("=" * 60 + "\n")

    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()