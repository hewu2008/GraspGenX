"""Visualize a GraspGenX grasp pose (camera frame) projected onto the RGB image.

GraspGenX pose convention: the 4x4 grasp pose is anchored at the GRIPPER BASE,
offset by ``fingertip_depth`` (= config.json "fingertip"[2], 0.16 m for the
Zerith gripper) behind the contact/tip point along the approach (+Z) axis:
    tip = base + fingertip_depth * R[:, 2]

We project BOTH the base (pose origin) and the tip/contact point onto the RGB
image via the camera intrinsics, then check the depth / segmentation at each.

Usage:
    python scripts/visualize_grasp_cam.py \
        --scene_dir assets/zerith/real_scene/02 \
        --pos 0.08100385011015943 0.03297408496282614 0.46304010716121163 \
        --euler_deg 3.673195729920793 -8.50719798997383 -25.67341101266362 \
        --out /tmp/grasp_vis.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R

K_COLOR = np.array(
    [[607.62, 0.00, 329.68],
     [0.00, 608.40, 243.36],
     [0.00, 0.00, 1.00]], dtype=np.float64)

FINGERTIP_DEPTH_M = 0.16  # zerith_left_gripper config.json "fingertip"[2]


def cam_to_pixel(pt_cam, K):
    """(3,) point in camera frame -> (u, v) pixel."""
    x, y, z = pt_cam
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.array([u, v])


def draw_arrow(ax, u0, v0, du, dv, length_px, color, lw):
    """Draw an arrow of fixed pixel length along (du, dv) from (u0, v0)."""
    norm = np.hypot(du, dv)
    if norm < 1e-9:
        return
    du, dv = du / norm * length_px, dv / norm * length_px
    ax.annotate(
        "", xy=(u0 + du, v0 + dv), xytext=(u0, v0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=18),
    )


def pixel_checks(name, pt_cam, depth, seg, K):
    """Project a camera point to a pixel and check depth/seg there."""
    u, v = cam_to_pixel(pt_cam, K)
    ui, vi = int(round(u)), int(round(v))
    H, W = depth.shape
    in_b = 0 <= ui < W and 0 <= vi < H
    d = depth[vi, ui] if in_b else np.nan
    s = seg[vi, ui] if in_b else None
    print(f"  [{name}] cam={np.round(pt_cam, 4)} -> pixel=({u:.1f},{v:.1f})  "
          f"depth={d:.3f} m (z={pt_cam[2]:.3f})  seg={s}")
    if np.isfinite(d) and d > 0:
        print(f"          depth vs z error = {abs(d - pt_cam[2]) * 1000:.1f} mm")
    return np.array([u, v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--pos", type=float, nargs=3, required=True)
    ap.add_argument("--euler_deg", type=float, nargs=3, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rgb_path = os.path.join(args.scene_dir, "rgb.png")
    depth_path = os.path.join(args.scene_dir, "depth.npy")
    seg_path = os.path.join(args.scene_dir, "seg.png")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    seg = np.asarray(Image.open(seg_path)).astype(np.int32)
    H, W = depth.shape

    print(f"rgb: shape={rgb.shape}")
    print(f"depth: shape={depth.shape} range={depth[depth>0].min():.3f}~{depth.max():.3f}")
    print(f"seg: labels={np.unique(seg)}")

    base = np.array(args.pos)
    mat = R.from_euler("xyz", args.euler_deg, degrees=True).as_matrix()
    approach = mat[:, 2]  # gripper +Z (approach direction)
    tip = base + FINGERTIP_DEPTH_M * approach  # contact point

    print(f"grasp base (pose origin, cam) = {base}")
    print(f"grasp approach(+Z, cam)        = {approach}")
    print(f"grasp tip/contact (cam)        = {tip}  (base + {FINGERTIP_DEPTH_M}*Z)")

    # --- Project base and tip ----------------------------------------------
    print("projections:")
    p_base = pixel_checks("base", base, depth, seg, K_COLOR)
    p_tip = pixel_checks("tip/contact", tip, depth, seg, K_COLOR)

    # axis endpoints for arrows (short)
    L = 0.04
    p_x = cam_to_pixel(base + L * mat[:, 0], K_COLOR)
    p_y = cam_to_pixel(base + L * mat[:, 1], K_COLOR)
    p_z = cam_to_pixel(base + L * mat[:, 2], K_COLOR)

    # --- Object localization ------------------------------------------------
    for lb in [l for l in np.unique(seg) if l > 0]:
        valid = (seg == lb) & (depth > 0) & np.isfinite(depth)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        cu, cv = xs.mean(), ys.mean()
        zs = depth[ys, xs]
        x_w = np.mean((xs - K_COLOR[0, 2]) * zs / K_COLOR[0, 0])
        y_w = np.mean((ys - K_COLOR[1, 2]) * zs / K_COLOR[1, 1])
        print(f"obj label {lb}: img centroid=({cu:.0f},{cv:.0f})  "
              f"cam pos=[{x_w:.4f},{y_w:.4f},{zs.mean():.4f}]  "
              f"depth range={zs.min():.3f}~{zs.max():.3f}")

    # --- Overlay plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)

    # object contour (all non-ground labels)
    for lb in [l for l in np.unique(seg) if l > 0]:
        ax.contour(seg == lb, levels=[0.5], colors="orange",
                   linewidths=1.2, alpha=0.9)

    # grasp base marker (pose origin)
    ax.plot(*p_base, marker="+", ms=16, mew=2.5, color="magenta")
    ax.add_patch(plt.Circle(p_base, 12, fill=False, ec="magenta", lw=1.5))

    # grasp tip / contact marker
    ax.plot(*p_tip, marker="o", ms=9, color="lime")
    ax.add_patch(plt.Circle(p_tip, 12, fill=False, ec="lime", lw=1.8))

    # line from base to tip (approach direction)
    ax.plot([p_base[0], p_tip[0]], [p_base[1], p_tip[1]],
            color="cyan", lw=1.5, ls="--", alpha=0.9)
    # short axis arrows
    draw_arrow(ax, *p_base, *(p_x - p_base), 28, "red", 1.6)
    draw_arrow(ax, *p_base, *(p_y - p_base), 28, "green", 1.6)
    draw_arrow(ax, *p_base, *(p_z - p_base), 40, "cyan", 1.6)

    # annotation
    txt = (
        f"base  (pose origin): [{base[0]:.3f},{base[1]:.3f},{base[2]:.3f}] m\n"
        f"tip/contact: [{tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}] m\n"
        f"euler_xyz(deg): [{args.euler_deg[0]:.1f},{args.euler_deg[1]:.1f},{args.euler_deg[2]:.1f}]\n"
        f"fingertip depth = {FINGERTIP_DEPTH_M} m"
    )
    ax.text(8, 28, txt, color="white", fontsize=10,
            bbox=dict(fc="black", alpha=0.55, ec="none"))

    ax.plot([], [], color="magenta", marker="+", ms=10, label="grasp base (pose origin)")
    ax.plot([], [], color="lime", marker="o", ms=8, label="grasp tip / contact point")
    ax.plot([], [], color="orange", lw=2, label="object seg contour")

    ax.set_title("GraspGenX grasp pose (camera frame) projected on RGB")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    out = args.out or os.path.join(os.getcwd(), "grasp_vis.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"saved overlay -> {out}")


if __name__ == "__main__":
    main()
