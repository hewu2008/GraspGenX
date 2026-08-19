#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize a depth.npy frame with interactive mouse-position readout.

Renders the depth map with a colormap, and live-updates an annotation showing
the pixel coordinate and depth value (in meters) under the cursor. Zero-depth
pixels are treated as invalid returns and reported as "invalid".

Usage
-----

    python scripts/vis_depth.py assets/sample_data/real_world/01/depth.npy
    python scripts/vis_depth.py assets/sample_data/real_world/01/depth.npy \
        --rgb assets/sample_data/real_world/01/rgb.png \
        --max_depth 3.0 --cmap turbo
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a depth.npy frame with mouse-position depth readout."
    )
    parser.add_argument(
        "depth_path",
        type=str,
        help="Path to depth.npy (HxW float32, depth in meters).",
    )
    parser.add_argument(
        "--rgb",
        type=str,
        default=None,
        help="Optional RGB image path to display side-by-side.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Upper bound (meters) for colormap scaling. Defaults to the 99th percentile "
        "of valid (nonzero) depths.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap name for the depth image.",
    )
    parser.add_argument(
        "--invalid_value",
        type=float,
        default=0.0,
        help="Depth value marking an invalid/missing return (default 0.0).",
    )
    return parser.parse_args()


def _load_rgb(rgb_path):
    if not rgb_path:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    rgb_path = Path(rgb_path)
    if not rgb_path.is_file():
        return None
    return np.asarray(Image.open(rgb_path))


def main():
    args = parse_args()
    depth_path = Path(args.depth_path)
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    depth = np.load(depth_path).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

    h, w = depth.shape
    valid_mask = depth != args.invalid_value
    valid_depths = depth[valid_mask]

    if valid_depths.size > 0:
        auto_max = float(np.percentile(valid_depths, 99))
    else:
        auto_max = 1.0
    max_depth = args.max_depth if args.max_depth is not None else auto_max

    depth_disp = np.where(valid_mask, np.clip(depth, 0.0, max_depth), np.nan)

    rgb = _load_rgb(args.rgb)

    n_cols = 2 if rgb is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.4 * n_cols, 4.8))
    if n_cols == 1:
        axes = [axes]

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        ax_depth = axes[1]
    else:
        ax_depth = axes[0]

    im = ax_depth.imshow(
        depth_disp,
        cmap=args.cmap,
        vmin=0.0,
        vmax=max_depth,
    )
    ax_depth.set_title(
        f"Depth  ({depth_path.name}  {h}x{w}  range=[{valid_depths.min():.4f},"
        f" {valid_depths.max():.4f}] m  max_disp={max_depth:.3f} m)"
    )
    ax_depth.axis("off")
    fig.colorbar(im, ax=ax_depth, label="depth [m]")

    hover_text = fig.text(
        0.5,
        0.02,
        "Move cursor over the depth image...",
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
    )

    def on_motion(event):
        if event.inaxes is not ax_depth:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            return
        d = float(depth[iy, ix])
        is_valid = valid_mask[iy, ix]
        status = f"{d:.4f} m" if is_valid else "invalid (no return)"
        hover_text.set_text(f"pixel (u={ix:4d}, v={iy:4d})  depth = {status}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


if __name__ == "__main__":
    main()
