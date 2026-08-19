#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a 16-bit depth PNG to a float32 depth.npy in meters.

GraspGenX's scene loader (graspgenx/utils/scene_loaders.py) expects
``depth.npy`` to be a 2D float32 array of metric depths in meters, with 0.0
marking an invalid/missing return. Real-world captures are commonly exported
as 16-bit PNGs in millimeters (e.g., RealSense ``.png`` exports), so the
default divisor is 1000.0 (mm -> m). Override with ``--depth_scale`` if your
PNG uses a different unit (e.g., 100.0 for centimeters, 1.0 for already-metric
values).

Usage
-----

    # mm PNG -> depth.npy (meters), default output alongside input
    python scripts/depth_png_to_npy.py assets/sample_data/real_world/02/depth.png

    # Explicit output path and unit
    python scripts/depth_png_to_npy.py \\
        assets/sample_data/real_world/02/depth.png \\
        --output assets/sample_data/real_world/02/depth.npy \\
        --depth_scale 1000.0
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a 16-bit depth PNG to float32 depth.npy (meters)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input depth PNG (uint16).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy path. Defaults to the input path with .npy extension.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Divisor to convert PNG values to meters (default 1000.0 for mm). "
        "Use 100.0 for cm, 1.0 if values are already in meters.",
    )
    parser.add_argument(
        "--invalid_value",
        type=float,
        default=0.0,
        help="Output value used for invalid (zero) pixels after scaling (default 0.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if the output file already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input depth PNG not found: {in_path}")

    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_suffix(".npy")
    )
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output {out_path} already exists. Use --overwrite to replace it."
        )

    img = Image.open(in_path)
    arr = np.asarray(img)
    if arr.dtype not in (np.uint16, np.int32, np.int64):
        if arr.dtype == np.uint8:
            raise ValueError(
                "Input PNG is 8-bit; this script expects a 16-bit depth PNG (uint16)."
            )
        arr = arr.astype(np.uint16, copy=False)

    valid_mask = arr > 0
    depth = np.where(valid_mask, arr.astype(np.float32) / float(args.depth_scale), args.invalid_value).astype(
        np.float32, copy=False
    )

    if valid_mask.any():
        vmin, vmax = float(depth[valid_mask].min()), float(depth[valid_mask].max())
    else:
        vmin, vmax = 0.0, 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth)

    print(f"[depth_png_to_npy] {in_path}")
    print(f"  input  shape={arr.shape} dtype={arr.dtype} "
          f"range=[{int(arr.min())}, {int(arr.max())}] (raw uint16)")
    print(f"  scale  /{args.depth_scale:g}  -> meters")
    print(f"  output {out_path}  shape={depth.shape} dtype={depth.dtype} "
          f"valid_range=[{vmin:.4f}, {vmax:.4f}] m  "
          f"invalid_pixels={(~valid_mask).sum()}")


if __name__ == "__main__":
    main()
