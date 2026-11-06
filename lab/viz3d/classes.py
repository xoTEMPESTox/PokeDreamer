"""Colour classes, palette inversion, sprite regions, and world stitching."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

DEFAULT_PALETTE_PATH = Path("lab/color/palette.json")

# 4-connected cross kernel for dilation
CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def load_palette(
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
) -> Tuple[List[str], Dict[Tuple[int, int, int], Tuple[int, int]]]:
    """Load palette.json into class names and an RGB -> (class_idx, shade) lookup table."""
    with open(palette_path, "r") as f:
        data = json.load(f)
    classes = data["classes"]
    names = list(classes.keys())
    lut: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    for c_idx, name in enumerate(names):
        for shade, rgb in enumerate(classes[name]):
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            # (245, 245, 240) collision: default to building_wall (shade 0)
            if rgb_tuple not in lut or name == "building_wall":
                lut[rgb_tuple] = (c_idx, shade)

    return names, lut


def frame_classes(
    rgb144: np.ndarray,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert (144, 160, 3) RGB frame to (cls, shade) with vectorized lookup and collision rule.

    Returns:
        cls: (144, 160) int8 array of class indices
        shade: (144, 160) uint8 array of shades (0..3)
    """
    names, lut = load_palette(palette_path)
    other_idx = names.index("other") if "other" in names else 21
    wall_idx = names.index("building_wall") if "building_wall" in names else -1
    npc_idx = names.index("sprite_npc") if "sprite_npc" in names else -1

    r = rgb144[..., 0].astype(np.int32)
    g = rgb144[..., 1].astype(np.int32)
    b = rgb144[..., 2].astype(np.int32)
    rgb_int = (r << 16) | (g << 8) | b

    unq_ints, inv = np.unique(rgb_int, return_inverse=True)

    cls_table = np.empty(len(unq_ints), dtype=np.int8)
    shade_table = np.empty(len(unq_ints), dtype=np.uint8)

    collision_val = (245 << 16) | (245 << 8) | 240
    has_collision = False

    for i, val in enumerate(unq_ints):
        rc = int((val >> 16) & 0xFF)
        gc = int((val >> 8) & 0xFF)
        bc = int(val & 0xFF)
        if (rc, gc, bc) in lut:
            c_idx, sh = lut[(rc, gc, bc)]
            cls_table[i] = c_idx
            shade_table[i] = sh
        else:
            cls_table[i] = other_idx
            shade_table[i] = 2

        if val == collision_val:
            has_collision = True

    cls_arr = cls_table[inv].reshape(rgb144.shape[:2])
    shade_arr = shade_table[inv].reshape(rgb144.shape[:2])

    # Collision rule for (245, 245, 240):
    # A pixel of that colour is sprite_npc if any 4-connected neighbour within 2 px
    # is a different sprite_npc shade (1, 2, or 3), else building_wall.
    if has_collision and npc_idx != -1 and wall_idx != -1:
        collision_mask = rgb_int == collision_val
        npc_other_shades = (cls_arr == npc_idx) & (shade_arr > 0)

        if np.any(npc_other_shades):
            # 2 iterations of 4-connected dilation covers Manhattan distance <= 2
            dilated = cv2.dilate(
                npc_other_shades.astype(np.uint8),
                CROSS_KERNEL,
                iterations=2,
            ).astype(bool)

            is_npc = collision_mask & dilated
            cls_arr[is_npc] = npc_idx
            shade_arr[is_npc] = 0

            is_wall = collision_mask & (~dilated)
            cls_arr[is_wall] = wall_idx
            shade_arr[is_wall] = 0
        else:
            # No other npc shades around, so all collision pixels are building_wall
            cls_arr[collision_mask] = wall_idx
            shade_arr[collision_mask] = 0

    return cls_arr, shade_arr


def sprite_regions(
    cls: np.ndarray,
    which: str | int = "sprite_npc",
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
    min_pixels: int = 20,
) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """Find connected 8-neighbour regions for a class, minimum min_pixels.

    Returns:
        List of (r0, r1, c0, c1, mask) tuples.
    """
    if isinstance(which, str):
        names, _ = load_palette(palette_path)
        target_idx = names.index(which)
    else:
        target_idx = which

    binary_mask = (cls == target_idx).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    regions: List[Tuple[int, int, int, int, np.ndarray]] = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue
        c0 = stats[label, cv2.CC_STAT_LEFT]
        r0 = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        c1 = c0 + w
        r1 = r0 + h
        sub_mask = labels[r0:r1, c0:c1] == label
        regions.append((r0, r1, c0, c1, sub_mask))

    return regions


def stitch(
    frames_rgb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    in_battle: np.ndarray | None = None,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
) -> Dict[str, Any]:
    """Stitch frames into a world-space class map using majority class and modal shade.

    Returns:
        dict with keys:
            cls: (H, W) int8 array (-1 for unseen)
            shade: (H, W) uint8 array
            origin: (row0, col0) tuple of world coordinates
            count: (H, W) int32 array of valid pixel counts
    """
    names, _ = load_palette(palette_path)
    ignore_classes = {
        names.index(n)
        for n in ("sprite_player", "sprite_npc", "text_box")
        if n in names
    }

    n_frames = len(frames_rgb)
    valid_indices = []
    for i in range(n_frames):
        if in_battle is not None and in_battle[i]:
            continue
        valid_indices.append(i)

    if not valid_indices:
        raise ValueError("No valid non-battle frames to stitch.")

    # Compute bounding box in world coordinates
    # world_col = screen_col + 16 * (xs - 4)
    # world_row = screen_row + 16 * (ys - 4)
    row_starts = [16 * (int(ys[i]) - 4) for i in valid_indices]
    col_starts = [16 * (int(xs[i]) - 4) for i in valid_indices]

    min_row = min(row_starts)
    max_row = max(r + 144 for r in row_starts)
    min_col = min(col_starts)
    max_col = max(c + 160 for c in col_starts)

    H = max_row - min_row
    W = max_col - min_col
    origin = (min_row, min_col)

    n_classes = len(names)
    class_counts = np.zeros((n_classes, H, W), dtype=np.int16)
    shade_counts = np.zeros((n_classes, 4, H, W), dtype=np.int16)

    for i in valid_indices:
        f_cls, f_shade = frame_classes(frames_rgb[i], palette_path=palette_path)
        r0 = 16 * (int(ys[i]) - 4) - min_row
        c0 = 16 * (int(xs[i]) - 4) - min_col

        # Vectorized accumulation per class
        for c_idx in range(n_classes):
            if c_idx in ignore_classes:
                continue
            c_mask = f_cls == c_idx
            if not np.any(c_mask):
                continue

            target_class_slice = class_counts[c_idx, r0 : r0 + 144, c0 : c0 + 160]
            target_class_slice += c_mask.astype(np.int16)

            for sh in range(4):
                sh_mask = c_mask & (f_shade == sh)
                if np.any(sh_mask):
                    target_shade_slice = shade_counts[
                        c_idx, sh, r0 : r0 + 144, c0 : c0 + 160
                    ]
                    target_shade_slice += sh_mask.astype(np.int16)

    total_count = class_counts.sum(axis=0)
    valid_mask = total_count > 0

    majority_cls = np.argmax(class_counts, axis=0).astype(np.int8)

    # For modal shade of majority class:
    # shade_counts: (n_classes, 4, H, W)
    modal_shades = np.argmax(shade_counts, axis=1)  # (n_classes, H, W)
    # Gather modal shade for majority class at each pixel
    # Advanced indexing on H, W
    grid_r, grid_c = np.indices((H, W))
    final_shade = modal_shades[majority_cls, grid_r, grid_c].astype(np.uint8)

    majority_cls[~valid_mask] = -1
    final_shade[~valid_mask] = 0

    return {
        "cls": majority_cls,
        "shade": final_shade,
        "origin": origin,
        "count": total_count.astype(np.int32),
    }


def agreement(
    frames_rgb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
) -> float:
    """Fraction of overlapping non-sprite pixels where consecutive moving frames agree."""
    names, _ = load_palette(palette_path)
    with open(palette_path, "r") as f:
        pal = json.load(f)["classes"]

    sprite_cols = set()
    for s_name in ("sprite_player", "sprite_npc", "text_box"):
        if s_name in pal:
            for rgb in pal[s_name]:
                sprite_cols.add(tuple(rgb))

    agreed = 0
    total = 0

    for i in range(len(frames_rgb) - 1):
        dx = int(xs[i + 1]) - int(xs[i])
        dy = int(ys[i + 1]) - int(ys[i])
        if dx == 0 and dy == 0:
            continue

        f0 = frames_rgb[i]
        f1 = frames_rgb[i + 1]

        # Shift: world_col = screen_col + 16 * dx
        dr = 16 * dy
        dc = 16 * dx

        r0_min = max(0, dr)
        r0_max = min(144, 144 + dr)
        c0_min = max(0, dc)
        c0_max = min(160, 160 + dc)

        r1_min = max(0, -dr)
        r1_max = min(144, 144 - dr)
        c1_min = max(0, -dc)
        c1_max = min(160, 160 - dc)

        sub0 = f0[r0_min:r0_max, c0_min:c0_max]
        sub1 = f1[r1_min:r1_max, c1_min:c1_max]

        if sub0.size == 0 or sub1.size == 0:
            continue

        # Check sprite pixels
        sub0_flat = sub0.reshape(-1, 3)
        sub1_flat = sub1.reshape(-1, 3)

        m0 = np.array([tuple(p) not in sprite_cols for p in sub0_flat]).reshape(
            sub0.shape[:2]
        )
        m1 = np.array([tuple(p) not in sprite_cols for p in sub1_flat]).reshape(
            sub1.shape[:2]
        )
        valid = m0 & m1

        eq = np.all(sub0 == sub1, axis=-1) & valid
        agreed += int(np.sum(eq))
        total += int(np.sum(valid))

    if total == 0:
        return 1.0

    return agreed / total
