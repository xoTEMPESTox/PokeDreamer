"""Full-map layout and tile classification from memory and ROM/VRAM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from lab.color.native import decode_vram_tiles
from lab.color.rules import _rule

DEFAULT_LABELS_PATH = Path("lab/color/labels.json")
DEFAULT_PALETTE_PATH = Path("lab/color/palette.json")
DEFAULT_GRIDS_PATH = Path("work/viz3d/map_tilegrids.npz")


def build_map_grid(
    tile_ids: np.ndarray,
    vram_bytes: bytes | np.ndarray,
    tileset: int,
    labels: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (tile_ids, colour_indices_image, class_per_tile) from arrays.

    Args:
        tile_ids: (H, W) 2D array of 8x8 map tile IDs.
        vram_bytes: 6144 bytes of VRAM tile data (384 tiles * 16 bytes).
        tileset: tileset ID (e.g. 0).
        labels: mapping "<tileset>:<vram_idx>" -> class name.

    Returns:
        tile_ids: (H, W) int array.
        colour_indices_image: (H*8, W*8) uint8 array (values in 0..3).
        class_per_tile: (H, W) object array of class name strings.
    """
    H, W = tile_ids.shape
    if isinstance(vram_bytes, (bytes, bytearray)):
        vram_arr = np.frombuffer(vram_bytes, dtype=np.uint8)
    else:
        vram_arr = np.asarray(vram_bytes, dtype=np.uint8)
    vram_tiles = decode_vram_tiles(vram_arr)  # (384, 8, 8) uint8

    cidx_img = np.zeros((H * 8, W * 8), dtype=np.uint8)
    class_per_tile = np.empty((H, W), dtype=object)

    for r in range(H):
        for c in range(W):
            t = int(tile_ids[r, c])
            # A map tile id t uses VRAM index 256 + t if t < 128 else t
            v = 256 + t if t < 128 else t
            cidx_img[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = vram_tiles[v]

            k = f"{tileset}:{v}"
            # Apply label if present, else rules.py logic, else "other"
            cname = labels.get(k) or _rule(0, v) or "other"
            class_per_tile[r, c] = cname

    return tile_ids, cidx_img, class_per_tile


def full_map(
    state_path: str | Path,
    map_tilegrids_path: str | Path = DEFAULT_GRIDS_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a state and extract full map layout, 2-bit color index image, and tile classes.

    Args:
        state_path: path to the .state file.
        map_tilegrids_path: path to map_tilegrids.npz.
        labels_path: path to labels.json.

    Returns:
        tile_ids: (H, W) int array
        colour_indices_image: (H*8, W*8) uint8 array
        class_per_tile: (H, W) object array of class name strings
    """
    from lab.env import PokemonRed

    env = PokemonRed(state=state_path)
    try:
        vram = bytes(env.pyboy.memory[0x8000:0x9800])
        tileset = int(env.pyboy.memory[0xD367])
        map_id = int(env.pyboy.memory[0xD35E])
    finally:
        env.close()

    grids = np.load(map_tilegrids_path)
    grid_key = f"map{map_id}"
    if grid_key not in grids:
        raise KeyError(f"Map key {grid_key} not found in {map_tilegrids_path}")
    tile_ids = grids[grid_key]

    with open(labels_path, "r") as f:
        labels = json.load(f)

    return build_map_grid(tile_ids, vram, tileset, labels)


def map_to_rgb(
    cidx_img: np.ndarray,
    class_per_tile: np.ndarray,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
) -> np.ndarray:
    """Render full map to RGB (144/160 scaling not applied, 1 map pixel = 1 image pixel).

    Uses Game Boy grayscale (255, 170, 85, 0) with water tiles tinted blue,
    matching work/viz3d/viridian_full_map.png.
    """
    gray_lut = np.array([255, 170, 85, 0], dtype=np.uint8)
    base_gray = gray_lut[cidx_img]  # (H*8, W*8)
    rgb = np.repeat(base_gray[..., None], 3, axis=-1)

    # Tint water tiles blue
    H, W = class_per_tile.shape
    for r in range(H):
        for c in range(W):
            if class_per_tile[r, c] == "water":
                # Apply water tint [0.5, 0.7, 1.0] matching viridian_full_map.png
                patch = rgb[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
                patch[..., 0] = (patch[..., 0] * 0.5).astype(np.uint8)
                patch[..., 1] = (patch[..., 1] * 0.7).astype(np.uint8)
                patch[..., 2] = np.clip(patch[..., 2] * 1.0 + 85, 0, 255).astype(
                    np.uint8
                )

    return rgb


def check_path_alignment(
    state_path: str | Path = "work/route_states/seed100_map1.state",
    traj_path: str | Path = "work/video/traj/run_seed_100.npz",
    out_path: str | Path = "work/viz3d/check_path_on_map.png",
) -> None:
    """Draw recorded player positions on the full map and save verification image."""
    tile_ids, cidx_img, class_per_tile = full_map(state_path)
    rgb = map_to_rgb(cidx_img, class_per_tile)

    r_data = np.load(traj_path)
    map_ids = r_data["map_ids"]
    xs = r_data["xs"]
    ys = r_data["ys"]

    # Select steps on map 1
    idx = np.where(map_ids == 1)[0]
    out_img = rgb.copy()

    # Draw player paths: each player tile is (16x, 16y)
    # Draw path segments
    pts = []
    for i in range(len(idx)):
        step = idx[i]
        x, y = int(xs[step]), int(ys[step])
        # Center of 16x16 player tile
        px = x * 16 + 8
        py = y * 16 + 8
        pts.append((px, py))

        # Check walkable
        # player tile covers grid 2y..2y+1, 2x..2x+1
        sub_classes = class_per_tile[2 * y : 2 * y + 2, 2 * x : 2 * x + 2].flatten()
        # Draw a semi-transparent box or outline around player tile
        cv2.rectangle(
            out_img,
            (x * 16 + 1, y * 16 + 1),
            (x * 16 + 15, y * 16 + 15),
            (255, 60, 60),
            1,
        )

    # Draw lines connecting consecutive steps if distance is <= 2 tiles
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if dist <= 32:  # consecutive steps
            cv2.line(out_img, p0, p1, (0, 255, 255), 2)

    # Mark start and end points
    if pts:
        cv2.circle(out_img, pts[0], 4, (0, 255, 0), -1)  # start green
        cv2.circle(out_img, pts[-1], 4, (255, 0, 255), -1)  # end magenta

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(out_path)
    print(f"Saved path check image to {out_path} ({len(idx)} positions plotted)")


if __name__ == "__main__":
    check_path_alignment()
