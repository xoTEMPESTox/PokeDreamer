"""Per-tile native colorization for real Pokémon Red frames.

Implements exact Game Boy DMG tile decoding, scroll/window mapping, sprite rendering,
validation against PyBoy screen buffer, tile inventory collection, contact sheets,
and palette-based colorization.
"""
from __future__ import annotations

import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Screen dimensions
SCREEN_H, SCREEN_W = 144, 160
_Y_COORDS, _X_COORDS = np.meshgrid(np.arange(SCREEN_H, dtype=np.int32), np.arange(SCREEN_W, dtype=np.int32), indexing="ij")
_BITS = np.arange(7, -1, -1, dtype=np.uint8)

# 4 Game Boy shades rendered by PyBoy in DMG mode:
# shade 0: white (255)
# shade 1: light grey (153)
# shade 2: dark grey (85)
# shade 3: black (0)
DMG_SHADES = np.array([255, 153, 85, 0], dtype=np.uint8)

# Visualization RGB palette for contact sheets: 0 = white, 3 = black
DMG_PALETTE_RGB = np.array([
    [255, 255, 255],  # 0: white
    [170, 170, 170],  # 1: light grey
    [85, 85, 85],     # 2: dark grey
    [0, 0, 0],        # 3: black
], dtype=np.uint8)

# Default modern pixel-art color ramps (saturated-but-soft, 0 lightest .. 3 darkest)
DEFAULT_CLASSES: Dict[str, List[List[int]]] = {
    "grass": [[220, 245, 180], [140, 210, 100], [70, 160, 60], [30, 80, 40]],
    "tall_grass": [[210, 240, 160], [120, 195, 80], [55, 140, 50], [25, 70, 35]],
    "tree": [[200, 235, 185], [100, 180, 110], [45, 120, 70], [20, 60, 40]],
    "path": [[245, 238, 215], [215, 195, 155], [160, 135, 95], [90, 70, 45]],
    "water": [[210, 245, 255], [110, 190, 240], [50, 130, 205], [20, 65, 130]],
    "ledge": [[240, 230, 200], [200, 180, 140], [140, 115, 80], [80, 60, 35]],
    "fence": [[235, 225, 205], [190, 165, 130], [130, 100, 70], [75, 50, 30]],
    "flowers": [[255, 230, 240], [240, 130, 160], [190, 60, 90], [100, 25, 50]],
    "building_wall": [[245, 245, 240], [200, 200, 195], [140, 140, 140], [70, 70, 75]],
    "roof_red": [[255, 225, 215], [235, 110, 95], [180, 55, 45], [100, 25, 20]],
    "roof_blue": [[220, 235, 255], [115, 160, 225], [55, 95, 175], [25, 45, 100]],
    "door": [[245, 230, 200], [195, 155, 100], [135, 95, 50], [70, 45, 20]],
    "sign": [[245, 240, 220], [205, 180, 135], [145, 115, 75], [80, 55, 30]],
    "floor_indoor": [[248, 242, 228], [218, 198, 168], [165, 138, 108], [98, 75, 52]],
    "counter": [[245, 240, 235], [205, 185, 170], [145, 120, 105], [80, 60, 50]],
    "carpet": [[250, 225, 225], [215, 105, 115], [160, 50, 65], [85, 20, 30]],
    "text_box": [[252, 252, 248], [195, 195, 200], [110, 110, 120], [35, 35, 45]],
    "sprite_player": [[250, 245, 240], [225, 85, 75], [150, 45, 40], [30, 30, 40]],
    "sprite_npc": [[245, 245, 240], [100, 175, 190], [45, 110, 130], [25, 45, 60]],
    "battle_bg": [[240, 248, 235], [180, 210, 175], [120, 150, 120], [50, 75, 60]],
    "battle_hp_bar": [[245, 250, 245], [120, 200, 120], [70, 150, 70], [30, 50, 35]],
    "other": [[245, 245, 245], [175, 180, 185], [105, 110, 120], [35, 38, 45]],
}


def format_key(tileset: int, layer: int, tile: int) -> str:
    """Format a key as 'tileset:layer:tile'."""
    return f"{int(tileset)}:{int(layer)}:{int(tile)}"


def parse_key(key: str) -> Tuple[int, int, int]:
    """Parse a 'tileset:layer:tile' key into integers."""
    parts = key.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid key format: {key}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def map_background_scroll(
    scx: int,
    scy: int,
    bg_map: np.ndarray,
    screen_shape: Tuple[int, int] = (144, 160),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure function mapping screen pixels to background tile IDs and sub-tile coordinates.

    Args:
        scx: Horizontal scroll offset (0..255).
        scy: Vertical scroll offset (0..255).
        bg_map: 2D array of tile IDs of shape (H_tiles, W_tiles).
        screen_shape: (H, W) screen dimensions in pixels.

    Returns:
        tiles: (H, W) int16 array of tile IDs.
        ty: (H, W) uint8 array of pixel row offsets inside tile (0..7).
        tx: (H, W) uint8 array of pixel col offsets inside tile (0..7).
    """
    H, W = screen_shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    map_h_tiles, map_w_tiles = bg_map.shape
    map_h_px = map_h_tiles * 8
    map_w_px = map_w_tiles * 8

    bg_y = (y_coords + scy) % map_h_px
    bg_x = (x_coords + scx) % map_w_px

    tile_r = bg_y // 8
    tile_c = bg_x // 8

    tiles = bg_map[tile_r, tile_c].astype(np.int16)
    ty = (bg_y % 8).astype(np.uint8)
    tx = (bg_x % 8).astype(np.uint8)
    return tiles, ty, tx


def map_window(
    wx: int,
    wy: int,
    win_map: np.ndarray,
    screen_shape: Tuple[int, int] = (144, 160),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure function mapping screen pixels to window tile IDs and sub-tile coordinates.

    Args:
        wx: Window X register (WX = 7 means left edge is at x=0).
        wy: Window Y register (WY = 0 means top edge is at y=0).
        win_map: 2D array of tile IDs of shape (H_tiles, W_tiles).
        screen_shape: (H, W) screen dimensions in pixels.

    Returns:
        mask: (H, W) bool array, True where window is visible on screen.
        tiles: (H, W) int16 array of tile IDs.
        ty: (H, W) uint8 array of pixel row offsets inside tile (0..7).
        tx: (H, W) uint8 array of pixel col offsets inside tile (0..7).
    """
    H, W = screen_shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    win_wx = wx - 7
    mask = (y_coords >= wy) & (x_coords >= max(0, win_wx)) & (win_wx < W) & (wy < H)

    map_h_tiles, map_w_tiles = win_map.shape
    map_h_px = map_h_tiles * 8
    map_w_px = map_w_tiles * 8

    win_y = (y_coords - wy) % map_h_px
    win_x = (x_coords - win_wx) % map_w_px

    tile_r = win_y // 8
    tile_c = win_x // 8

    tiles = win_map[tile_r, tile_c].astype(np.int16)
    ty = (win_y % 8).astype(np.uint8)
    tx = (win_x % 8).astype(np.uint8)
    return mask, tiles, ty, tx


def decode_vram_tiles(vram_bytes: np.ndarray | bytes | list) -> np.ndarray:
    """Decode 2BPP Game Boy tile graphics from VRAM (0x8000-0x97FF) into 2-bit color indices.

    Args:
        vram_bytes: 6144 bytes of tile data for 384 tiles (16 bytes per tile).
    Returns:
        tiles_cidx: (384, 8, 8) uint8 array with values in {0, 1, 2, 3}.
    """
    raw = np.asarray(vram_bytes, dtype=np.uint8).reshape(384, 8, 2)
    b1 = raw[:, :, 0:1]
    b2 = raw[:, :, 1:2]
    bit1 = (b1 >> _BITS) & 1
    bit2 = (b2 >> _BITS) & 1
    return (((bit2 << 1) | bit1)).astype(np.uint8)


def decode_tile_map(map_bytes: np.ndarray | bytes | list, tiledata_select: bool) -> np.ndarray:
    """Decode a 32x32 Game Boy tilemap using unsigned or signed addressing.

    Args:
        map_bytes: 1024 bytes of tile indices from 0x9800 or 0x9C00.
        tiledata_select: LCDC bit 4. True for 0x8000-0x8FFF (unsigned 0..255),
                         False for 0x8800-0x97FF (signed -128..127 -> 128..383).
    Returns:
        (32, 32) int16 array of tile identifiers (0..383).
    """
    raw = np.asarray(map_bytes, dtype=np.uint8).reshape(32, 32)
    if tiledata_select:
        return raw.astype(np.int16)
    return ((raw ^ 0x80).astype(np.int16) + 128)


def rebuild_screen_shades(cidx: np.ndarray, pal: np.ndarray, shade_lut: Optional[np.ndarray] = None) -> np.ndarray:
    """Rebuild Game Boy screen shades from 2-bit color indices and palette registers.

    Args:
        cidx: (H, W) uint8 array of 2-bit color indices {0, 1, 2, 3}.
        pal: (H, W) uint8 array of palette register bytes (BGP/OBP0/OBP1).
        shade_lut: (4,) uint8 array of shades (default PyBoy DMG shades [255, 153, 85, 0]).
    Returns:
        (H, W) uint8 array of pixel shades.
    """
    if shade_lut is None:
        shade_lut = DMG_SHADES
    shade_idx = (pal >> (cidx * 2)) & 0b11
    return shade_lut[shade_idx]


def pixel_tiles(pyboy) -> Dict[str, Any]:
    """Compute per-pixel tile identification, layer, and 2-bit color index for current frame.

    Computes from tile maps, scroll and window registers, LCDC bits, and OAM sprites
    with sprite priority over background where sprite pixels are non-transparent.

    Args:
        pyboy: A running PyBoy instance.
    Returns:
        dict with:
            'layer': (144, 160) uint8 array (0 background, 1 window, 2 sprite)
            'tile': (144, 160) int16 array of tile IDs
            'cidx': (144, 160) uint8 array of 2-bit colour index (0..3) before palette
            'pal': (144, 160) uint8 array of palette register values for validation
            'tileset': int, current map tileset id from RAM 0xD367
            'in_battle': bool, whether in battle (RAM 0xD057 != 0)
    """
    tileset = int(pyboy.memory[0xD367])
    in_battle = bool(pyboy.memory[0xD057] != 0)

    # Game Boy registers
    lcdc = pyboy.memory[0xFF40]
    scy = pyboy.memory[0xFF42]
    scx = pyboy.memory[0xFF43]
    wy = pyboy.memory[0xFF4A]
    wx = pyboy.memory[0xFF4B]
    bgp = pyboy.memory[0xFF47]
    obp0 = pyboy.memory[0xFF48]
    obp1 = pyboy.memory[0xFF49]

    # LCDC flags
    # bit 7: LCD enable
    # bit 6: Window tile map area (0: 9800, 1: 9C00)
    # bit 5: Window enable
    # bit 4: BG & Window tile data area (0: 8800 signed, 1: 8000 unsigned)
    # bit 3: BG tile map area (0: 9800, 1: 9C00)
    # bit 2: OBJ size (0: 8x8, 1: 8x16)
    # bit 1: OBJ enable
    # bit 0: BG & Window enable
    lcd_enable = bool(lcdc & 0x80)
    win_map_offset = 0x9C00 if (lcdc & 0x40) else 0x9800
    win_enable = bool(lcdc & 0x20)
    tiledata_select = bool(lcdc & 0x10)
    bg_map_offset = 0x9C00 if (lcdc & 0x08) else 0x9800
    sprite_height = 16 if (lcdc & 0x04) else 8
    sprite_enable = bool(lcdc & 0x02)
    bg_enable = bool(lcdc & 0x01)

    H, W = 144, 160

    if not lcd_enable:
        return {
            "layer": np.zeros((H, W), dtype=np.uint8),
            "tile": np.zeros((H, W), dtype=np.int16),
            "cidx": np.zeros((H, W), dtype=np.uint8),
            "pal": np.full((H, W), bgp, dtype=np.uint8),
            "tileset": tileset,
            "in_battle": in_battle,
            "tiles_cidx": np.zeros((384, 8, 8), dtype=np.uint8),
        }

    # Decode VRAM tiles (0x8000..0x97FF) through PyBoy memory API
    vram_bytes = pyboy.memory[0x8000:0x9800]
    tiles_cidx = decode_vram_tiles(vram_bytes)

    # Decode tile maps
    bg_map = decode_tile_map(pyboy.memory[bg_map_offset : bg_map_offset + 1024], tiledata_select)
    win_map = decode_tile_map(pyboy.memory[win_map_offset : win_map_offset + 1024], tiledata_select)

    # 1. Background layer
    if bg_enable:
        bg_tiles, bg_ty, bg_tx = map_background_scroll(scx, scy, bg_map, (H, W))
        layer = np.zeros((H, W), dtype=np.uint8)
        tile = bg_tiles.copy()
        cidx = tiles_cidx[bg_tiles, bg_ty, bg_tx].astype(np.uint8)
    else:
        layer = np.zeros((H, W), dtype=np.uint8)
        tile = np.zeros((H, W), dtype=np.int16)
        cidx = np.zeros((H, W), dtype=np.uint8)

    pal = np.full((H, W), bgp, dtype=np.uint8)

    # 2. Window layer
    win_wx = wx - 7
    if win_enable and wy < H and win_wx < W:
        win_mask, win_tiles, win_ty, win_tx = map_window(wx, wy, win_map, (H, W))
        layer[win_mask] = 1
        tile[win_mask] = win_tiles[win_mask]
        cidx[win_mask] = tiles_cidx[win_tiles[win_mask], win_ty[win_mask], win_tx[win_mask]]
        pal[win_mask] = bgp

    # 3. Sprite (OBJ) layer
    if sprite_enable:
        oam = [pyboy.memory[0xFE00 + i] for i in range(160)]
        sprites_per_line: List[List[Tuple[int, int]]] = [[] for _ in range(H)]
        for n in range(0, 160, 4):
            sy = oam[n] - 16
            sx = oam[n + 1] - 8
            if sy < H and sy + sprite_height > 0 and -8 < sx < W:
                for ly in range(max(0, sy), min(H, sy + sprite_height)):
                    if len(sprites_per_line[ly]) < 10:
                        sprites_per_line[ly].append((sx, n))

        for ly in range(H):
            line_sprites = sprites_per_line[ly]
            if not line_sprites:
                continue

            # Non-CGB sprite priority: lower x has higher priority, tie-breaker lower OAM index n.
            # Sort descending so highest priority sprite draws last and stays on top.
            line_sprites.sort(key=lambda item: (item[0] << 16) | item[1], reverse=True)

            for sx, n in line_sprites:
                sy = oam[n] - 16
                tile_idx = oam[n + 2]
                if sprite_height == 16:
                    tile_idx &= 0xFE
                attrs = oam[n + 3]
                xflip = bool(attrs & 0x20)
                yflip = bool(attrs & 0x40)
                bg_priority = bool(attrs & 0x80)
                sp_pal = obp1 if (attrs & 0x10) else obp0

                dy = ly - sy
                yy = (sprite_height - 1 - dy) if yflip else dy
                actual_tile = (tile_idx + 1) if yy >= 8 else tile_idx
                actual_yy = yy % 8

                for dx in range(8):
                    px = sx + dx
                    if 0 <= px < W:
                        xx = (7 - dx) if xflip else dx
                        sp_cidx = tiles_cidx[actual_tile, actual_yy, xx]
                        if sp_cidx != 0:
                            # If BG priority bit is set, sprite appears behind BG color 1, 2, 3
                            if bg_priority and cidx[ly, px] != 0:
                                pass
                            else:
                                layer[ly, px] = 2
                                tile[ly, px] = actual_tile
                                cidx[ly, px] = sp_cidx
                                pal[ly, px] = sp_pal

    return {
        "layer": layer,
        "tile": tile,
        "cidx": cidx,
        "pal": pal,
        "tileset": tileset,
        "in_battle": in_battle,
        "tiles_cidx": tiles_cidx,
    }


def validate(states: Optional[List[Tuple[str, int]]] = None, total_min_frames: int = 200) -> float:
    """Validate rebuild_screen against PyBoy's actual screen buffer on >= 200 frames.

    Includes walking, mid-scroll frames, text boxes, Pokémon Center, and battle.
    Returns pixel agreement fraction (0..1).
    """
    from lab.env import PokemonRed

    if states is None:
        states = [
            ("work/route_states/to_grass.state", 100),
            ("work/route_states/leave_viridian_pc.state", 40),
            ("work/route_states/seed100_map51.state", 40),
            ("work/route_states/seed100_map13.state", 40),
        ]

    total_pixels = 0
    matching_pixels = 0
    frames_checked = 0
    rng = random.Random(42)

    for st_path, n_steps in states:
        if not Path(st_path).exists():
            continue
        with PokemonRed(state=st_path) as env:
            for s in range(n_steps):
                action = "a" if (s % 10 == 0) else rng.choice(["down", "up", "left", "right"])
                env.step(action)
                px = pixel_tiles(env.pyboy)
                rebuilt = rebuild_screen_shades(px["cidx"], px["pal"])
                expected = env.pyboy.screen.ndarray[:, :, 0]
                matching_pixels += int((rebuilt == expected).sum())
                total_pixels += 144 * 160
                frames_checked += 1

    # Also collect battle frames
    with PokemonRed(state="work/route_states/to_grass.state") as env:
        for _ in range(400):
            if env.pyboy.memory[0xD057] != 0:
                px = pixel_tiles(env.pyboy)
                rebuilt = rebuild_screen_shades(px["cidx"], px["pal"])
                expected = env.pyboy.screen.ndarray[:, :, 0]
                matching_pixels += int((rebuilt == expected).sum())
                total_pixels += 144 * 160
                frames_checked += 1
                env.step("a")
                if frames_checked >= max(total_min_frames + 20, 250):
                    break
            else:
                env.step(rng.choice(["down", "left", "right", "up", "a"]))

    agreement = matching_pixels / max(1, total_pixels)
    print(f"Validation complete: {frames_checked} frames tested, agreement = {agreement * 100:.4f}% ({matching_pixels}/{total_pixels} pixels)")
    return agreement


def create_skeleton_palette(collected_keys: List[str]) -> Dict[str, Any]:
    """Create skeleton palette with ~22 classes and map all collected keys to 'other'."""
    palette = {
        "classes": {cls: [list(c) for c in ramp] for cls, ramp in DEFAULT_CLASSES.items()},
        "tiles": {str(k): "other" for k in collected_keys},
    }
    return palette


def colorize(px: Dict[str, Any], palette: Dict[str, Any]) -> np.ndarray:
    """Colorize a frame given pixel_tiles dict and palette dict.

    Each pixel takes the colour palette['classes'][class_of_key][cidx].

    Args:
        px: Dict from pixel_tiles(pyboy) with 'layer', 'tile', 'cidx', 'tileset'.
        palette: Dict with 'classes' (name -> 4x3 RGB) and 'tiles' (key -> class).
    Returns:
        (144, 160, 3) uint8 RGB array.
    """
    layer = px["layer"]
    tile = px["tile"]
    cidx = px["cidx"]
    tileset = px["tileset"]

    classes = palette["classes"]
    tile_map = palette.get("tiles", {})
    other_ramp = np.asarray(classes.get("other", DEFAULT_CLASSES["other"]), dtype=np.uint8)

    H, W = layer.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # Vectorized lookup by unique (layer, tile) pairs in this frame
    stacked = np.stack([layer.ravel(), tile.ravel()], axis=1)
    unique_pairs = np.unique(stacked, axis=0)

    for l_val, t_val in unique_pairs:
        key = f"{tileset}:{l_val}:{t_val}"
        cls_name = tile_map.get(key, "other")
        ramp = np.asarray(classes.get(cls_name, other_ramp), dtype=np.uint8)
        mask = (layer == l_val) & (tile == t_val)
        out[mask] = ramp[cidx[mask]]

    return out


def collect(
    out: str = "work/color/native/tiles.npz",
    steps_per_state: int = 1500,
    min_battles: int = 5,
    states_dir: str = "work/route_states",
    state_indices: Optional[List[int]] = None,
    checkpoint_file: str = "work/color/native/_collect_checkpoint.pkl",
    do_battles: bool = True,
    finalize: bool = True,
) -> Dict[str, Any]:
    """Collect tile usage inventory, 8x8 cidx patterns, and 24x24 context crops from route states.

    - From every route state, takes 1,500 steps of random walk (pressing A every 10th step).
    - From to_grass.state, walks randomly in Route 1 grass until at least min_battles occur.
    - Saves tiles.npz and reports key counts per tileset and per layer.
    """
    import pickle
    from lab.env import PokemonRed

    state_files = sorted(glob.glob(f"{states_dir}/*.state"))
    ckpt_path = Path(checkpoint_file)

    # Load existing checkpoint if present
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            saved = pickle.load(f)
            key_counts = saved["key_counts"]
            key_images = saved["key_images"]
            key_crops = saved["key_crops"]
            key_interior = saved["key_interior"]
            battle_counts = saved["battle_counts"]
            battles_seen = saved.get("battles_seen", 0)
    else:
        key_counts: Dict[Tuple[int, int, int], int] = {}
        key_images: Dict[Tuple[int, int, int], np.ndarray] = {}
        key_crops: Dict[Tuple[int, int, int], np.ndarray] = {}
        key_interior: Dict[Tuple[int, int, int], bool] = {}
        battle_counts: Dict[Tuple[int, int, int], int] = {}
        battles_seen = 0

    rng = random.Random(1337)
    actions = ["down", "left", "right", "up"]

    def record_frame(pb, px):
        ts = px["tileset"]
        in_b = px["in_battle"]
        layer = px["layer"]
        tile = px["tile"]
        cidx = px["cidx"]
        tiles_cidx = px.get("tiles_cidx")
        if tiles_cidx is None:
            tiles_cidx = decode_vram_tiles(pb.memory[0x8000:0x9800])
        padded_cidx = None

        pairs = layer.astype(np.int32) * 1000 + tile.astype(np.int32)
        u_pairs, counts = np.unique(pairs, return_counts=True)

        for u_p, cnt in zip(u_pairs, counts):
            l_val = int(u_p // 1000)
            t_val = int(u_p % 1000)
            k = (ts, l_val, t_val)

            key_counts[k] = key_counts.get(k, 0) + int(cnt)
            if in_b:
                battle_counts[k] = battle_counts.get(k, 0) + int(cnt)

            if k not in key_images:
                if 0 <= t_val < 384:
                    key_images[k] = tiles_cidx[t_val].copy()
                else:
                    key_images[k] = np.zeros((8, 8), dtype=np.uint8)

            has_crop = k in key_crops
            is_int = key_interior.get(k, False)
            if not has_crop or not is_int:
                mask = (layer == l_val) & (tile == t_val)
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    dist = (ys - 72) ** 2 + (xs - 80) ** 2
                    best_i = int(np.argmin(dist))
                    cy, cx = int(ys[best_i]), int(xs[best_i])

                    if l_val in (0, 1):
                        scy = pb.memory[0xFF42]
                        scx = pb.memory[0xFF43]
                        sub_y = (cy + scy) % 8
                        sub_x = (cx + scx) % 8
                        y0 = cy - sub_y
                        x0 = cx - sub_x
                    else:
                        y0 = max(0, min(144 - 8, cy - 4))
                        x0 = max(0, min(160 - 8, cx - 4))

                    now_int = (8 <= y0 <= 128 and 8 <= x0 <= 144)
                    if not has_crop or (now_int and not is_int):
                        if padded_cidx is None:
                            padded_cidx = np.pad(cidx, ((24, 24), (24, 24)), mode="edge")
                        crop_y = y0 + 16
                        crop_x = x0 + 16
                        crop = padded_cidx[crop_y : crop_y + 24, crop_x : crop_x + 24]
                        if crop.shape == (24, 24):
                            key_crops[k] = crop.copy()
                            key_interior[k] = now_int

    # Determine which state files to run
    if state_indices is not None:
        target_files = [state_files[i] for i in state_indices if i < len(state_files)]
    else:
        target_files = state_files

    if target_files:
        print(f"Processing {len(target_files)} route states ({steps_per_state} steps each)...")
        for idx, st_path in enumerate(target_files):
            t_st = time.time()
            with PokemonRed(state=st_path) as env:
                pb = env.pyboy
                for s in range(steps_per_state):
                    btn = "a" if (s % 10 == 0) else rng.choice(actions)
                    pb.button(btn, 8)
                    pb.tick(24, render=True)
                    px = pixel_tiles(pb)
                    record_frame(pb, px)
            print(f"  {Path(st_path).name}: {steps_per_state} steps in {time.time() - t_st:.1f} s (distinct keys so far: {len(key_counts)})")

    # Battle collection if requested
    if do_battles and battles_seen < min_battles:
        print(f"Walking in grass from to_grass.state until >= {min_battles} battles occur (current: {battles_seen})...")
        with PokemonRed(state="work/route_states/to_grass.state") as env:
            pb = env.pyboy
            for s in range(3000):
                in_b = pb.memory[0xD057] != 0
                if in_b:
                    battles_seen += 1
                    print(f"  Battle {battles_seen} triggered at step {s}!")
                    for _ in range(60):
                        px = pixel_tiles(pb)
                        record_frame(pb, px)
                        pb.button("a", 8)
                        pb.tick(24, render=True)
                        if pb.memory[0xD057] == 0:
                            break
                    if battles_seen >= min_battles:
                        break
                else:
                    btn = "a" if (s % 10 == 0) else rng.choice(actions)
                    pb.button(btn, 8)
                    pb.tick(24, render=True)
                    px = pixel_tiles(pb)
                    record_frame(pb, px)

    # Save checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "key_counts": key_counts,
            "key_images": key_images,
            "key_crops": key_crops,
            "key_interior": key_interior,
            "battle_counts": battle_counts,
            "battles_seen": battles_seen,
        }, f)

    if not finalize:
        return {"distinct_keys": len(key_counts), "battles_seen": battles_seen}

    # Sort keys by usage descending
    sorted_keys = sorted(key_counts.keys(), key=lambda k: key_counts[k], reverse=True)
    K = len(sorted_keys)

    tileset_arr = np.array([k[0] for k in sorted_keys], dtype=np.int16)
    layer_arr = np.array([k[1] for k in sorted_keys], dtype=np.uint8)
    tile_arr = np.array([k[2] for k in sorted_keys], dtype=np.int16)
    key_strs = np.array([format_key(*k) for k in sorted_keys])
    counts_arr = np.array([key_counts[k] for k in sorted_keys], dtype=np.int64)
    battle_counts_arr = np.array([battle_counts.get(k, 0) for k in sorted_keys], dtype=np.int64)

    images_arr = np.zeros((K, 8, 8), dtype=np.uint8)
    crops_arr = np.zeros((K, 24, 24), dtype=np.uint8)
    for i, k in enumerate(sorted_keys):
        images_arr[i] = key_images.get(k, np.zeros((8, 8), dtype=np.uint8))
        c_img = key_crops.get(k)
        if c_img is not None and c_img.shape == (24, 24):
            crops_arr[i] = c_img
        elif c_img is not None:
            pad_y = max(0, 24 - c_img.shape[0])
            pad_x = max(0, 24 - c_img.shape[1])
            crops_arr[i] = np.pad(c_img, ((0, pad_y), (0, pad_x)), mode="edge")[:24, :24]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keys=key_strs,
        tileset=tileset_arr,
        layer=layer_arr,
        tile=tile_arr,
        counts=counts_arr,
        battle_counts=battle_counts_arr,
        images=images_arr,
        crops=crops_arr,
    )
    print(f"Saved inventory to {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # Report breakdown
    report_breakdown(tileset_arr, layer_arr)

    return {
        "keys": key_strs,
        "tileset": tileset_arr,
        "layer": layer_arr,
        "tile": tile_arr,
        "counts": counts_arr,
        "battle_counts": battle_counts_arr,
        "images": images_arr,
        "crops": crops_arr,
    }


def report_breakdown(tileset_arr: np.ndarray, layer_arr: np.ndarray) -> None:
    """Print the number of keys per tileset and per layer."""
    print("\n=== Key Inventory Breakdown ===")
    unique_ts = np.unique(tileset_arr)
    for ts in sorted(unique_ts):
        mask = (tileset_arr == ts)
        print(f"Tileset {ts:2d}: {mask.sum():4d} keys (bg: {(mask & (layer_arr == 0)).sum()}, win: {(mask & (layer_arr == 1)).sum()}, sprite: {(mask & (layer_arr == 2)).sum()})")

    unique_layers = np.unique(layer_arr)
    layer_names = {0: "Background (0)", 1: "Window (1)", 2: "Sprite (2)"}
    print("--- Per Layer ---")
    for l in sorted(unique_layers):
        l_cnt = (layer_arr == l).sum()
        print(f"Layer {l} ({layer_names.get(l, str(l))}): {l_cnt:4d} keys")
    print(f"Total keys: {len(tileset_arr)}\n")


def contact_sheets(
    inventory_path: str = "work/color/native/tiles.npz",
    out_dir: str = "work/color/native",
    cols: int = 6,
) -> List[str]:
    """Generate visual contact sheets: one per tileset, one for sprites, one for battle frames.

    Each tile enlarged 6x with key printed under it, next to it 24x24 context crop enlarged 4x
    with the tile outlined. Colour index 0 = white, 3 = black. Sorted by usage.
    """
    data = np.load(inventory_path)
    keys = data["keys"]
    tilesets = data["tileset"]
    layers = data["layer"]
    tiles = data["tile"]
    counts = data["counts"]
    battle_counts = data["battle_counts"]
    images = data["images"]
    crops = data["crops"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_sheets: List[str] = []

    font = ImageFont.load_default()

    # Define groups:
    # 1. Per tileset (layers 0 & 1)
    unique_ts = sorted(np.unique(tilesets))
    groups: Dict[str, np.ndarray] = {}
    for ts in unique_ts:
        mask = (tilesets == ts) & (layers != 2)
        if mask.any():
            groups[f"sheet_tileset_{ts}.png"] = np.where(mask)[0]

    # 2. Sprites (layer 2)
    sp_mask = (layers == 2)
    if sp_mask.any():
        groups["sheet_sprites.png"] = np.where(sp_mask)[0]

    # 3. Battle frames
    b_mask = (battle_counts > 0)
    if b_mask.any():
        groups["sheet_battle.png"] = np.where(b_mask)[0]

    # Layout dimensions
    # 8x8 tile enlarged 6x -> 48x48
    # 24x24 context crop enlarged 4x -> 96x96
    # Cell width: 8 margin + 48 tile + 8 gap + 96 crop + 8 margin = 168 px
    # Cell height: 8 margin + 96 crop + 4 gap + 14 text + 6 margin = 128 px
    cell_w, cell_h = 168, 128

    for filename, indices in groups.items():
        if len(indices) == 0:
            continue
        n = len(indices)
        rows = (n + cols - 1) // cols
        sheet_w = cols * cell_w
        sheet_h = rows * cell_h

        img = Image.new("RGB", (sheet_w, sheet_h), color=(242, 242, 244))
        draw = ImageDraw.Draw(img)

        for i, idx in enumerate(indices):
            r, c = i // cols, i % cols
            x0, y0 = c * cell_w, r * cell_h

            # Draw cell background and subtle border
            draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 3, y0 + cell_h - 3], fill=(255, 255, 255), outline=(215, 215, 220))

            # Left: 8x8 tile enlarged 6x (48x48)
            tile_8x8 = images[idx]
            rgb_tile_8x8 = DMG_PALETTE_RGB[tile_8x8]
            rgb_tile_48x48 = np.repeat(np.repeat(rgb_tile_8x8, 6, axis=0), 6, axis=1)
            img.paste(Image.fromarray(rgb_tile_48x48), (x0 + 8, y0 + 8))
            draw.rectangle([x0 + 7, y0 + 7, x0 + 8 + 48, y0 + 8 + 48], outline=(180, 180, 185))

            # Printed key label under tile
            k_txt = str(keys[idx])
            cnt_txt = f"{counts[idx]:,}"
            draw.text((x0 + 8, y0 + 60), k_txt, fill=(20, 20, 20), font=font)
            draw.text((x0 + 8, y0 + 74), f"n={cnt_txt}", fill=(100, 100, 105), font=font)

            # Right: 24x24 context crop enlarged 4x (96x96)
            crop_24x24 = crops[idx]
            rgb_crop_24x24 = DMG_PALETTE_RGB[crop_24x24]
            rgb_crop_96x96 = np.repeat(np.repeat(rgb_crop_24x24, 4, axis=0), 4, axis=1)
            cx0 = x0 + 8 + 48 + 8
            img.paste(Image.fromarray(rgb_crop_96x96), (cx0, y0 + 8))
            draw.rectangle([cx0 - 1, y0 + 7, cx0 + 96, y0 + 8 + 96], outline=(180, 180, 185))

            # Red outline around target tile in center [8:16, 8:16] -> [32:64, 32:64]
            draw.rectangle([cx0 + 32, y0 + 8 + 32, cx0 + 64 - 1, y0 + 8 + 64 - 1], outline=(230, 30, 30), width=1)

        sheet_file = out_path / filename
        img.save(sheet_file)
        generated_sheets.append(str(sheet_file))
        print(f"Saved contact sheet: {sheet_file} ({sheet_file.stat().st_size / 1024:.1f} KB, {n} tiles)")

    return generated_sheets


def demo(
    out: str = "work/color/native/demo.png",
    palette_path: str = "work/color/native/palette.json",
) -> str:
    """Generate side-by-side demo PNG of 4 real frames: outdoor, forest, Pokémon Center, battle.

    Rendered as grey | colourised-with-the-skeleton (everything 'other') side by side, 3x nearest-neighbour.
    """
    from lab.env import PokemonRed

    out_file = Path(out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(palette_path, "r") as f:
        palette = json.load(f)

    # 4 target scenarios: outdoor, forest, Pokémon Center, battle
    scenarios = [
        ("Outdoor (Viridian City)", "work/route_states/to_grass.state", 0),
        ("Viridian Forest", "work/route_states/seed100_map51.state", 0),
        ("Pokémon Center", "work/route_states/leave_viridian_pc.state", 0),
    ]

    row_images: List[Image.Image] = []
    font = ImageFont.load_default()

    for label, state_file, _ in scenarios:
        with PokemonRed(state=state_file) as env:
            # Take a step or observe
            env.step("pass")
            pb = env.pyboy
            grey = pb.screen.ndarray[:, :, :3]
            px = pixel_tiles(pb)
            col = colorize(px, palette)

            # Grey | Colorized side-by-side
            pair = np.concatenate([grey, col], axis=1)  # (144, 320, 3)
            # 3x nearest-neighbour
            pair_3x = np.repeat(np.repeat(pair, 3, axis=0), 3, axis=1)  # (432, 960, 3)

            # Banner with label
            row_img = Image.new("RGB", (960, 432 + 28), color=(30, 32, 36))
            draw = ImageDraw.Draw(row_img)
            draw.text((16, 8), f"{label}   [ Left: Original Grey | Right: Colourised (Skeleton) ]", fill=(240, 240, 245), font=font)
            row_img.paste(Image.fromarray(pair_3x), (0, 28))
            row_images.append(row_img)

    # Reaching a battle frame
    print("Capturing battle demo frame...")
    with PokemonRed(state="work/route_states/to_grass.state") as env:
        rng = random.Random(42)
        battle_captured = False
        for s in range(350):
            if env.pyboy.memory[0xD057] != 0:
                pb = env.pyboy
                grey = pb.screen.ndarray[:, :, :3]
                px = pixel_tiles(pb)
                col = colorize(px, palette)
                pair = np.concatenate([grey, col], axis=1)
                pair_3x = np.repeat(np.repeat(pair, 3, axis=0), 3, axis=1)
                row_img = Image.new("RGB", (960, 432 + 28), color=(30, 32, 36))
                draw = ImageDraw.Draw(row_img)
                draw.text((16, 8), "Wild Battle   [ Left: Original Grey | Right: Colourised (Skeleton) ]", fill=(240, 240, 245), font=font)
                row_img.paste(Image.fromarray(pair_3x), (0, 28))
                row_images.append(row_img)
                battle_captured = True
                break
            action = "a" if (s % 10 == 0) else rng.choice(["down", "left", "right", "up"])
            env.step(action)

        if not battle_captured:
            # Fallback to an extra state if battle was not encountered within steps
            with PokemonRed(state="work/route_states/seed100_map12.state") as fallback_env:
                fallback_env.step("pass")
                pb = fallback_env.pyboy
                grey = pb.screen.ndarray[:, :, :3]
                px = pixel_tiles(pb)
                col = colorize(px, palette)
                pair = np.concatenate([grey, col], axis=1)
                pair_3x = np.repeat(np.repeat(pair, 3, axis=0), 3, axis=1)
                row_img = Image.new("RGB", (960, 432 + 28), color=(30, 32, 36))
                draw = ImageDraw.Draw(row_img)
                draw.text((16, 8), "Route 1   [ Left: Original Grey | Right: Colourised (Skeleton) ]", fill=(240, 240, 245), font=font)
                row_img.paste(Image.fromarray(pair_3x), (0, 28))
                row_images.append(row_img)

    # Combine all 4 rows vertically
    total_w = 960
    total_h = sum(r.height for r in row_images)
    demo_img = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    curr_y = 0
    for r in row_images:
        demo_img.paste(r, (0, curr_y))
        curr_y += r.height

    demo_img.save(out_file)
    print(f"Saved demo image to {out_file} ({out_file.stat().st_size / 1024:.1f} KB, dimensions: {total_w}x{total_h})")
    return str(out_file)


_orig_pixel_tiles = pixel_tiles


def _decode_gb_text(bytes_seq) -> str:
    res = []
    for b in bytes_seq:
        b_int = int(b)
        if b_int == 0x50:
            break
        elif 0x80 <= b_int <= 0x99:
            res.append(chr(b_int - 0x80 + ord('A')))
        elif 0xA0 <= b_int <= 0xB9:
            res.append(chr(b_int - 0xA0 + ord('a')))
        elif 0xF6 <= b_int <= 0xFF:
            res.append(chr(b_int - 0xF6 + ord('0')))
        elif b_int == 0x7F:
            res.append(" ")
        elif b_int == 0xEF:
            res.append("_M")
        elif b_int == 0xF5:
            res.append("_F")
    return "".join(res).strip().upper()


def pixel_tiles(pyboy) -> Dict[str, Any]:
    px = _orig_pixel_tiles(pyboy)
    try:
        mem = getattr(pyboy, "memory", None)
        if mem is not None and px.get("in_battle"):
            e_hp = (int(mem[0xCFE6]) << 8) | int(mem[0xCFE7])
            e_max = (int(mem[0xCFF4]) << 8) | int(mem[0xCFF5])
            e_frac = float(e_hp / e_max) if e_max > 0 else 1.0

            p_hp = (int(mem[0xD015]) << 8) | int(mem[0xD016])
            p_max = (int(mem[0xD023]) << 8) | int(mem[0xD024])
            if p_max == 0:
                p_hp = (int(mem[0xD16C]) << 8) | int(mem[0xD16D])
                p_max = (int(mem[0xD18D]) << 8) | int(mem[0xD18E])
            p_frac = float(p_hp / p_max) if p_max > 0 else 1.0

            px["enemy_hp_frac"] = max(0.0, min(1.0, e_frac))
            px["player_hp_frac"] = max(0.0, min(1.0, p_frac))

            px["enemy_name"] = _decode_gb_text([mem[0xCFDA + i] for i in range(11)])
            px["player_mon_name"] = _decode_gb_text([mem[0xD009 + i] for i in range(11)])

            px["enemy_types"] = (int(mem[0xCFEA]), int(mem[0xCFEB]))
            px["player_mon_types"] = (int(mem[0xD019]), int(mem[0xD01A]))

            p_m = int(mem[0xCFD2])
            e_m = int(mem[0xCFCC])
            if p_m > 0:
                px["move_type"] = int(mem[0xCFD5])
            elif e_m > 0:
                px["move_type"] = int(mem[0xCFCF])
            else:
                px["move_type"] = 0x00
        else:
            px["enemy_hp_frac"] = 1.0
            px["player_hp_frac"] = 1.0
            px["enemy_name"] = ""
            px["player_mon_name"] = ""
            px["enemy_types"] = (0x00, 0x00)
            px["player_mon_types"] = (0x00, 0x00)
            px["move_type"] = 0x00
    except Exception:
        px["enemy_hp_frac"] = 1.0
        px["player_hp_frac"] = 1.0
        px["enemy_name"] = ""
        px["player_mon_name"] = ""
        px["enemy_types"] = (0x00, 0x00)
        px["player_mon_types"] = (0x00, 0x00)
        px["move_type"] = 0x00
    t_arr = px.get("tile")
    px["player_mon_out"] = bool(np.any((t_arr[56:, :] >= 355) & (t_arr[56:, :] <= 363))) if t_arr is not None else True
    return px
