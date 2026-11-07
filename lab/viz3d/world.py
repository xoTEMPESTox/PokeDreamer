"""CPU world builder: combined Viridian + Route 1 maps, ground textures, scatter masks, and world.npz."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from lab.viz3d.classes import load_palette
from lab.viz3d.mapgrid import full_map
from lab.viz3d.bpy_build import CLASS_NAMES, PALETTE_SRGB

# Scale: 1 Game Boy pixel = 1 cm = 0.01 Blender units (meters)
S = 0.01

FORBIDDEN_BLADE_CLASSES = {
    "path", "water", "building_wall", "roof_red", "roof_blue",
    "door", "sign", "fence", "tree"
}


def assemble_combined_maps(
    map1_state: str | Path = "work/route_states/seed100_map1.state",
    map12_state: str | Path = "work/route_states/seed100_map12.state",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble combined Viridian (map 1) + Route 1 (map 12) class and shade maps.

    Combined size: 1152 rows x 640 cols px (144 x 80 tiles of 8x8).
    Viridian is placed at (0, 0).
    Route 1 is placed at (row 576, col 160).
    The two 160-px-wide strips beside Route 1 (cols 0..160 and 480..640) are filled with 'grass'.
    """
    t1, c1, cl1 = full_map(map1_state)
    t12, c12, cl12 = full_map(map12_state)

    H1_px, W1_px = c1.shape  # (576, 640)
    H12_px, W12_px = c12.shape  # (576, 320)

    total_H_px = H1_px + H12_px  # 1152
    total_W_px = W1_px  # 640

    names, _ = load_palette()
    name_to_idx = {n: i for i, n in enumerate(names[:23])}

    # Convert tile string classes to per-pixel class index maps (int8)
    def map_to_pixel_class(class_per_tile):
        Ht, Wt = class_per_tile.shape
        cls_px = np.empty((Ht * 8, Wt * 8), dtype=np.int8)
        for r in range(Ht):
            for c in range(Wt):
                cname = class_per_tile[r, c]
                cidx = name_to_idx.get(cname, name_to_idx.get("other", 21))
                cls_px[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = cidx
        return cls_px

    cls1_px = map_to_pixel_class(cl1)
    cls12_px = map_to_pixel_class(cl12)

    combined_cls = np.empty((total_H_px, total_W_px), dtype=np.int8)
    combined_shade = np.empty((total_H_px, total_W_px), dtype=np.uint8)

    # 1. Place Viridian at top: rows 0..576, cols 0..640
    combined_cls[:H1_px, :W1_px] = cls1_px
    combined_shade[:H1_px, :W1_px] = c1

    # 2. Place Route 1 at (row 576, col 160)
    r1_row = H1_px
    r1_col = 160
    combined_cls[r1_row : r1_row + H12_px, r1_col : r1_col + W12_px] = cls12_px
    combined_shade[r1_row : r1_row + H12_px, r1_col : r1_col + W12_px] = c12

    # 3. Find pure-grass 16x16 tile pattern from Viridian
    grass_idx = name_to_idx["grass"]
    pure_grass_patch = None
    for r in range(0, H1_px - 15, 16):
        for c in range(0, W1_px - 15, 16):
            if np.all(cls1_px[r : r + 16, c : c + 16] == grass_idx):
                pure_grass_patch = c1[r : r + 16, c : c + 16]
                break
        if pure_grass_patch is not None:
            break

    if pure_grass_patch is None:
        # Fallback default shade pattern
        pure_grass_patch = np.zeros((16, 16), dtype=np.uint8)

    # Tile pure grass patch across left strip (cols 0..160) and right strip (cols 480..640)
    for r in range(r1_row, total_H_px, 16):
        for c in range(0, r1_col, 16):
            combined_cls[r : r + 16, c : c + 16] = grass_idx
            combined_shade[r : r + 16, c : c + 16] = pure_grass_patch
        for c in range(r1_col + W12_px, total_W_px, 16):
            combined_cls[r : r + 16, c : c + 16] = grass_idx
            combined_shade[r : r + 16, c : c + 16] = pure_grass_patch

    # Also build tile-level class strings array (144, 80)
    combined_tile_classes = np.empty((total_H_px // 8, total_W_px // 8), dtype=object)
    combined_tile_classes[: H1_px // 8, : W1_px // 8] = cl1
    combined_tile_classes[r1_row // 8 : (r1_row + H12_px) // 8, r1_col // 8 : (r1_col + W12_px) // 8] = cl12
    for r in range(r1_row // 8, total_H_px // 8):
        for c in range(0, r1_col // 8):
            combined_tile_classes[r, c] = "grass"
        for c in range((r1_col + W12_px) // 8, total_W_px // 8):
            combined_tile_classes[r, c] = "grass"

    # Tile-level ids array (144, 80)
    combined_tile_ids = np.zeros((total_H_px // 8, total_W_px // 8), dtype=np.int32)
    combined_tile_ids[: H1_px // 8, : W1_px // 8] = t1
    combined_tile_ids[r1_row // 8 : (r1_row + H12_px) // 8, r1_col // 8 : (r1_col + W12_px) // 8] = t12

    # 4. Reclassify 2x2 sign tiles ([70, 71] / [86, 87]) as 'sign'
    # and paint ground under each sign with surrounding ground (no flat sign art on ground).
    sign_idx = name_to_idx["sign"]
    H_t, W_t = combined_tile_ids.shape
    sign_art = None  # the game's own 16x16 sign pixels (shades), kept for the 3D boards before the ground is painted over
    for r in range(H_t - 1):
        for c in range(W_t - 1):
            if (combined_tile_ids[r, c] == 70 and combined_tile_ids[r, c + 1] == 71 and
                    combined_tile_ids[r + 1, c] == 86 and combined_tile_ids[r + 1, c + 1] == 87):
                if sign_art is None:
                    sign_art = combined_shade[r * 8 : (r + 2) * 8, c * 8 : (c + 2) * 8].copy()
                combined_tile_classes[r : r + 2, c : c + 2] = "sign"
                combined_cls[r * 8 : (r + 2) * 8, c * 8 : (c + 2) * 8] = sign_idx
                # Paint ground under sign with surrounding ground (adjacent 2x2 tiles to left)
                if c >= 2:
                    combined_shade[r * 8 : (r + 2) * 8, c * 8 : (c + 2) * 8] = combined_shade[r * 8 : (r + 2) * 8, (c - 2) * 8 : c * 8]
                elif c + 2 < W_t:
                    combined_shade[r * 8 : (r + 2) * 8, c * 8 : (c + 2) * 8] = combined_shade[r * 8 : (r + 2) * 8, (c + 2) * 8 : (c + 4) * 8]

    return combined_cls, combined_shade, combined_tile_classes, combined_tile_ids, sign_art


def compute_scatter_mask(cls_px: np.ndarray) -> np.ndarray:
    """Compute RGBA float scatter mask (1152x640):

    R = grass density factor (1.0 on grass, 0.35 on 'other', 0 elsewhere)
    G = tall_grass density factor (1.0 on tall_grass, 0 elsewhere)
    B = flowers density factor (1.0 on flowers, 0 elsewhere)
    A = 1.0 where blades may grow at all (not path/water/wall/roof/door/sign/fence/tree)
    """
    H, W = cls_px.shape
    mask = np.zeros((H, W, 4), dtype=np.float32)

    names, _ = load_palette()
    name_to_idx = {n: i for i, n in enumerate(names[:23])}

    grass_idx = name_to_idx.get("grass", 0)
    other_idx = name_to_idx.get("other", 21)
    tg_idx = name_to_idx.get("tall_grass", 1)
    flower_idx = name_to_idx.get("flowers", 7)

    forbidden_indices = {name_to_idx[n] for n in FORBIDDEN_BLADE_CLASSES if n in name_to_idx}

    mask[cls_px == grass_idx, 0] = 1.0
    mask[cls_px == other_idx, 0] = 0.35
    mask[cls_px == tg_idx, 1] = 1.0
    mask[cls_px == flower_idx, 2] = 1.0

    # Alpha: 1.0 where blades may grow at all (not forbidden)
    is_forbidden = np.isin(cls_px, list(forbidden_indices))
    mask[~is_forbidden, 3] = 1.0

    return mask


def create_ground_images(
    cls_px: np.ndarray,
    shade_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ground_crisp.png (1152x640 grey), ground_raw.png (576x320 grey 2x mean),

    and ground_color.png (1152x640 RGB colour via PALETTE_SRGB).
    """
    H, W = shade_px.shape
    grey_lut = np.array([255, 170, 85, 0], dtype=np.uint8)

    # 1. Crisp Game Boy grey (1152x640)
    ground_crisp = grey_lut[shade_px]

    # 2. Raw Game Boy grey (576x320: exact 2x2 mean of crisp image)
    raw_f = (
        ground_crisp[0::2, 0::2].astype(np.float32)
        + ground_crisp[0::2, 1::2].astype(np.float32)
        + ground_crisp[1::2, 0::2].astype(np.float32)
        + ground_crisp[1::2, 1::2].astype(np.float32)
    ) / 4.0
    ground_raw = np.round(raw_f).astype(np.uint8)

    # 3. Ground color RGB (1152x640)
    ground_color = np.zeros((H, W, 3), dtype=np.uint8)
    names, _ = load_palette()
    path_ramp = PALETTE_SRGB.get("path", PALETTE_SRGB["other"])
    for cidx, cname in enumerate(names[:23]):
        c_mask = (cls_px == cidx)
        if not np.any(c_mask):
            continue
        # For ground under sign, paint with path color so no flat sign art appears
        ramp = path_ramp if cname == "sign" else PALETTE_SRGB.get(cname, PALETTE_SRGB["other"])
        for sh in range(4):
            sh_mask = c_mask & (shade_px == sh)
            if np.any(sh_mask):
                ground_color[sh_mask] = ramp[sh]

    return ground_crisp, ground_raw, ground_color


def extract_world_elements(
    cls_px: np.ndarray,
    tile_classes: np.ndarray,
) -> dict:
    """Extract tree blocks, fences, real buildings, and sign posts from combined maps."""
    H_px, W_px = cls_px.shape
    H_tiles, W_tiles = H_px // 8, W_px // 8

    names, _ = load_palette()
    name_to_idx = {n: i for i, n in enumerate(names[:23])}
    tree_idx = name_to_idx["tree"]
    fence_idx = name_to_idx["fence"]

    # 1. Tree block instances (16x16 player tile blocks)
    tree_list = []
    np.random.seed(42)
    for r in range(0, H_px - 15, 16):
        for c in range(0, W_px - 15, 16):
            patch = cls_px[r : r + 16, c : c + 16]
            if np.mean(patch == tree_idx) > 0.55:
                cx = (c + 8) * S
                cy = -(r + 8) * S
                phase = (cx * 7.91 + cy * 11.37) % (2.0 * np.pi)
                # Seeded scale 0.9-1.1 and yaw 0-2pi
                pos_hash = int((c * 1000 + r) * 2654435761) & 0xFFFFFFF
                rng = np.random.RandomState(pos_hash)
                scale = float(rng.uniform(0.9, 1.1))
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
                tree_list.append([cx, cy, 0.0, phase, scale, yaw])
    tree_blocks = np.array(tree_list, dtype=np.float32) if tree_list else np.empty((0, 6), dtype=np.float32)

    # 2. Fence bollards (16x16 tile blocks)
    fence_list = []
    for r in range(0, H_px - 15, 16):
        for c in range(0, W_px - 15, 16):
            patch = cls_px[r : r + 16, c : c + 16]
            if np.mean(patch == fence_idx) > 0.35:
                cx = (c + 8) * S
                cy = -(r + 8) * S
                fence_list.append([cx, cy])
    fence_tiles = np.array(fence_list, dtype=np.float32) if fence_list else np.empty((0, 2), dtype=np.float32)

    # 3. Building components on 8x8 tile grid
    building_names = {"building_wall", "door", "roof_red", "roof_blue"}
    is_building = np.zeros((H_tiles, W_tiles), dtype=bool)
    is_wall = np.zeros((H_tiles, W_tiles), dtype=bool)
    is_roof = np.zeros((H_tiles, W_tiles), dtype=bool)

    for r in range(H_tiles):
        for c in range(W_tiles):
            cname = tile_classes[r, c]
            if cname in building_names:
                is_building[r, c] = True
            if cname in {"building_wall", "door"}:
                is_wall[r, c] = True
            if cname in {"roof_red", "roof_blue"}:
                is_roof[r, c] = True

    # 8-connected components on building tiles
    num_labels, labels = cv2.connectedComponents(is_building.astype(np.uint8), connectivity=8)

    sign_tiles_list = []
    real_wall_tiles = []
    real_roof_tiles = []

    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id)
        comp_size = int(np.sum(comp_mask))
        if comp_size < 4:
            # Ledge post / fence end tile (preserved as low wooden posts)
            coords = np.argwhere(comp_mask)
            for r, c in coords:
                cx = (c * 8 + 4) * S
                cy = -(r * 8 + 4) * S
                sign_tiles_list.append([cx, cy, float(r), float(c)])
        else:
            # Real building
            wall_coords = np.argwhere(comp_mask & is_wall)
            for r, c in wall_coords:
                real_wall_tiles.append([r, c])
            roof_coords = np.argwhere(comp_mask & is_roof)
            for r, c in roof_coords:
                real_roof_tiles.append([r, c])

    sign_tiles = np.array(sign_tiles_list, dtype=np.float32) if sign_tiles_list else np.empty((0, 4), dtype=np.float32)
    wall_tiles = np.array(real_wall_tiles, dtype=np.int32) if real_wall_tiles else np.empty((0, 2), dtype=np.int32)
    roof_tiles = np.array(real_roof_tiles, dtype=np.int32) if real_roof_tiles else np.empty((0, 2), dtype=np.int32)

    # 4. Standing 3D sign boards (2x2 tile blocks classified as 'sign')
    sign_boards_list = []
    for r in range(H_tiles - 1):
        for c in range(W_tiles - 1):
            if (tile_classes[r, c] == "sign" and tile_classes[r, c + 1] == "sign" and
                    tile_classes[r + 1, c] == "sign" and tile_classes[r + 1, c + 1] == "sign"):
                cx = (c * 8 + 8) * S
                cy = -(r * 8 + 8) * S
                sign_boards_list.append([cx, cy, float(r), float(c)])
    sign_boards = np.array(sign_boards_list, dtype=np.float32) if sign_boards_list else np.empty((0, 4), dtype=np.float32)

    # 5. Water tiles
    water_coords = np.argwhere(tile_classes == "water")
    water_tiles = np.array(water_coords, dtype=np.int32) if len(water_coords) else np.empty((0, 2), dtype=np.int32)

    return {
        "tree_blocks": tree_blocks,
        "fence_tiles": fence_tiles,
        "sign_tiles": sign_tiles,
        "sign_boards": sign_boards,
        "wall_tiles": wall_tiles,
        "roof_tiles": roof_tiles,
        "water_tiles": water_tiles,
        "pond_bed_depth": np.float32(-0.030),
    }


def build_world(
    out_dir: str | Path = "work/viz3d/v2/world",
    map1_state: str | Path = "work/route_states/seed100_map1.state",
    map12_state: str | Path = "work/route_states/seed100_map12.state",
) -> Path:
    """Build combined Viridian + Route 1 world assets and save to out_dir."""
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cls_px, shade_px, tile_classes, tile_ids, sign_art = assemble_combined_maps(map1_state, map12_state)

    # 1. Scatter mask
    scatter_mask = compute_scatter_mask(cls_px)
    mask_16 = np.clip(scatter_mask * 65535.0, 0, 65535).astype(np.uint16)
    # OpenCV expects BGRA order
    mask_bgra = mask_16[..., [2, 1, 0, 3]]
    cv2.imwrite(str(out_path / "scatter_mask.png"), mask_bgra)

    # 2. Ground textures
    crisp, raw, color = create_ground_images(cls_px, shade_px)
    cv2.imwrite(str(out_path / "ground_crisp.png"), crisp)
    cv2.imwrite(str(out_path / "ground_raw.png"), raw)
    cv2.imwrite(str(out_path / "ground_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    # 3. Element lists
    elements = extract_world_elements(cls_px, tile_classes)

    # 4. Save world.npz
    npz_path = out_path / "world.npz"
    np.savez_compressed(
        npz_path,
        cls=cls_px,
        shade=shade_px,
        origin_route1=np.array([160, 576], dtype=np.int32),
        size_px=np.array([640, 1152], dtype=np.int32),
        tree_blocks=elements["tree_blocks"],
        fence_tiles=elements["fence_tiles"],
        sign_tiles=elements["sign_tiles"],
        sign_boards=elements["sign_boards"],
        wall_tiles=elements["wall_tiles"],
        roof_tiles=elements["roof_tiles"],
        water_tiles=elements["water_tiles"],
        pond_bed_depth=elements["pond_bed_depth"],
    )

    # 5. Save sign texture image (16x16 RGB): the game's own sign pixels, coloured with the sign ramp
    sign_ramp = PALETTE_SRGB.get("sign", PALETTE_SRGB["other"])
    sign_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    for sh in range(4):
        sign_rgb[sign_art == sh] = sign_ramp[sh]
    cv2.imwrite(str(out_path / "sign_texture.png"), cv2.cvtColor(sign_rgb, cv2.COLOR_RGB2BGR))

    print(f"Built world assets in {out_path}:")
    print(f"  world.npz (size: {cls_px.shape})")
    print(f"  tree_blocks: {len(elements['tree_blocks'])} instances")
    print(f"  fence_tiles: {len(elements['fence_tiles'])} instances")
    print(f"  sign_tiles: {len(elements['sign_tiles'])} instances")
    print(f"  sign_boards: {len(elements['sign_boards'])} instances")
    print(f"  water_tiles: {len(elements['water_tiles'])} tiles")
    return npz_path


if __name__ == "__main__":
    build_world()
