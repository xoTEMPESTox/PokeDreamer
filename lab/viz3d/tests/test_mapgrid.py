"""Tests for lab.viz3d.mapgrid (toy arrays, CPU only)."""
import numpy as np
import pytest

from lab.viz3d.mapgrid import build_map_grid


def test_build_map_grid_toy():
    # 2x3 grid of 8x8 tiles -> image of (16, 24)
    tile_ids = np.array([
        [0, 1, 140],
        [20, 25, 200]
    ], dtype=np.int16)

    # Synthetic VRAM data: 384 tiles * 16 bytes = 6144 bytes
    vram_bytes = bytearray(6144)
    # Set tile 256 (from tile 0) to all shade 1
    # 2BPP: plane 0 = 0xFF, plane 1 = 0x00 -> color 1
    for row in range(8):
        vram_bytes[256 * 16 + row * 2] = 0xFF
        vram_bytes[256 * 16 + row * 2 + 1] = 0x00

    # Set tile 276 (map tile id 20 = 0x14) to all shade 3
    # 2BPP: plane 0 = 0xFF, plane 1 = 0xFF -> color 3
    for row in range(8):
        vram_bytes[276 * 16 + row * 2] = 0xFF
        vram_bytes[276 * 16 + row * 2 + 1] = 0xFF

    labels = {
        "0:256": "grass",
        "0:257": "path",
        "0:276": "water",
        "0:140": "other",
        "0:200": "text_box",
    }

    t_ids, cidx_img, class_per_tile = build_map_grid(
        tile_ids, vram_bytes, tileset=0, labels=labels
    )

    assert t_ids.shape == (2, 3)
    assert cidx_img.shape == (16, 24)
    assert class_per_tile.shape == (2, 3)

    # Tile (0, 0) is tile id 0 -> vram 256 -> 'grass'
    assert class_per_tile[0, 0] == "grass"
    assert np.all(cidx_img[0:8, 0:8] == 1)

    # Tile (1, 0) is tile id 20 -> vram 276 -> 'water'
    assert class_per_tile[1, 0] == "water"
    assert np.all(cidx_img[8:16, 0:8] == 3)

    # Tile (0, 2) is tile id 140 (>= 128) -> vram 140 -> 'other'
    assert class_per_tile[0, 2] == "other"
