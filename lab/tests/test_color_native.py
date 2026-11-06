"""Unit tests on toy arrays for native per-tile colorization.

Tests pure scroll mapping, key formatting, colorize ramp and index mapping,
VRAM decoding, and palette initialization without requiring emulator runtime.
"""
from __future__ import annotations

import numpy as np
import pytest

from lab.color.native import (
    colorize,
    create_skeleton_palette,
    decode_tile_map,
    decode_vram_tiles,
    format_key,
    map_background_scroll,
    map_window,
    parse_key,
    rebuild_screen_shades,
)


def test_key_formatting():
    """Verify key formatting and parsing roundtrip."""
    k = format_key(0, 1, 42)
    assert k == "0:1:42"
    ts, l, t = parse_key(k)
    assert ts == 0
    assert l == 1
    assert t == 42

    # Negative / large tile numbers
    k2 = format_key(6, 2, 383)
    assert parse_key(k2) == (6, 2, 383)

    with pytest.raises(ValueError):
        parse_key("invalid_key")


def test_synthetic_2tile_background_scroll():
    """Verify that a synthetic 2-tile background with SCX=4 maps pixels to the right tiles."""
    # 2-tile background: Tile 10 at col 0 (pixels 0..7), Tile 20 at col 1 (pixels 8..15)
    bg_map = np.array([[10, 20]], dtype=np.int16)
    screen_shape = (1, 16)  # 1 row, 16 pixels wide

    tiles, ty, tx = map_background_scroll(scx=4, scy=0, bg_map=bg_map, screen_shape=screen_shape)

    # With SCX=4:
    # Screen x=0..3 map to bg_x=4..7 -> Tile 10, tx=4..7
    # Screen x=4..11 map to bg_x=8..15 -> Tile 20, tx=0..7
    # Screen x=12..15 map to bg_x=16..19 (wrapped to 0..3) -> Tile 10, tx=0..3
    expected_tiles = np.array([[10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10]], dtype=np.int16)
    expected_tx = np.array([[4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]], dtype=np.uint8)

    assert np.array_equal(tiles, expected_tiles), f"Got {tiles}, expected {expected_tiles}"
    assert np.array_equal(tx, expected_tx), f"Got {tx}, expected {expected_tx}"
    assert np.all(ty == 0)


def test_colorize_uses_class_ramp_and_colour_index():
    """Verify colorize maps each pixel through class ramp according to 2-bit colour index."""
    px = {
        "tileset": 0,
        "layer": np.array([[0, 0], [1, 2]], dtype=np.uint8),
        "tile": np.array([[10, 10], [20, 30]], dtype=np.int16),
        "cidx": np.array([[0, 3], [1, 2]], dtype=np.uint8),
    }

    palette = {
        "classes": {
            "grass": [[255, 255, 255], [100, 200, 100], [50, 150, 50], [0, 80, 0]],
            "water": [[200, 240, 255], [100, 180, 240], [40, 120, 200], [10, 50, 120]],
            "other": [[220, 220, 220], [160, 160, 160], [100, 100, 100], [40, 40, 40]],
        },
        "tiles": {
            "0:0:10": "grass",
            "0:1:20": "water",
            # "0:2:30" is omitted, so it falls back to "other"
        },
    }

    rgb = colorize(px, palette)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8

    # (0, 0): key 0:0:10 -> grass, cidx 0 -> [255, 255, 255]
    assert np.array_equal(rgb[0, 0], [255, 255, 255])

    # (0, 1): key 0:0:10 -> grass, cidx 3 -> [0, 80, 0]
    assert np.array_equal(rgb[0, 1], [0, 80, 0])

    # (1, 0): key 0:1:20 -> water, cidx 1 -> [100, 180, 240]
    assert np.array_equal(rgb[1, 0], [100, 180, 240])

    # (1, 1): key 0:2:30 -> fallback "other", cidx 2 -> [100, 100, 100]
    assert np.array_equal(rgb[1, 1], [100, 100, 100])


def test_decode_vram_tiles():
    """Verify 2BPP bitplane decoding for Game Boy tile graphics."""
    # Construct 1 tile with 16 bytes:
    # row 0: byte1 = 0b10100000, byte2 = 0b11000000 -> cidx = [3, 2, 1, 0, 0, 0, 0, 0]
    tile_bytes = np.zeros((384, 16), dtype=np.uint8)
    tile_bytes[0, 0] = 0b10100000
    tile_bytes[0, 1] = 0b11000000

    decoded = decode_vram_tiles(tile_bytes.flatten())
    assert decoded.shape == (384, 8, 8)
    expected_row0 = np.array([3, 2, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    assert np.array_equal(decoded[0, 0], expected_row0)


def test_decode_tile_map_signed_and_unsigned():
    """Verify unsigned (0x8000) and signed (0x8800) tilemap indexing."""
    raw = np.zeros(1024, dtype=np.uint8)
    raw[0] = 0
    raw[1] = 127
    raw[2] = 128
    raw[3] = 255

    # Unsigned (tiledata_select = True)
    unsigned_map = decode_tile_map(raw, tiledata_select=True)
    assert unsigned_map[0, 0] == 0
    assert unsigned_map[0, 1] == 127
    assert unsigned_map[0, 2] == 128
    assert unsigned_map[0, 3] == 255

    # Signed (tiledata_select = False):
    # 0..127 -> 256..383
    # 128..255 -> 128..255
    signed_map = decode_tile_map(raw, tiledata_select=False)
    assert signed_map[0, 0] == 256
    assert signed_map[0, 1] == 383
    assert signed_map[0, 2] == 128
    assert signed_map[0, 3] == 255


def test_skeleton_palette_structure():
    """Verify create_skeleton_palette produces valid structure and classes."""
    sample_keys = ["0:0:291", "0:2:8", "6:0:100"]
    pal = create_skeleton_palette(sample_keys)

    assert "classes" in pal
    assert "tiles" in pal
    assert len(pal["classes"]) >= 20

    for cls_name, ramp in pal["classes"].items():
        assert len(ramp) == 4
        for rgb in ramp:
            assert len(rgb) == 3
            assert all(0 <= c <= 255 for c in rgb)

    for k in sample_keys:
        assert pal["tiles"][k] == "other"
