"""Unit tests for learned colorizer (toy arrays, no GPU, no emulator).

Tests:
1. class_map and ramp consistency with rules.colorize_frame on synthetic px dicts.
2. Colorizer model architecture: output shape and parameter budget (<= 2M).
3. colorize72 output shape and dtype on batch and single frames.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from lab.color.learned import Colorizer, class_map, colorize72, prepare_input
from lab.color.rules import PLAYER_BOX, colorize_frame

CLASSES = {
    name: [[i * 10 + 1, i * 10 + 2, i * 10 + 3]] * 4
    for i, name in enumerate([
        "other", "grass", "text_box", "sprite_npc", "sprite_player", "battle_bg",
        "tall_grass", "tree", "path", "water", "ledge", "fence", "flowers",
        "building_wall", "roof_red", "roof_blue", "door", "sign", "floor_indoor",
        "counter", "carpet", "battle_hp_bar", "rock",
    ])
}


def _synthetic_px(layer, tile, cidx=None, tileset=0, in_battle=False):
    if cidx is None:
        cidx = np.zeros_like(tile, dtype=np.uint8)
    return {
        "layer": np.asarray(layer, dtype=np.uint8),
        "tile": np.asarray(tile, dtype=np.int16),
        "cidx": np.asarray(cidx, dtype=np.uint8),
        "tileset": tileset,
        "in_battle": in_battle,
    }


def test_class_map_consistency_with_rules_colorize():
    """Verify that ramps[class_map(px, pal), px['cidx']] == colorize_frame(px, pal)."""
    palette = {
        "classes": CLASSES,
        "tiles": {
            "0:0:300": "grass",
            "0:1:300": "grass",
            "0:0:200": "tree",
            "0:2:5": "sprite_npc",
        },
    }
    names = list(CLASSES)
    ramps = np.asarray([CLASSES[n] for n in names], dtype=np.uint8)

    # 1. Overworld frame with player in center and NPC sprite in corner
    H, W = 144, 160
    layer = np.zeros((H, W), dtype=np.uint8)
    tile = np.full((H, W), 300, dtype=np.int16)
    cidx = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)

    (y0, y1), (x0, x1) = PLAYER_BOX
    layer[y0:y1, x0:x1] = 2
    tile[y0:y1, x0:x1] = 5

    # NPC sprite in corner
    layer[0:8, 0:8] = 2
    tile[0:8, 0:8] = 5

    # Font / text box tile
    layer[120:136, 10:50] = 1
    tile[120:136, 10:50] = 140

    px = _synthetic_px(layer, tile, cidx=cidx, tileset=0, in_battle=False)

    cm = class_map(px, palette)
    assert cm.shape == (H, W)
    assert np.issubdtype(cm.dtype, np.integer)

    # Player box check
    assert cm[y0, x0] == names.index("sprite_player")
    # NPC sprite check
    assert cm[0, 0] == names.index("sprite_npc")
    # Grass check
    assert cm[100, 100] == names.index("grass")
    # Text box font check
    assert cm[125, 20] == names.index("text_box")

    # Exact equality with rules.colorize_frame
    col_expected = colorize_frame(px, palette)
    col_actual = ramps[cm, px["cidx"]]
    np.testing.assert_array_equal(col_actual, col_expected)


def test_class_map_battle_mode():
    """Verify in battle overworld tiles map to battle_bg, font still text_box."""
    palette = {
        "classes": CLASSES,
        "tiles": {
            "0:0:300": "grass",
        },
    }
    names = list(CLASSES)
    ramps = np.asarray([CLASSES[n] for n in names], dtype=np.uint8)

    H, W = 144, 160
    layer = np.zeros((H, W), dtype=np.uint8)
    tile = np.full((H, W), 300, dtype=np.int16)
    cidx = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)

    # Font tile
    tile[100:110, 100:110] = 140

    px = _synthetic_px(layer, tile, cidx=cidx, tileset=0, in_battle=True)
    cm = class_map(px, palette)

    # Overworld label (grass) overridden to battle_bg in battle
    assert cm[10, 10] == names.index("battle_bg")
    # Font tile remains text_box in battle
    assert cm[105, 105] == names.index("text_box")

    col_expected = colorize_frame(px, palette)
    col_actual = ramps[cm, px["cidx"]]
    np.testing.assert_array_equal(col_actual, col_expected)


def test_class_map_default_palette():
    """Verify class_map(px) with default palette works and matches colorize_frame."""
    from lab.color.learned import load_palette
    palette = load_palette()
    names = list(palette["classes"])
    ramps = np.asarray([palette["classes"][n] for n in names], dtype=np.uint8)

    H, W = 144, 160
    layer = np.zeros((H, W), dtype=np.uint8)
    tile = np.full((H, W), 300, dtype=np.int16)
    cidx = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)

    (y0, y1), (x0, x1) = PLAYER_BOX
    layer[y0:y1, x0:x1] = 2
    tile[y0:y1, x0:x1] = 5

    px = _synthetic_px(layer, tile, cidx=cidx, tileset=0, in_battle=False)
    cm = class_map(px)  # palette=None uses default palette
    assert cm.shape == (H, W)

    col_expected = colorize_frame(px, palette)
    col_actual = ramps[cm, px["cidx"]]
    np.testing.assert_array_equal(col_actual, col_expected)


def test_model_output_shape_and_parameters():
    """Verify Colorizer parameter count is <= 2M and forward pass produces (B, 23, 144, 160)."""
    model = Colorizer()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 2_000_000, f"Model has {n_params} parameters, exceeding 2.0M budget!"

    x = torch.randn(2, 5, 144, 160)
    out = model(x)
    assert out.shape == (2, 23, 144, 160)
    assert out.dtype == torch.float32


class ToyUpsampler(nn.Module):
    """Toy upsampler for testing colorize72 without requiring trained weights."""
    def forward(self, x):
        pass


def test_colorize72_shape_and_dtype():
    """Verify colorize72 returns (N, 144, 160, 3) uint8 arrays for batches and single frames."""
    palette = {"classes": CLASSES, "tiles": {}}
    model = Colorizer()
    toy_up = ToyUpsampler()

    # Monkey-patch upsample to return deterministic 4-shade frames
    def fake_upsample(m, frames, device="cpu", batch=64):
        f = np.asarray(frames)
        single = (f.ndim == 2)
        if single:
            f = f[None]
        N = len(f)
        # Return 4 shades {0, 85, 153, 255}
        shades = np.full((N, 144, 160), 255, dtype=np.uint8)
        shades[:, :72, :80] = 153
        shades[:, 72:, :80] = 85
        shades[:, 72:, 80:] = 0
        return shades[0] if single else shades

    import lab.color.learned as learned_mod
    orig_upsample = learned_mod.upsample
    try:
        learned_mod.upsample = fake_upsample

        # Batch of 2 frames
        batch_72 = np.random.randint(0, 256, size=(2, 72, 80), dtype=np.uint8)
        res_batch = colorize72(batch_72, model, toy_up, palette, device="cpu")
        assert res_batch.shape == (2, 144, 160, 3)
        assert res_batch.dtype == np.uint8

        # Single frame
        single_72 = np.random.randint(0, 256, size=(72, 80), dtype=np.uint8)
        res_single = colorize72(single_72, model, toy_up, palette, device="cpu")
        assert res_single.shape == (144, 160, 3)
        assert res_single.dtype == np.uint8
    finally:
        learned_mod.upsample = orig_upsample
