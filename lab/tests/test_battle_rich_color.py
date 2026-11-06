"""Unit tests for richer battle colorization rules and learned class map parity.

Tests:
1. Region assignments: enemy sprite, player sprite, HP boxes, text box, background.
2. HP fill thresholds: >50% green (battle_hp_high), 20-50% amber (battle_hp_mid), <20% red (battle_hp_low).
3. Species ramps and type fallback logic.
4. Attack effect sprites by move type.
5. Exact parity between rules.colorize_frame and learned.class_map in battle mode.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from lab.color.learned import class_map, load_palette
from lab.color.rules import assign_battle_classes, colorize_frame


def _make_battle_px(
    layer=None,
    tile=None,
    cidx=None,
    enemy_hp_frac=1.0,
    player_hp_frac=1.0,
    enemy_name="RATTATA",
    player_mon_name="CHARMANDER",
    enemy_types=(0x00, 0x00),
    player_mon_types=(0x14, 0x14),
    move_type=None,
):
    H, W = 144, 160
    if layer is None:
        layer = np.zeros((H, W), dtype=np.uint8)
    if tile is None:
        tile = np.full((H, W), 383, dtype=np.int16)  # battle_bg tile
    if cidx is None:
        cidx = np.zeros((H, W), dtype=np.uint8)

    # Include player HP bar tile by default (mid-battle state)
    tile_arr = np.asarray(tile, dtype=np.int16)
    tile_arr[70, 100] = 356
    return {
        "layer": np.asarray(layer, dtype=np.uint8),
        "tile": tile_arr,
        "cidx": np.asarray(cidx, dtype=np.uint8),
        "tileset": 0,
        "in_battle": True,
        "enemy_hp_frac": enemy_hp_frac,
        "player_hp_frac": player_hp_frac,
        "enemy_name": enemy_name,
        "player_mon_name": player_mon_name,
        "enemy_types": enemy_types,
        "player_mon_types": player_mon_types,
        "move_type": move_type,
    }


def test_hp_bar_thresholds():
    """Verify HP bar fill tiles get battle_hp_high, battle_hp_mid, battle_hp_low."""
    palette = load_palette()
    names = list(palette["classes"])

    # High HP (>50%)
    px_high = _make_battle_px(enemy_hp_frac=0.75, player_hp_frac=0.55)
    # Put enemy HP bar tile at (20, 40) and player HP bar tile at (75, 100)
    px_high["tile"][20, 40] = 360  # HP bar fill
    px_high["tile"][75, 100] = 360
    cm_high = class_map(px_high, palette)
    assert names[cm_high[20, 40]] == "battle_hp_high"
    assert names[cm_high[75, 100]] == "battle_hp_high"

    # Mid HP (20% - 50%)
    px_mid = _make_battle_px(enemy_hp_frac=0.35, player_hp_frac=0.20)
    px_mid["tile"][20, 40] = 360
    px_mid["tile"][75, 100] = 360
    cm_mid = class_map(px_mid, palette)
    assert names[cm_mid[20, 40]] == "battle_hp_mid"
    assert names[cm_mid[75, 100]] == "battle_hp_mid"

    # Low HP (<20%)
    px_low = _make_battle_px(enemy_hp_frac=0.10, player_hp_frac=0.05)
    px_low["tile"][20, 40] = 360
    px_low["tile"][75, 100] = 360
    cm_low = class_map(px_low, palette)
    assert names[cm_low[20, 40]] == "battle_hp_low"
    assert names[cm_low[75, 100]] == "battle_hp_low"

    # Absent fraction defaults to high (green)
    px_none = _make_battle_px(enemy_hp_frac=None, player_hp_frac=None)
    px_none["tile"][20, 40] = 360
    cm_none = class_map(px_none, palette)
    assert names[cm_none[20, 40]] == "battle_hp_high"


def test_battle_region_assignments():
    """Verify enemy sprite, player sprite, box borders, text box, and bg regions."""
    palette = load_palette()
    names = list(palette["classes"])

    px = _make_battle_px(enemy_name="RATTATA", player_mon_name="CHARMANDER")

    # Enemy sprite region (e.g. row 20, col 110, tile 270)
    px["tile"][20, 110] = 270
    # Player mon sprite region (e.g. row 60, col 30, tile 320)
    px["tile"][60, 30] = 320
    # HP box border (tile 370)
    px["tile"][10, 10] = 370
    # Text box font tile (tile 150)
    px["tile"][120, 50] = 150
    # Background (tile 383)
    px["tile"][5, 5] = 383

    cm = class_map(px, palette)
    assert names[cm[20, 110]] == "mon_rattata"
    assert names[cm[60, 30]] == "mon_charmander"
    assert names[cm[10, 10]] == "battle_box"
    assert names[cm[120, 50]] == "text_box"
    assert names[cm[5, 5]] == "battle_bg"


def test_battle_intro_trainer_sprite():
    """Verify trainer back sprite uses battle_trainer class when player mon is not yet out."""
    palette = load_palette()
    names = list(palette["classes"])

    px = _make_battle_px(enemy_name="PIDGEY", player_mon_name="CHARMANDER")
    # Remove player HP bar so it is an intro frame
    px["tile"][70, 100] = 383
    px["tile"][60, 30] = 320  # Trainer sprite tile in player region

    cm = class_map(px, palette)
    assert names[cm[60, 30]] == "battle_trainer"



def test_species_ramp_and_type_fallback():
    """Verify unknown species falls back to primary type ramp or battle_enemy/mon."""
    palette = load_palette()
    names = list(palette["classes"])

    # Unknown species, but Water type (0x15)
    px_water = _make_battle_px(
        enemy_name="UNKNOWNMON",
        enemy_types=(0x15, 0x15),
    )
    px_water["tile"][20, 110] = 270
    cm_water = class_map(px_water, palette)
    assert names[cm_water[20, 110]] == "mon_type_water"

    # Unknown species, unknown type -> battle_enemy fallback
    px_unk = _make_battle_px(
        enemy_name="XYZ",
        enemy_types=(0x99, 0x99),
    )
    px_unk["tile"][20, 110] = 270
    cm_unk = class_map(px_unk, palette)
    assert names[cm_unk[20, 110]] == "battle_enemy"


def test_attack_effect_sprite():
    """Verify OAM sprites (layer == 2) during battle are coloured by move type."""
    palette = load_palette()
    names = list(palette["classes"])

    # Fire move (0x14)
    px_fire = _make_battle_px(move_type=0x14)
    px_fire["layer"][40, 50] = 2  # OAM effect sprite
    cm_fire = class_map(px_fire, palette)
    assert names[cm_fire[40, 50]] == "effect_fire"

    # Normal move (0x00)
    px_norm = _make_battle_px(move_type=0x00)
    px_norm["layer"][40, 50] = 2
    cm_norm = class_map(px_norm, palette)
    assert names[cm_norm[40, 50]] == "effect_normal"

    # Water move (0x15)
    px_water = _make_battle_px(move_type=0x15)
    px_water["layer"][40, 50] = 2
    cm_water = class_map(px_water, palette)
    assert names[cm_water[40, 50]] == "effect_water"


def test_class_map_parity_with_colorize_frame_in_battle():
    """Verify ramps[class_map(px, pal), px['cidx']] == colorize_frame(px, pal) in battle."""
    palette = load_palette()
    names = list(palette["classes"])
    ramps = np.asarray([palette["classes"][n] for n in names], dtype=np.uint8)

    H, W = 144, 160
    layer = np.zeros((H, W), dtype=np.uint8)
    tile = np.full((H, W), 383, dtype=np.int16)
    cidx = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)

    # Put enemy sprite, player sprite, HP bars, box borders, font, and effect sprite
    tile[10:30, 100:130] = 270   # enemy sprite
    tile[50:70, 20:50] = 320     # player mon sprite
    tile[20, 40:46] = 360        # enemy HP bar
    tile[75, 100:106] = 360      # player HP bar
    tile[15:18, 30:50] = 370     # HP box
    tile[120:135, 10:140] = 150  # font
    layer[45:55, 60:70] = 2      # effect sprite

    px = _make_battle_px(
        layer=layer,
        tile=tile,
        cidx=cidx,
        enemy_hp_frac=0.4,
        player_hp_frac=0.8,
        enemy_name="PIDGEY",
        player_mon_name="BULBASAUR",
        move_type=0x14,
    )

    cm = class_map(px, palette)
    col_cf = colorize_frame(px, palette)
    col_cm = ramps[cm, px["cidx"]]
    np.testing.assert_array_equal(col_cm, col_cf)
