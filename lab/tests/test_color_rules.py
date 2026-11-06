"""Class rules on top of lab.color.native: layer-agnostic tile labels, the shared font, the player box, battles."""
import numpy as np

from lab.color.rules import PLAYER_BOX, build_tile_map, colorize_frame

CLASSES = {name: [[i * 10 + 1, i * 10 + 2, i * 10 + 3]] * 4 for i, name in enumerate(
    ["other", "grass", "text_box", "sprite_npc", "sprite_player", "battle_bg"])}


def test_labels_apply_to_both_bg_and_window_and_font_is_text():
    keys = ["0:0:300", "0:1:300", "0:1:140", "6:0:140", "0:2:12", "0:0:999"]
    tiles = build_tile_map({"0:300": "grass"}, keys)
    assert tiles["0:0:300"] == tiles["0:1:300"] == "grass"          # same tile id, same graphic, either layer
    assert tiles["0:1:140"] == tiles["6:0:140"] == "text_box"       # font region 128-255 in every tileset
    assert tiles["0:2:12"] == "sprite_npc"
    assert tiles["0:0:999"] == "other"


def _px(layer, tile, in_battle=False):
    return {"layer": layer, "tile": tile, "cidx": np.zeros_like(tile), "tileset": 0, "in_battle": in_battle}


def test_player_box_and_battle_override():
    palette = {"classes": CLASSES, "tiles": build_tile_map({"0:300": "grass"}, ["0:0:300", "0:2:5"])}
    layer = np.zeros((144, 160), np.int16); tile = np.full((144, 160), 300, np.int16)
    (y0, y1), (x0, x1) = PLAYER_BOX
    layer[y0:y1, x0:x1] = 2; tile[y0:y1, x0:x1] = 5          # the player at the screen centre
    layer[0:8, 0:8] = 2; tile[0:8, 0:8] = 5                    # an NPC sprite elsewhere
    out = colorize_frame(_px(layer, tile), palette)
    assert tuple(out[y0, x0]) == tuple(CLASSES["sprite_player"][0])
    assert tuple(out[0, 0]) == tuple(CLASSES["sprite_npc"][0])
    assert tuple(out[100, 100]) == tuple(CLASSES["grass"][0])
    tile[100, 100] = 140
    out = colorize_frame(_px(layer, tile, in_battle=True), palette)
    assert tuple(out[50, 150]) == tuple(CLASSES["battle_bg"][0])    # overworld labels don't apply in battle
    assert tuple(out[100, 100]) == tuple(CLASSES["text_box"][0])
