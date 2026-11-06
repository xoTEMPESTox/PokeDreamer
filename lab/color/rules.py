"""Colour classes for real frames, on top of lab.color.native.pixel_tiles (display only; no metric uses it).

Measured on the collected inventory (work/color/native/tiles.npz, 2026-11-06):
- The same tile id shows the same graphic on the background and the window layer within a tileset, and the
  window layer also carries map tiles (gym floor, forest), so labels are keyed by "tileset:tile".
- Text (letters, digits) lives in tile ids 128-255, the font region shared by every tileset.
- The player is OAM slots 0-3, always drawn at the screen centre; its tile ids change with facing.
- In battle the game reloads tile memory with battle graphics, so overworld labels must not apply.
"""
import numpy as np

FONT = (128, 256)
PLAYER_BOX = ((60, 76), (64, 80))  # screen rows, cols of the player's 16x16 sprite (PyBoy sprite x 64/72, y 60/68,
                                   # already in screen coordinates; checked on seed100_map51.state)


def _rule(layer, tile):
    if layer == 2:
        return "sprite_npc"
    if FONT[0] <= tile < FONT[1]:
        return "text_box"
    return None


def build_tile_map(labels, keys):
    """labels {"tileset:tile": class} and inventory keys "tileset:layer:tile" -> palette["tiles"]."""
    out = {}
    for k in keys:
        ts, layer, tile = (int(v) for v in k.split(":"))
        out[k] = _rule(layer, tile) or labels.get(f"{ts}:{tile}", "other")
    return out


def colorize_frame(px, palette):
    """pixel_tiles dict -> (144, 160, 3) uint8: each pixel takes palette["classes"][class][cidx]."""
    classes, tiles = palette["classes"], palette["tiles"]
    names = list(classes)
    ramps = np.asarray([classes[n] for n in names], np.uint8)           # (classes, 4, 3)
    layer, tile, ts = px["layer"], px["tile"], px["tileset"]
    battle = bool(px.get("in_battle"))
    pairs, inv = np.unique(np.stack([layer.ravel(), tile.ravel()], 1), axis=0, return_inverse=True)
    lut = []
    for l, t in pairs:
        name = (_rule(l, t) or "battle_bg") if battle else (tiles.get(f"{ts}:{l}:{t}") or _rule(l, t) or "other")
        lut.append(names.index(name if name in classes else "other"))
    cls = np.asarray(lut)[inv.ravel()].reshape(layer.shape)
    if not battle:
        (y0, y1), (x0, x1) = PLAYER_BOX
        box = np.zeros(layer.shape, bool)
        box[y0:y1, x0:x1] = True
        cls[box & (layer == 2)] = names.index("sprite_player")
    return ramps[cls, px["cidx"]]
