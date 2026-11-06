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
    else:
        cls = assign_battle_classes(cls, px, names)
    return ramps[cls, px["cidx"]]


TYPE_ID_TO_NAME = {
    0x00: "normal",
    0x01: "fighting",
    0x02: "flying",
    0x03: "poison",
    0x04: "ground",
    0x05: "rock",
    0x07: "bug",
    0x08: "ghost",
    0x14: "fire",
    0x15: "water",
    0x16: "grass",
    0x17: "electric",
    0x18: "psychic",
    0x19: "ice",
    0x1A: "dragon",
}


def assign_battle_classes(cls, px, names):
    """Assigns rich battle classes: per-species colours, attack effects, and HP bars."""
    if isinstance(names, dict) and "classes" in names:
        names_list = list(names["classes"])
    elif isinstance(names, dict):
        names_list = list(names)
    else:
        names_list = list(names)

    def _idx(name, fallback="other"):
        if name in names_list:
            return names_list.index(name)
        if fallback in names_list:
            return names_list.index(fallback)
        return 0

    tile = px["tile"]
    layer = px.get("layer")
    H, W = tile.shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 1. HP Bar colours
    e_frac = px.get("enemy_hp_frac")
    p_frac = px.get("player_hp_frac")
    if e_frac is None:
        e_frac = 1.0
    if p_frac is None:
        p_frac = 1.0

    def _hp_class(frac):
        if frac > 0.50:
            return "battle_hp_high"
        elif frac >= 0.20:
            return "battle_hp_mid"
        else:
            return "battle_hp_low"

    e_hp_cls = _idx(_hp_class(e_frac), "battle_hp_bar")
    p_hp_cls = _idx(_hp_class(p_frac), "battle_hp_bar")

    # 2. Per-Pokémon species colours
    enemy_name = str(px.get("enemy_name", "")).strip().upper().replace(" ", "_")
    e_types = px.get("enemy_types", (0, 0))
    e_t1 = e_types[0] if isinstance(e_types, (tuple, list)) and len(e_types) > 0 else 0
    e_tname = TYPE_ID_TO_NAME.get(int(e_t1))

    enemy_cls = None
    if enemy_name and f"mon_{enemy_name.lower()}" in names_list:
        enemy_cls = names_list.index(f"mon_{enemy_name.lower()}")
    elif e_tname and f"mon_type_{e_tname}" in names_list:
        enemy_cls = names_list.index(f"mon_type_{e_tname}")
    else:
        enemy_cls = _idx("battle_enemy", "battle_bg")

    player_name = str(px.get("player_mon_name", "")).strip().upper().replace(" ", "_")
    p_types = px.get("player_mon_types", (0, 0))
    p_t1 = p_types[0] if isinstance(p_types, (tuple, list)) and len(p_types) > 0 else 0
    p_tname = TYPE_ID_TO_NAME.get(int(p_t1))

    player_cls = None
    if player_name and f"mon_{player_name.lower()}" in names_list:
        player_cls = names_list.index(f"mon_{player_name.lower()}")
    elif p_tname and f"mon_type_{p_tname}" in names_list:
        player_cls = names_list.index(f"mon_type_{p_tname}")
    else:
        player_cls = _idx("battle_player_mon", "battle_bg")

    # 3. Attack effect sprites (layer == 2 during battle)
    raw_move_type = px.get("move_type")
    m_type_val = int(raw_move_type) if raw_move_type is not None else 0x00
    m_tname = TYPE_ID_TO_NAME.get(m_type_val, "normal")
    effect_cls_name = f"effect_{m_tname}"
    effect_cls = _idx(effect_cls_name, "effect_normal" if "effect_normal" in names_list else "battle_enemy")

    box_cls = _idx("battle_box", "battle_bg")
    text_cls = _idx("text_box", "other")

    is_text = (tile >= 128) & (tile < 256)

    # Enemy Pokémon sprite (top right)
    enemy_sprite = ((tile >= 256) & (tile <= 304)) | ((y_coords < 56) & (x_coords >= 96) & (tile != 383))
    cls[enemy_sprite & ~is_text] = enemy_cls

    # HP Bar tiles: 355..363
    is_bar = (tile >= 355) & (tile <= 363)
    enemy_bar = is_bar & (y_coords < 56)
    cls[enemy_bar & ~is_text] = e_hp_cls
    player_bar = is_bar & (y_coords >= 56)
    cls[player_bar & ~is_text] = p_hp_cls

    # Player Pokémon back sprite (bottom left) vs Trainer during intro
    player_mon_out = px.get("player_mon_out")
    if player_mon_out is None:
        player_mon_out = bool(np.any(player_bar))

    trainer_cls = _idx("battle_trainer", "battle_player_mon")
    active_player_cls = player_cls if player_mon_out else trainer_cls

    player_sprite = ((tile >= 305) & (tile <= 353)) | ((y_coords >= 40) & (y_coords < 96) & (x_coords >= 8) & (x_coords < 64) & (tile != 383))
    cls[player_sprite & ~is_text] = active_player_cls

    # HP boxes and borders
    is_box = np.isin(tile, [354, 364, 365]) | ((tile >= 366) & (tile <= 382))
    cls[is_box & ~is_text] = box_cls

    # Attack effect sprites (layer == 2 in battle)
    if layer is not None:
        cls[layer == 2] = effect_cls

    # Text stays text_box
    cls[is_text] = text_cls

    return cls
