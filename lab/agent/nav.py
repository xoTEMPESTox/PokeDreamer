"""nav.py — Map-aware navigation and battle skills for Pokémon Red (System 1).

Honest documentation:
    This module implements a map-aware planner over map data read directly from
    Game Boy memory (privileged), not a learned model.
    It reads map block layouts from WRAM (wOverworldMap, 0xC6E8), map dimensions from
    wCurMapWidth/Height (0xD369/0xD368), tileset block definitions from ROM via
    wTilesetBlocksPtr (0xD52C) and wTilesetBank (0xD52B), and passable tiles from
    wTilesetCollisionPtr (0xD530). NPC positions are read from wSpriteStateData2 (0xC200).
    It uses Breadth-First Search (BFS) to find optimal obstacle-avoiding paths,
    and dynamically replans when a step is blocked or a battle intervenes.

Capabilities:
    1. build_walkability_grid: constructs a full-map 2D passable matrix from WRAM/ROM.
    2. bfs_path: computes shortest directional action sequence to target coordinates.
    3. navigate_to: executes goal-directed navigation with closed-loop replanning and mid-route battle recovery.
    4. battle_skill: in-battle skill reading party/enemy stats, choosing optimal moves (e.g. Ember vs Rock/Ground/Bug),
       reliably navigating the FIGHT menu cursor, handling multi-mon switch prompts, and allowing evolution.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from lab.agent.catch_quest import get_battle_dialog, is_battle_menu_active
from lab.agent.system12 import party_status
from lab.env import ROM_PATH, PokemonRed
from lab.game.ram_addresses import (
    HP_ADDRESSES,
    IN_BATTLE,
    LEVEL_ADDRESSES,
    MAP_ID,
    MAX_HP_ADDRESSES,
    PARTY_SIZE,
    PLAYER_X,
    PLAYER_Y,
    TEXT_BOX_ID,
)

# ── Map Identifiers (Verified against constants/map_constants.asm) ────────────
MAP_PALLET_TOWN                  = 0   # 0x00
MAP_VIRIDIAN_CITY                = 1   # 0x01
MAP_PEWTER_CITY                  = 2   # 0x02
MAP_ROUTE_1                      = 12  # 0x0C
MAP_ROUTE_2                      = 13  # 0x0D
MAP_VIRIDIAN_POKECENTER          = 41  # 0x29
MAP_VIRIDIAN_MART                = 42  # 0x2A
MAP_VIRIDIAN_FOREST_NORTH_GATE   = 47  # 0x2F
MAP_VIRIDIAN_FOREST_SOUTH_GATE   = 50  # 0x32
MAP_VIRIDIAN_FOREST              = 51  # 0x33
MAP_PEWTER_GYM                   = 54  # 0x36
MAP_PEWTER_POKECENTER            = 58  # 0x3A

# ── Move Constants (Verified against constants/move_constants.asm) ───────────
MOVE_SCRATCH = 0x0A  # 10
MOVE_LEER    = 0x2B  # 43
MOVE_GROWL   = 0x2D  # 45
MOVE_EMBER   = 0x34  # 52

MOVE_NAMES = {
    0x0A: "Scratch",
    0x2B: "Leer",
    0x2D: "Growl",
    0x34: "Ember",
}

# ── Type Constants (Verified against constants/type_constants.asm) ───────────
TYPE_NORMAL   = 0x00
TYPE_FIGHTING = 0x01
TYPE_FLYING   = 0x02
TYPE_POISON   = 0x03
TYPE_GROUND   = 0x04
TYPE_ROCK     = 0x05
TYPE_BUG      = 0x07
TYPE_FIRE     = 0x14
TYPE_WATER    = 0x15
TYPE_GRASS    = 0x16
TYPE_ELECTRIC = 0x17
TYPE_PSYCHIC  = 0x18
TYPE_ICE      = 0x19
TYPE_DRAGON   = 0x1A

# ── Memory Addresses ─────────────────────────────────────────────────────────
WRAM_OVERWORLD_MAP      = 0xC6E8  # Overworld map blocks buffer (1300 bytes)
WRAM_TILE_MAP           = 0xC3A0  # On-screen 20x18 tile map
WRAM_CUR_MAP_TILES      = 0xD367  # Current map tileset ID
WRAM_CUR_MAP_HEIGHT     = 0xD368  # Map height in 4x4 blocks
WRAM_CUR_MAP_WIDTH      = 0xD369  # Map width in 4x4 blocks
WRAM_TILESET_BANK       = 0xD52B  # ROM bank containing tileset block data
WRAM_TILESET_BLOCKS_PTR = 0xD52C  # 2-byte pointer to block -> tile mapping
WRAM_TILESET_COLL_PTR   = 0xD530  # 2-byte pointer to list of passable tiles
WRAM_SPRITE_STATE_2     = 0xC200  # NPC sprite state 2 table (16 structs * 16 bytes)
WRAM_CURRENT_MENU_ITEM  = 0xCC26  # Menu cursor item index (0-based)
WRAM_PLAYER_MOVE_INDEX  = 0xCC2C  # Remembered battle move cursor index
WRAM_ENEMY_MON_SPECIES  = 0xCFE5  # Enemy species ID
WRAM_ENEMY_MON_HP       = 0xCFE6  # 2-byte big-endian enemy current HP
WRAM_ENEMY_MON_TYPE1    = 0xCFEA  # Enemy type 1
WRAM_ENEMY_MON_TYPE2    = 0xCFEB  # Enemy type 2
WRAM_ENEMY_MON_LEVEL    = 0xCFF3  # Enemy level
WRAM_ENEMY_MON_MAX_HP   = 0xCFF4  # 2-byte big-endian enemy max HP
WRAM_ENEMY_MON_DEFENSE  = 0xCFF6  # 2-byte big-endian enemy defense
WRAM_ENEMY_MON_SPECIAL  = 0xCFFC  # 2-byte big-endian enemy special
WRAM_BATTLE_MON_MOVES   = 0xD01C  # Active battle mon moves (4 bytes)
WRAM_BATTLE_MON_PP      = 0xD02D  # Active battle mon PP (4 bytes)
WRAM_PARTY_MON1_MOVES   = 0xD173  # Party mon 1 moves (4 bytes)
WRAM_PARTY_MON1_PP      = 0xD188  # Party mon 1 PP (4 bytes)
WRAM_OBTAINED_BADGES    = 0xD356  # Bitmask: bit 0 = Boulder Badge

# ── Warp / Transition Table for Quest Route ──────────────────────────────────
# Documented doorway/exit tiles per map on the Viridian City -> Pewter Gym route:
ROUTE_WARPS: dict[tuple[int, int], list[tuple[int, int]]] = {
    # Viridian City (1) -> Route 1 (12): step south across Y=35
    (MAP_VIRIDIAN_CITY, MAP_ROUTE_1): [(21, 35), (22, 35)],
    # Route 1 (12) -> Viridian City (1): step north across Y=0
    (MAP_ROUTE_1, MAP_VIRIDIAN_CITY): [(10, 0), (11, 0)],
    # Viridian City (1) -> Route 2 (13): step north across Y=0
    (MAP_VIRIDIAN_CITY, MAP_ROUTE_2): [(17, 0), (18, 0), (19, 0)],
    # Viridian City (1) -> Viridian Pokecenter (41): door at (23, 25), step from (23, 26)
    (MAP_VIRIDIAN_CITY, MAP_VIRIDIAN_POKECENTER): [(23, 25), (23, 26)],
    # Viridian Pokecenter (41) -> Viridian City (1): exit doors
    (MAP_VIRIDIAN_POKECENTER, MAP_VIRIDIAN_CITY): [(3, 7), (4, 7)],
    # Route 2 (13, south) -> Viridian Forest South Gate (50): door at (3, 43)
    (MAP_ROUTE_2, MAP_VIRIDIAN_FOREST_SOUTH_GATE): [(3, 43)],
    # South Gate (50) -> Viridian Forest (51): exit door north at (5, 0)
    (MAP_VIRIDIAN_FOREST_SOUTH_GATE, MAP_VIRIDIAN_FOREST): [(5, 0)],
    # South Gate (50) -> Route 2 (13, south): exit door south at (4, 7) / (5, 7)
    (MAP_VIRIDIAN_FOREST_SOUTH_GATE, MAP_ROUTE_2): [(4, 7), (5, 7)],
    # Route 2 (13, south) -> Viridian City (1): step south across Y=71
    (MAP_ROUTE_2, MAP_VIRIDIAN_CITY): [(x, 71) for x in range(15, 21)],
    # Viridian Forest (51) -> North Gate (47): exit door at (1, 0)
    (MAP_VIRIDIAN_FOREST, MAP_VIRIDIAN_FOREST_NORTH_GATE): [(1, 0), (2, 0)],
    # Viridian Forest (51) -> South Gate (50): exit doors at (16, 47), (17, 47)
    (MAP_VIRIDIAN_FOREST, MAP_VIRIDIAN_FOREST_SOUTH_GATE): [(16, 47), (17, 47)],
    # North Gate (47) -> Route 2 (13, north): exit door north at (5, 0)
    (MAP_VIRIDIAN_FOREST_NORTH_GATE, MAP_ROUTE_2): [(5, 0), (4, 0)],
    # North Gate (47) -> Viridian Forest (51): exit door south at (4, 7), (5, 7)
    (MAP_VIRIDIAN_FOREST_NORTH_GATE, MAP_VIRIDIAN_FOREST): [(4, 7), (5, 7)],
    # Route 2 (13, north) -> Pewter City (2): step north across Y=0
    (MAP_ROUTE_2, MAP_PEWTER_CITY): [(x, 0) for x in range(20)],
    # Pewter City (2) -> Pewter Pokecenter (58): door at (13, 25), approach from (13, 26)
    (MAP_PEWTER_CITY, MAP_PEWTER_POKECENTER): [(13, 25), (13, 26)],
    # Pewter Pokecenter (58) -> Pewter City (2): exit doors
    (MAP_PEWTER_POKECENTER, MAP_PEWTER_CITY): [(3, 7), (4, 7)],
    # Pewter City (2) -> Pewter Gym (54): door at (16, 17), approach from (16, 18)
    (MAP_PEWTER_CITY, MAP_PEWTER_GYM): [(16, 17), (16, 18)],
    # Pewter Gym (54) -> Pewter City (2): exit doors
    (MAP_PEWTER_GYM, MAP_PEWTER_CITY): [(4, 13), (5, 13)],
}

# Cache for ROM blockset data: (bank, ptr) -> bytes
_ROM_BLOCK_CACHE: dict[tuple[int, int], bytes] = {}


def _get_block_data(rom_path: Path, bank: int, ptr: int) -> bytes:
    """Read tileset block mapping data (16 bytes per block) from ROM."""
    key = (bank, ptr)
    if key in _ROM_BLOCK_CACHE:
        return _ROM_BLOCK_CACHE[key]
    rom_offset = bank * 0x4000 + (ptr - 0x4000)
    with open(rom_path, "rb") as f:
        f.seek(rom_offset)
        data = f.read(16 * 256)
    _ROM_BLOCK_CACHE[key] = data
    return data


# ── Walkability Grid Construction ─────────────────────────────────────────────
def build_walkability_grid(
    env: PokemonRed,
    dynamic_blocked: Optional[Set[Tuple[int, int]]] = None,
    target_tiles: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[int, int, Callable[[int, int], bool]]:
    """Build a walkability query function for the currently loaded map.

    Reads:
        - wCurMapWidth (0xD369) and wCurMapHeight (0xD368) in blocks.
        - wOverworldMap (0xC6E8) block matrix (with 3-block padding on all sides).
        - wTilesetBlocksPtr (0xD52C) and wTilesetBank (0xD52B) for block->tile mappings.
        - wTilesetCollisionPtr (0xD530) for list of passable tile IDs terminated by 0xFF.
        - wSpriteStateData2 (0xC200) for active NPC positions.

    Returns:
        width_tiles (int): Total map width in player tiles (blocks * 2).
        height_tiles (int): Total map height in player tiles (blocks * 2).
        is_walkable(tx, ty): Pure boolean function returning True if (tx, ty) can be entered.
    """
    mem = env.pyboy.memory
    w_blocks = int(mem[WRAM_CUR_MAP_WIDTH])
    h_blocks = int(mem[WRAM_CUR_MAP_HEIGHT])
    w_tiles = w_blocks * 2
    h_tiles = h_blocks * 2
    stride = w_blocks + 6

    b_ptr = (int(mem[WRAM_TILESET_BLOCKS_PTR + 1]) << 8) | int(mem[WRAM_TILESET_BLOCKS_PTR])
    bank = int(mem[WRAM_TILESET_BANK])
    block_data = _get_block_data(env.rom_path, bank, b_ptr)

    col_ptr = (int(mem[WRAM_TILESET_COLL_PTR + 1]) << 8) | int(mem[WRAM_TILESET_COLL_PTR])
    passable_tiles: set[int] = set()
    addr = col_ptr
    while True:
        b = int(mem[addr])
        if b == 0xFF:
            break
        passable_tiles.add(b)
        addr += 1

    # Active NPC obstacles (sprites 1..15)
    npc_positions: set[tuple[int, int]] = set()
    for s in range(1, 16):
        base2 = WRAM_SPRITE_STATE_2 + s * 0x10
        raw_x = int(mem[base2 + 5])
        raw_y = int(mem[base2 + 4])
        if raw_x > 0 or raw_y > 0:
            npc_pos = (raw_x - 4, raw_y - 4)
            # If target explicitly includes this NPC (e.g. interacting with Brock/Nurse), do not block
            if not target_tiles or npc_pos not in target_tiles:
                npc_positions.add(npc_pos)

    blocked_set = dynamic_blocked or set()

    def is_walkable(tx: int, ty: int) -> bool:
        if (tx, ty) in blocked_set:
            return False
        if not (0 <= tx < w_tiles and 0 <= ty < h_tiles):
            return False
        if (tx, ty) in npc_positions:
            return False
        bx = tx // 2
        by = ty // 2
        sub_x = tx % 2
        sub_y = ty % 2
        block_idx = int(mem[WRAM_OVERWORLD_MAP + stride * (by + 3) + (bx + 3)])
        # Sub-block tile offset: (sub_y * 2 + 1) * 4 + (sub_x * 2)
        offset = (sub_y * 2 + 1) * 4 + (sub_x * 2)
        tile_id = block_data[block_idx * 16 + offset]
        return tile_id in passable_tiles

    return w_tiles, h_tiles, is_walkable


# ── Breadth-First Search (BFS) Pathfinding ───────────────────────────────────
def bfs_path(
    env: PokemonRed,
    target_tiles: list[tuple[int, int]] | set[tuple[int, int]],
    dynamic_blocked: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[List[str]]:
    """Compute shortest sequence of direction actions from player pos to target_tiles.

    Actions: 'down', 'left', 'right', 'up'.
    Returns None if no reachable path exists.
    """
    targets = set(target_tiles)
    w_tiles, h_tiles, is_walkable = build_walkability_grid(env, dynamic_blocked, target_tiles=targets)
    _, ram = env.observe()
    sx, sy = int(ram["x"]), int(ram["y"])

    if (sx, sy) in targets:
        return []

    queue: deque[tuple[int, int, list[str]]] = deque([(sx, sy, [])])
    visited: set[tuple[int, int]] = {(sx, sy)}
    blocked = dynamic_blocked or set()

    # Move deltas: (dx, dy, action_name)
    moves = [
        (0, 1, "down"),
        (-1, 0, "left"),
        (1, 0, "right"),
        (0, -1, "up"),
    ]

    while queue:
        cx, cy, path = queue.popleft()
        if (cx, cy) in targets:
            return path
        for dx, dy, act in moves:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            # Allow entering target tiles even if considered obstacle/door by collision
            if (nx, ny) in targets and (nx, ny) not in blocked:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [act]))
            elif is_walkable(nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny, path + [act]))

    return None


# ── Move Choice Rule ─────────────────────────────────────────────────────────
def choose_battle_move(env: PokemonRed) -> tuple[int, str]:
    """Select the best move index and name according to game mechanics rule.

    Rule:
        - Geodude & Onix (Rock/Ground) and Viridian Forest Bug Pokémon have very low Special
          and/or high Defense. Ember (Special, 40 power, Fire) melts them.
        - If Ember has PP > 0 and enemy is Rock, Ground, Bug, Grass, Ice or has Defense > Special:
          choose EMBER.
        - If Scratch (Normal physical, 40 power) has PP > 0:
          used against neutral/flying/normal targets with low defense (e.g. Rattata, Pidgey)
          to preserve Ember PP, or when Ember is out of PP.
        - Fallback: first available move with PP > 0.
        - If all moves out of PP: index 0 (Struggle).

    Returns:
        (move_index_0_to_3, move_name)
    """
    mem = env.pyboy.memory

    # Enemy status
    e_type1 = int(mem[WRAM_ENEMY_MON_TYPE1])
    e_type2 = int(mem[WRAM_ENEMY_MON_TYPE2])
    e_types = {e_type1, e_type2}
    e_def = (int(mem[WRAM_ENEMY_MON_DEFENSE]) << 8) | int(mem[WRAM_ENEMY_MON_DEFENSE + 1])
    e_spc = (int(mem[WRAM_ENEMY_MON_SPECIAL]) << 8) | int(mem[WRAM_ENEMY_MON_SPECIAL + 1])

    # Player active moves (from battle_mon or party_mon1)
    moves = [int(mem[WRAM_BATTLE_MON_MOVES + i]) for i in range(4)]
    if all(m == 0 for m in moves):
        moves = [int(mem[WRAM_PARTY_MON1_MOVES + i]) for i in range(4)]
    pps = [int(mem[WRAM_BATTLE_MON_PP + i]) & 0x3F for i in range(4)]
    if all(p == 0 for p in pps):
        pps = [int(mem[WRAM_PARTY_MON1_PP + i]) & 0x3F for i in range(4)]

    ember_idx = moves.index(MOVE_EMBER) if MOVE_EMBER in moves and pps[moves.index(MOVE_EMBER)] > 0 else None
    scratch_idx = moves.index(MOVE_SCRATCH) if MOVE_SCRATCH in moves and pps[moves.index(MOVE_SCRATCH)] > 0 else None

    # Ember targets: Rock, Ground, Bug, Grass, Ice, or low Special targets
    weak_to_fire_types = {TYPE_ROCK, TYPE_GROUND, TYPE_BUG, TYPE_GRASS, TYPE_ICE}
    prefer_ember = bool(e_types & weak_to_fire_types) or (e_def >= 20 and e_def > e_spc and e_spc > 0)

    if prefer_ember and ember_idx is not None:
        return ember_idx, "Ember"
    elif scratch_idx is not None:
        return scratch_idx, "Scratch"
    elif ember_idx is not None:
        return ember_idx, "Ember"

    # Any move with PP
    for idx, (m, p) in enumerate(zip(moves, pps)):
        if m != 0 and p > 0:
            return idx, MOVE_NAMES.get(m, f"Move_0x{m:02X}")

    return 0, "Struggle"


# ── System 1 Battle Skill ────────────────────────────────────────────────────
def battle_skill(env: PokemonRed, budget: int = 800) -> dict[str, Any]:
    """Autonomous battle reflex with move selection, switch prompt handling, and evolution.

    Handles:
        1. 2x2 Battle menu (FIGHT, PKMN, ITEM, RUN): ensures cursor is on FIGHT at (14, 9), selects FIGHT.
        2. Move selection menu: tracks cursor arrow 0xED in col 5 (rows 13..16), navigates to target move, confirms with A.
        3. Switch prompt ('Will [Player] change POKéMON?'): declines with B.
        4. Move learning prompt ('learned [Move]'): clears with A.
        5. Evolution sequence: Charmander -> Charmeleon at L16: advances with A (never B).

    Returns:
        dict with won (bool), blackout (bool), presses (int), moves_used (dict).
    """
    presses = 0
    moves_used: dict[str, int] = {}
    _, ram = env.observe()
    if not ram["in_battle"]:
        return {"won": True, "blackout": False, "presses": 0, "moves_used": moves_used}

    mem = env.pyboy.memory
    start_map = int(ram["map_id"])
    blackout_occurred = False

    while env.observe()[1]["in_battle"] and presses < budget:
        # Check if starter HP dropped to 0
        st = party_status(env)
        if st["hp"] == 0 and st["party_size"] > 0:
            blackout_occurred = True

        # Check if 2x2 Battle Menu is awaiting input
        if is_battle_menu_active(env):
            # Ensure cursor is at FIGHT: row 14, col 9
            if mem[WRAM_TILE_MAP + 14 * 20 + 9] != 0xED:
                env.step("up")
                env.step("left")
                presses += 2
                continue

            env.step("a")
            presses += 1
            continue

        # Check if Move Selection Menu is awaiting input: cursor arrow 0xED in column 5
        cur_cursor = None
        for m_i in range(4):
            if mem[WRAM_TILE_MAP + (13 + m_i) * 20 + 5] == 0xED:
                cur_cursor = m_i
                break

        if cur_cursor is not None:
            target_idx, move_name = choose_battle_move(env)
            for _ in range(4):
                if cur_cursor == target_idx or presses >= budget:
                    break
                act_move = "down" if cur_cursor < target_idx else "up"
                env.step(act_move)
                presses += 1
                for m_i in range(4):
                    if mem[WRAM_TILE_MAP + (13 + m_i) * 20 + 5] == 0xED:
                        cur_cursor = m_i
                        break

            # Confirm move with A
            env.step("a")
            presses += 1
            moves_used[move_name] = moves_used.get(move_name, 0) + 1
            continue

        # Check for multi-Pokémon switch prompt ('Will you change POKéMON?')
        dialog = get_battle_dialog(env)
        if "change" in dialog or "Will" in dialog:
            env.step("b")  # B declines switch prompt (selects NO)
            presses += 1
            continue

        # Check if party menu was accidentally opened: back out with B
        if mem[WRAM_TILE_MAP + 1 * 20 + 1] in (0x80, 0x90):  # party screen indicator
            env.step("b")
            presses += 1
            continue

        # Advance battle animations, dialogue, EXP gain, and level up screens with A
        env.step("a")
        presses += 1

    # Check battle conclusion and blackout state
    st = party_status(env)
    end_map = int(env.observe()[1]["map_id"])
    blackout = (
        st["hp"] == 0
        or blackout_occurred
        or (end_map in (MAP_VIRIDIAN_POKECENTER, MAP_PEWTER_POKECENTER) and start_map not in (MAP_VIRIDIAN_POKECENTER, MAP_PEWTER_POKECENTER))
    )
    won = not blackout and (st["hp"] > 0)

    # Post-battle text clearance and evolution handling
    # Charmander evolves at Level 16. Pressing B cancels evolution! Press A through evolution.
    for _ in range(40):
        if presses >= budget:
            break
        if mem[TEXT_BOX_ID] != 0:
            env.step("a")
        else:
            env.step("pass")
        presses += 1
        if mem[TEXT_BOX_ID] == 0 and not env.observe()[1]["in_battle"]:
            break

    return {
        "won": won,
        "blackout": blackout,
        "presses": presses,
        "moves_used": moves_used,
    }


# ── System 1 Heal Skill (Universal for Viridian & Pewter) ────────────────────
def heal_pkmn_center(env: PokemonRed, budget: int = 600, exit_after: bool = True) -> dict[str, Any]:
    """Universal heal skill for Viridian (map 41) or Pewter (map 58) Pokémon Centers.

    Navigates into the local Pokémon Center from anywhere, approaches Nurse Joy at (3, 3),
    conducts healing dialogue with A until full HP, clears text with B, and exits.
    """
    presses = 0
    _, ram = env.observe()
    cur_map = int(ram["map_id"])
    mem = env.pyboy.memory

    if cur_map in (MAP_PEWTER_CITY, MAP_PEWTER_GYM, MAP_PEWTER_POKECENTER):
        target_pc = MAP_PEWTER_POKECENTER
        target_city = MAP_PEWTER_CITY
    else:
        target_pc = MAP_VIRIDIAN_POKECENTER
        target_city = MAP_VIRIDIAN_CITY

    # Step 1: If not in target city or target PC, navigate to target city first
    if cur_map != target_pc and cur_map != target_city:
        r_city = navigate_to(env, target_city, budget=budget - presses)
        presses += r_city["presses"]
        cur_map = int(env.observe()[1]["map_id"])

    # Step 2: If in target city, navigate into Pokémon Center
    if cur_map == target_city:
        r_nav = navigate_to(env, target_pc, budget=budget - presses)
        presses += r_nav["presses"]
        cur_map = int(env.observe()[1]["map_id"])

    if cur_map != target_pc:
        return {"presses": presses, "success": False, "healed": False}

    # Step 2: Walk to Nurse Joy counter at (3, 3)
    r_counter = navigate_to(env, cur_map, target_tiles=(3, 3), budget=budget - presses)
    presses += r_counter["presses"]

    # Face UP towards Nurse Joy
    while int(env.observe()[1]["facing"]) != 4 and presses < budget:
        env.step("up")
        presses += 1

    # Step 3: Talk to Nurse Joy with A until HP is fully restored
    max_hp = (int(mem[MAX_HP_ADDRESSES[0]]) << 8) | int(mem[MAX_HP_ADDRESSES[0] + 1])
    healed = False
    for _ in range(60):
        if presses >= budget:
            break
        cur_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
        if cur_hp == max_hp:
            healed = True
            break
        env.step("a")
        presses += 1

    # Step 4: Step down and clear dialogue
    while int(env.observe()[1]["y"]) == 3 and presses < budget:
        env.step("down")
        presses += 1
        if int(env.observe()[1]["y"]) == 3:
            env.step("b")
            presses += 1

    # Step 5: Exit back to overworld city
    exit_map = MAP_VIRIDIAN_CITY if cur_map == MAP_VIRIDIAN_POKECENTER else MAP_PEWTER_CITY
    if exit_after:
        r_exit = navigate_to(env, exit_map, budget=budget - presses)
        presses += r_exit["presses"]

    final_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
    success = healed and (final_hp == max_hp) and (not exit_after or int(env.observe()[1]["map_id"]) == exit_map)
    return {
        "presses": presses,
        "success": success,
        "healed": healed,
        "hp": final_hp,
        "max_hp": max_hp,
    }


# ── Map-Aware Navigation Skill (System 1) ────────────────────────────────────
def navigate_to(
    env: PokemonRed,
    target_map: int | list[int] | tuple[int, ...],
    target_tiles: Optional[list[tuple[int, int]] | tuple[int, int]] = None,
    budget: int = 2500,
    battle_handler: Optional[Callable[[PokemonRed], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Navigate to target map and/or target coordinates with closed-loop BFS.

    Features:
        - Reads exact map obstacles from WRAM/ROM.
        - Resolves mid-route wild encounters and trainer battles using battle_skill.
        - Automatically replans when an action is blocked or NPC moves.
        - Handles warp doors and transition loading.
    """
    mem = env.pyboy.memory
    presses = 0
    events: list[dict[str, Any]] = []
    battles_started = 0
    battles_won = 0
    blackouts = 0
    handler = battle_handler or battle_skill
    target_maps = [target_map] if isinstance(target_map, int) else list(target_map)

    # Normalize target_tiles
    if target_tiles is None:
        target_set: set[tuple[int, int]] = set()
    elif isinstance(target_tiles, tuple) and len(target_tiles) == 2 and isinstance(target_tiles[0], int):
        target_set = {target_tiles}
    else:
        target_set = set(target_tiles)

    blocked_cells: set[tuple[int, int]] = set()
    prev_map = int(env.observe()[1]["map_id"])

    while presses < budget:
        _, ram = env.observe()
        cur_map = int(ram["map_id"])
        cur_pos = (int(ram["x"]), int(ram["y"]))

        # Success check: on target map and (if specified) on target tiles
        if cur_map in target_maps:
            if not target_set or cur_pos in target_set:
                # Settle transition into new map coordinates
                for _ in range(4):
                    env.step("pass")
                    presses += 1
                return {
                    "presses": presses,
                    "success": True,
                    "events": events,
                    "battles_started": battles_started,
                    "battles_won": battles_won,
                    "blackouts": blackouts,
                }

        # Check for party fainting / overworld blackout (e.g. poison in overworld)
        st_party = party_status(env)
        if st_party["hp"] == 0 and st_party["party_size"] > 0:
            blackouts += 1
            # Clear blackout dialog until respawn in Pokemon Center
            for _ in range(50):
                if presses >= budget:
                    break
                env.step("a")
                presses += 1
                if party_status(env)["map_id"] in (MAP_VIRIDIAN_POKECENTER, MAP_PEWTER_POKECENTER):
                    break
            events.append({"type": "blackout", "map": cur_map, "presses": presses})
            return {
                "presses": presses,
                "success": False,
                "events": events,
                "battles_started": battles_started,
                "battles_won": battles_won,
                "blackouts": blackouts,
            }

        # Map transition handling: clear dynamic blocks on map switch
        if cur_map != prev_map:
            prev_map = cur_map
            blocked_cells.clear()
            for _ in range(4):
                if presses < budget:
                    env.step("pass")
                    presses += 1
            continue

        # In-Battle reflex
        if ram["in_battle"]:
            battles_started += 1
            b_res = handler(env)
            presses += b_res.get("presses", 0)
            if b_res.get("blackout", False):
                blackouts += 1
                events.append({"type": "blackout", "map": cur_map, "presses": presses})
                return {
                    "presses": presses,
                    "success": False,
                    "events": events,
                    "battles_started": battles_started,
                    "battles_won": battles_won,
                    "blackouts": blackouts,
                }
            battles_won += 1
            events.append({"type": "battle_won", "map": cur_map, "moves": b_res.get("moves_used", {})})
            blocked_cells.clear()
            continue

        # Trainer line-of-sight pre-battle dialogue box or interaction text
        # If player did not move on last step and not in battle, pressing A/B advances text

        # Determine target coordinates on current map
        if cur_map in target_maps:
            active_targets = target_set
        else:
            active_targets = set()
            for t_map in target_maps:
                if (cur_map, t_map) in ROUTE_WARPS:
                    active_targets = set(ROUTE_WARPS[(cur_map, t_map)])
                    break

        if not active_targets:
            # No warp defined, cannot route
            break

        # If player reached warp/border tile on current map, step through the transition
        if cur_pos in active_targets and cur_map not in target_maps:
            # Transition step into warp/border
            if cur_pos[1] == 0:
                trans_act = "up"
            elif cur_pos[1] >= (int(mem[WRAM_CUR_MAP_HEIGHT]) * 2 - 1):
                trans_act = "down"
            elif cur_map in (MAP_VIRIDIAN_POKECENTER, MAP_PEWTER_POKECENTER, MAP_PEWTER_GYM):
                trans_act = "down"
            else:
                trans_act = "up"
            env.step(trans_act)
            presses += 1
            for _ in range(4):
                if presses < budget:
                    env.step("pass")
                    presses += 1
            continue

        # Compute BFS path
        path = bfs_path(env, active_targets, dynamic_blocked=blocked_cells)
        if not path:
            # If path blocked by dynamic obstacles, reset dynamic obstacles and retry
            if blocked_cells:
                blocked_cells.clear()
                path = bfs_path(env, active_targets)
            if not path:
                # Completely unreachable
                break

        # Execute next action
        act = path[0]
        dx, dy = {"down": (0, 1), "up": (0, -1), "left": (-1, 0), "right": (1, 0)}[act]
        intended_cell = (cur_pos[0] + dx, cur_pos[1] + dy)

        _, next_ram = env.step(act)
        presses += 1
        new_pos = (int(next_ram["x"]), int(next_ram["y"]))

        # Detect physical block or trainer engagement: position did not change and not in battle
        if new_pos == cur_pos and not next_ram["in_battle"]:
            # Check if this is a trainer engaging the player or text dialog
            engaged = False
            for t_step in range(16):
                if presses >= budget:
                    break
                t_act = "a" if t_step % 2 == 1 else "pass"
                _, chk_ram = env.step(t_act)
                presses += 1
                if chk_ram["in_battle"]:
                    engaged = True
                    break
            if not engaged and (int(env.observe()[1]["x"]), int(env.observe()[1]["y"])) == cur_pos and not env.observe()[1]["in_battle"]:
                blocked_cells.add(intended_cell)

    success = (env.observe()[1]["map_id"] in target_maps) and (not target_set or (int(env.observe()[1]["x"]), int(env.observe()[1]["y"])) in target_set)
    return {
        "presses": presses,
        "success": success,
        "events": events,
        "battles_started": battles_started,
        "battles_won": battles_won,
        "blackouts": blackouts,
    }
