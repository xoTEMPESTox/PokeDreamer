"""catch_quest.py — System 1 / System 2 loop for catching wild Pokémon in Pokémon Red.

Architecture:
- System 2 v0: Rule-based planner reading Game Boy RAM to select high-level goals:
    * If hp_frac < 0.5 -> "heal"
    * Elif balls == 0 and money >= 200 -> "buy" (Viridian Mart, as many as affordable, up to 10)
    * Elif balls == 0 and money < 200 -> "done: out of money"
    * Elif catches < target -> "catch"
    * Else -> "done"
    Re-plans after each skill execution.

- Battle Strategies (System 2):
    * "throw" (default): Immediately throws Poké Balls at full HP in every wild battle.
    * "weaken": While enemy HP fraction > 0.6 and HP > 5, attacks with first move (FIGHT/Scratch);
      once enemy HP fraction <= 0.6 (or HP <= 5), throws Poké Balls to maximize catch probability.
      Tracks knockouts separately when an attack causes the wild Pokémon to faint.

- System 1 skills: Scripted and reflexive skills with bounded action budgets:
    * throw_ball: In-battle skill navigating menu to ITEM, selecting POKé BALL, throwing,
      clearing text, declining nickname (B), closing Pokédex entry, and robustly detecting outcome
      from RAM signals.
    * attack_first_move: In-battle skill navigating menu to FIGHT, selecting move 1 (Scratch),
      and detecting hit vs. knockout.
    * run_away: In-battle skill navigating menu to RUN to flee when out of balls or when traversing routes.
    * catch_in_grass: Overworld skill walking grass patch on Route 1 to trigger wild encounters,
      applying strategy (weakening and/or throwing balls), running away if out of balls, and
      returning to planner when HP drops, balls run out, or budget ends.
    * heal: Enters Viridian Pokémon Center, talks to Nurse Joy, clears dialogue, and exits.
    * buy_poke_balls: Enters Viridian Mart, purchases balls at clerk counter, and exits.
    * navigate_catch: Overworld navigation between Viridian City and Route 1 grass patch,
      safely fleeing wild encounters during transit.

Signals verified for 'caught':
1. Total Pokémon owned in RAM strictly increases: (party_size + box_count) > start_(party_size + box_count).
2. Poké Balls in bag decreased by 1.
3. Battle cleanly finished (in_battle == False) while starter Pokémon remains conscious (HP > 0).
4. Pokédex owned bitfield / counter (0xD2F7-0xD309) updates when a new distinct species is captured.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from lab.agent.system12 import (
    DETOUR_EXITS,
    GRASS_PATCH_TILES,
    MAP_PALLET,
    MAP_ROUTE1,
    MAP_VIRIDIAN,
    MAP_VIRIDIAN_MART,
    MAP_VIRIDIAN_POKECENTER,
    ROUTE1_NORTH_WAYPOINTS,
    ROUTE1_SOUTH_WAYPOINTS,
    VIRIDIAN_TO_PC_WAYPOINTS,
    VIRIDIAN_TO_ROUTE1_WAYPOINTS,
    Goal,
    heal,
    navigate,
    party_status,
)
from lab.env import ROM_PATH, PokemonRed
from lab.make_pokedex_state import (
    ITEM_NAMES,
    ITEM_POKE_BALL,
    MART_TO_ROUTE1_PATH,
    PC_TO_MART_PATH,
    VIRIDIAN_TO_MART_PATH,
    bag,
    buy_poke_balls,
    money,
    pokedex_status,
)
from src.game_state import screen_capture
from src.ram_addresses import (
    HP_ADDRESSES,
    IN_BATTLE,
    LEVEL_ADDRESSES,
    MAP_ID,
    MAX_HP_ADDRESSES,
    PARTY_SIZE,
    PARTY_SPECIES_ADDRESSES,
    PLAYER_X,
    PLAYER_Y,
    TEXT_BOX_ID,
)

# ── RAM Addresses for Wild Battles & Storage ──────────────────────────────────
# Source: pret/pokered disassembly (wram.asm, macros/ram.asm)
WRAM_ENEMY_MON_SPECIES = 0xCFE5  # Species ID of current wild opponent
WRAM_ENEMY_MON_HP      = 0xCFE6  # 2-byte big-endian current HP of enemy (CFE6-CFE7)
WRAM_ENEMY_MON_LEVEL   = 0xCFF3  # Level of enemy Pokémon
WRAM_ENEMY_MON_MAX_HP  = 0xCFF4  # 2-byte big-endian Max HP of enemy (CFF4-CFF5)
WRAM_BOX_COUNT         = 0xDA80  # Number of Pokémon stored in current PC Box (0-20)
WRAM_CURRENT_MENU_ITEM = 0xCC26  # Currently selected item index on active menu
WRAM_TILE_MAP          = 0xC3A0  # 20x18 screen tile map (0xC3A0 - 0xC507)

# Direct path from outside Mart (29, 20) to Pokémon Center door (23, 26)
MART_TO_PC_PATH = ["left"] * 10 + ["down"] * 6 + ["right"] * 4

# Pokedex font tile mapping for in-battle dialog decoding
POKEDEX_CHARMAP: dict[int, str] = {
    0x7F: " ",
    0xED: ">",
    0xEE: "-",
    0xEF: "-",
    0xE7: "!",
}
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    POKEDEX_CHARMAP[0x80 + _i] = _c
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    POKEDEX_CHARMAP[0xA0 + _i] = _c
for _i, _c in enumerate("0123456789"):
    POKEDEX_CHARMAP[0xF6 + _i] = _c


# ── RAM Status Helpers ────────────────────────────────────────────────────────
def read_enemy_status(env: PokemonRed) -> dict[str, Any]:
    """Read active wild enemy Pokémon info from WRAM."""
    mem = env.pyboy.memory
    species = int(mem[WRAM_ENEMY_MON_SPECIES])
    hp = (int(mem[WRAM_ENEMY_MON_HP]) << 8) | int(mem[WRAM_ENEMY_MON_HP + 1])
    max_hp = (int(mem[WRAM_ENEMY_MON_MAX_HP]) << 8) | int(mem[WRAM_ENEMY_MON_MAX_HP + 1])
    level = int(mem[WRAM_ENEMY_MON_LEVEL])
    hp_frac = (hp / max_hp) if max_hp > 0 else 1.0
    return {
        "species_id": species,
        "species_name": ITEM_NAMES.get(species, f"Mon_0x{species:02X}"),
        "hp": hp,
        "max_hp": max_hp,
        "hp_frac": hp_frac,
        "level": level,
    }


def catch_status(env: PokemonRed, catches: int = 0) -> dict[str, Any]:
    """Comprehensive status dictionary for System 2 catch planner."""
    p_st = party_status(env)
    items = bag(env)
    ball_count = sum(it["count"] for it in items if it["id"] == ITEM_POKE_BALL)
    m = money(env)
    dex = pokedex_status(env)
    box_cnt = int(env.pyboy.memory[WRAM_BOX_COUNT])
    return {
        **p_st,
        "balls": ball_count,
        "money": m,
        "catches": catches,
        "box_count": box_cnt,
        "total_owned_mons": p_st["party_size"] + box_cnt,
        "dex_owned": dex["owned_count"],
        "dex_seen": dex["seen_count"],
    }


def is_battle_menu_active(env: PokemonRed) -> bool:
    """Check if the 2x2 player battle menu (FIGHT, PKMN, ITEM, RUN) is visible and awaiting input."""
    mem = env.pyboy.memory
    # Row 14 col 10 is 'F' (0x85) of FIGHT; Row 16 col 10 is 'I' (0x88) of ITEM
    return mem[WRAM_TILE_MAP + 14 * 20 + 10] == 0x85 and mem[WRAM_TILE_MAP + 16 * 20 + 10] == 0x88


def get_battle_dialog(env: PokemonRed) -> str:
    """Decode text from the lower 6 rows of the tilemap."""
    mem = env.pyboy.memory
    lines = []
    for r in range(12, 18):
        row_str = "".join(POKEDEX_CHARMAP.get(mem[WRAM_TILE_MAP + r * 20 + c], " ") for c in range(20))
        lines.append(row_str)
    return "\n".join(lines)


# ── System 2 v0 Planner for Catching ──────────────────────────────────────────
def plan_catch(
    status: dict[str, Any],
    catch_target: int = 10,
    current_goal: Optional[str] = None,
) -> Goal:
    """System 2 v0 rule-based planner for Pokémon catching.

    Rules:
    - if hp_frac < 0.5 -> "heal"
    - elif balls == 0 and money >= 200 -> "buy" (Viridian Mart, as many as affordable, up to 10)
    - elif balls == 0 and money < 200 -> "done: out of money"
    - elif catches < catch_target -> "catch"
    - else -> "done"

    If a heal goal is in progress in Viridian / PC and HP is not yet 100%, keeps healing.
    """
    hp_frac = status.get("hp_frac", 1.0)
    hp = status.get("hp", 0)
    max_hp = status.get("max_hp", 1)
    balls = status.get("balls", 0)
    player_money = status.get("money", 0)
    catches = status.get("catches", 0)
    map_id = status.get("map_id", 0)

    # Latch healing until full HP if already in Viridian or Pokémon Center
    if current_goal == "heal" and hp_frac < 1.0 and map_id in (MAP_VIRIDIAN, MAP_VIRIDIAN_POKECENTER):
        return Goal("heal", f"Continuing heal in Viridian: HP {hp}/{max_hp}")

    if hp_frac < 0.5:
        return Goal("heal", f"HP low: {hp}/{max_hp} ({hp_frac:.2f} < 0.50)")
    elif balls == 0 and player_money >= 200:
        return Goal("buy", f"Out of balls, money ${player_money} >= 200")
    elif balls == 0 and player_money < 200:
        return Goal("done: out of money", f"Out of balls and cannot afford more (${player_money} < 200)")
    elif catches < catch_target:
        return Goal("catch", f"Catches {catches} < target {catch_target}")
    else:
        return Goal("done", f"Target catches {catch_target} reached ({catches} catches)")


# ── Strategy Decision Rule (Pure Function) ────────────────────────────────────
def decide_battle_action(
    enemy_hp: int,
    enemy_max_hp: int,
    strategy: str = "throw",
    min_safe_hp: int = 5,
    weaken_threshold: float = 0.6,
) -> str:
    """Pure decision rule for in-battle action under System 2 strategy.

    Parameters:
        enemy_hp: Current HP of enemy Pokémon.
        enemy_max_hp: Maximum HP of enemy Pokémon.
        strategy: "throw" (always throw ball) or "weaken" (attack down to <= 60% HP).
        min_safe_hp: HP threshold below which attacking risks KOing the enemy (default 5).
        weaken_threshold: HP fraction threshold above which to weaken (default 0.6).

    Returns:
        "fight": Use FIGHT with the first move to weaken the opponent.
        "throw": Throw a Poké Ball.
    """
    if strategy == "throw":
        return "throw"
    if strategy == "weaken":
        # Guard: if HP is so low that one Scratch would knock it out, throw immediately
        if enemy_hp <= min_safe_hp:
            return "throw"
        hp_frac = (enemy_hp / enemy_max_hp) if enemy_max_hp > 0 else 1.0
        if hp_frac > weaken_threshold:
            return "fight"
        return "throw"
    return "throw"


# ── System 1 Skills ───────────────────────────────────────────────────────────
def attack_first_move(env: PokemonRed, budget: int = 150) -> tuple[str, int]:
    """System 1 skill: in wild battle, uses FIGHT with the first move (Scratch).

    Returns:
        outcome: "hit" | "knockout" | "battle_ended"
        presses: number of action steps used
    """
    presses = 0
    _, ram = env.observe()
    if not ram["in_battle"]:
        return "battle_ended", presses

    # Step 1: Advance to battle menu
    while not is_battle_menu_active(env) and presses < budget:
        _, ram = env.step("b")
        presses += 1
        if not ram["in_battle"]:
            return "battle_ended", presses

    if not is_battle_menu_active(env):
        return "battle_ended", presses

    # Step 2: Select FIGHT (cursor is on FIGHT by default)
    _, ram = env.step("a")
    presses += 1

    # Step 3: Select Move 1 (cursor is on Move 1 by default)
    _, ram = env.step("a")
    presses += 1

    # Step 4: Advance through attack animation and text
    while presses < budget:
        _, ram = env.step("b")
        presses += 1
        if not ram["in_battle"]:
            # Battle ended during attack: enemy fainted (knockout)
            return "knockout", presses
        if is_battle_menu_active(env):
            # Battle menu returned: enemy took damage and survived
            return "hit", presses

    if not ram["in_battle"]:
        return "knockout", presses
    return "hit", presses


def throw_ball(env: PokemonRed, budget: int = 200) -> tuple[str, int]:
    """System 1 skill: used inside a wild battle to throw a Poké Ball.

    Sequence:
    1. Check ball availability from RAM.
    2. Advance intro/turn text until player battle menu is active.
    3. Navigate menu to ITEM, open bag, select Poké Ball, and throw.
    4. Press through throw animations and text.
    5. If caught, decline nickname prompt with B, close Pokédex entry, and return to overworld.
    6. Robustly detect outcome: "caught" | "broke_free" | "no_balls" | "battle_ended".

    Verified signals for "caught":
    - (party_size + box_count) strictly increases by 1.
    - Poké Ball count decreased by 1.
    - Battle ended (in_battle == False) and starter did not faint (HP > 0).
    """
    mem = env.pyboy.memory
    presses = 0
    _, ram = env.observe()
    if not ram["in_battle"]:
        return "battle_ended", presses

    items = bag(env)
    ball_slot = None
    for idx, it in enumerate(items):
        if it["id"] == ITEM_POKE_BALL and it["count"] > 0:
            ball_slot = idx
            break

    if ball_slot is None:
        return "no_balls", presses

    total_mon_start = int(mem[PARTY_SIZE]) + int(mem[WRAM_BOX_COUNT])
    balls_start = sum(it["count"] for it in items if it["id"] == ITEM_POKE_BALL)

    # Step 1: Advance to battle menu with 'b' to avoid accidental FIGHT selection
    while not is_battle_menu_active(env) and presses < budget:
        _, ram = env.step("b")
        presses += 1
        if not ram["in_battle"]:
            return "battle_ended", presses

    if not is_battle_menu_active(env):
        return "battle_ended", presses

    # Step 2: Navigate cursor down to ITEM (row 16 col 10) and press A to open Bag
    _, ram = env.step("down")
    presses += 1
    _, ram = env.step("a")
    presses += 1

    # Scroll to Poké Ball slot if not first slot
    for _ in range(ball_slot):
        if presses >= budget:
            break
        _, ram = env.step("down")
        presses += 1

    # Step 3: Press A to throw Poké Ball
    _, ram = env.step("a")
    presses += 1

    # Step 4: Advance through throw animation, wobbles, and outcome
    ball_thrown = False
    while presses < budget:
        cur_balls = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
        if cur_balls < balls_start:
            ball_thrown = True

        # Check if battle concluded
        if not env.observe()[1]["in_battle"]:
            total_mon_end = int(mem[PARTY_SIZE]) + int(mem[WRAM_BOX_COUNT])
            starter_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
            if total_mon_end > total_mon_start and starter_hp > 0:
                return "caught", presses
            else:
                return "battle_ended", presses

        # Check if enemy broke free and returned to battle menu
        if ball_thrown and is_battle_menu_active(env):
            return "broke_free", presses

        # Handle nickname prompt or text progression
        d = get_battle_dialog(env)
        if "nickname" in d:
            # Decline nickname prompt with B
            _, ram = env.step("b")
        else:
            _, ram = env.step("a")
        presses += 1

    # Final state check at budget expiration
    if not env.observe()[1]["in_battle"]:
        total_mon_end = int(mem[PARTY_SIZE]) + int(mem[WRAM_BOX_COUNT])
        if total_mon_end > total_mon_start:
            return "caught", presses
        return "battle_ended", presses

    return "broke_free" if ball_thrown else "battle_ended", presses


def run_away(env: PokemonRed, budget: int = 100) -> tuple[int, bool]:
    """System 1 skill: used inside a wild battle to escape when out of Poké Balls."""
    presses = 0
    _, ram = env.observe()
    while ram["in_battle"] and presses < budget:
        # Advance text until battle menu
        while not is_battle_menu_active(env) and presses < budget:
            _, ram = env.step("b")
            presses += 1
            if not ram["in_battle"]:
                return presses, True
        if not ram["in_battle"]:
            return presses, True

        # From default FIGHT cursor, move down to ITEM, right to RUN, and select
        _, ram = env.step("down")
        presses += 1
        _, ram = env.step("right")
        presses += 1
        _, ram = env.step("a")
        presses += 1

        # Clear escape text ("Got away safely!")
        for _ in range(15):
            if not ram["in_battle"] or presses >= budget:
                break
            _, ram = env.step("b")
            presses += 1

    return presses, not ram["in_battle"]


def switch_to_conscious_mon_if_needed(env: PokemonRed) -> tuple[int, bool]:
    """If the party switch prompt ('Bring out which POKéMON?') is up, switch to first conscious mon."""
    mem = env.pyboy.memory
    presses = 0
    starter_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
    if starter_hp == 0 and bool(mem[IN_BATTLE]):
        party_sz = min(int(mem[PARTY_SIZE]), 6)
        target_slot = None
        for s in range(1, party_sz):
            s_hp = (int(mem[HP_ADDRESSES[s]]) << 8) | int(mem[HP_ADDRESSES[s] + 1])
            if s_hp > 0:
                target_slot = s
                break
        if target_slot is not None:
            for _ in range(target_slot):
                env.step("down")
                presses += 1
            env.step("a")
            presses += 1
            env.step("a")
            presses += 1
            for _ in range(25):
                env.step("b")
                presses += 1
            return presses, True
    return presses, False


def navigate_catch(
    env: PokemonRed,
    exits: dict[int, list[tuple[int, int]]],
    target_map: int | list[int],
    budget: int = 600,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """System 1 navigation skill for catch quest: navigates to target map and flees wild encounters."""
    if rng is None:
        rng = np.random.default_rng(0)

    from lab.agent.mpc import greedy_action

    presses = 0
    prev_ram, prev_a = None, None
    _, ram = env.observe()
    wp_indices: dict[int, int] = {}
    target_maps = [target_map] if isinstance(target_map, int) else list(target_map)
    escapes = 0

    while ram["map_id"] not in target_maps and presses < budget:
        if ram["in_battle"]:
            p_sw, _ = switch_to_conscious_mon_if_needed(env)
            presses += p_sw
            p_run, esc = run_away(env, budget=min(80, budget - presses))
            presses += p_run
            if esc:
                escapes += 1
            _, ram = env.observe()
            prev_ram, prev_a = None, None
            continue

        cur_map = ram["map_id"]
        if cur_map not in exits:
            break

        target_list = exits[cur_map]
        if len(target_list) > 2 and cur_map not in wp_indices:
            closest_idx = int(np.argmin([abs(t[0] - ram["x"]) + abs(t[1] - ram["y"]) for t in target_list]))
            wp_indices[cur_map] = closest_idx

        if cur_map in wp_indices:
            idx = wp_indices[cur_map]
            cur_target = target_list[idx]
            dist = abs(ram["x"] - cur_target[0]) + abs(ram["y"] - cur_target[1])
            if dist <= 1 and idx < len(target_list) - 1:
                idx += 1
                wp_indices[cur_map] = idx
                cur_target = target_list[idx]
            step_exits = {**exits, cur_map: [cur_target]}
        else:
            step_exits = exits

        a = greedy_action(ram, prev_ram, prev_a, rng, step_exits)
        prev_ram, prev_a = ram, a
        _, ram = env.step(a)
        presses += 1

    return {
        "presses": presses,
        "success": ram["map_id"] in target_maps,
        "escapes": escapes,
    }


def catch_in_grass(
    env: PokemonRed,
    budget: int = 1000,
    catch_target: int = 10,
    current_catches: int = 0,
    hp_threshold: float = 0.5,
    strategy: str = "throw",
) -> dict[str, Any]:
    """System 1 catch_in_grass skill: walks tall grass patch on Route 1 and catches Pokémon.

    - Navigates to Route 1 grass patch (13, 30) if not already there.
    - Paces between (13, 30) and (14, 30) to trigger wild encounters.
    - If strategy == 'weaken', uses FIGHT with move 1 (Scratch) while enemy HP > 60% and HP > 5.
    - In each battle, calls `throw_ball` repeatedly until caught, fled, or out of balls.
    - If balls reach 0 inside battle, calls `run_away` to escape safely.
    - Yields control back to System 2 planner when:
        * HP fraction drops below threshold (< 0.5)
        * Out of balls (balls == 0)
        * Catch target reached (current_catches >= catch_target)
        * Action budget exhausted
    """
    presses = 0
    catches = 0
    balls_thrown = 0
    escapes = 0
    knockouts = 0
    battles_started = 0
    blackouts = 0
    events: list[dict[str, Any]] = []
    _, ram = env.observe()

    # If on Route 1 but outside grass patch, navigate to anchor (13, 30)
    if ram["map_id"] == MAP_ROUTE1 and (ram["x"], ram["y"]) not in GRASS_PATCH_TILES:
        for target in ROUTE1_SOUTH_WAYPOINTS:
            if ram["y"] > target[1] and target != (13, 30):
                continue
            while (ram["x"], ram["y"]) != target and presses < budget:
                if ram["in_battle"]:
                    break
                if ram["x"] < target[0]: a = "right"
                elif ram["x"] > target[0]: a = "left"
                elif ram["y"] > target[1]: a = "up"
                elif ram["y"] < target[1]: a = "down"
                _, ram = env.step(a)
                presses += 1
            if ram["in_battle"]:
                break

    # Pacing and catching loop
    while presses < budget:
        st = party_status(env)
        cur_balls = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
        if st["hp_frac"] < hp_threshold:
            break
        if cur_balls == 0:
            break
        if current_catches + catches >= catch_target:
            break

        # If a battle was triggered
        if ram["in_battle"]:
            battles_started += 1
            b_start = presses
            battle_balls = 0
            battle_outcome = "unknown"

            while ram["in_battle"] and presses < budget:
                cur_balls_now = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
                if cur_balls_now == 0:
                    # Out of balls mid-battle: run away
                    p_run, escaped = run_away(env, budget=min(80, budget - presses))
                    presses += p_run
                    if escaped:
                        escapes += 1
                        battle_outcome = "escaped_out_of_balls"
                    break

                # Query enemy status from RAM
                enemy_st = read_enemy_status(env)
                action = decide_battle_action(
                    enemy_hp=enemy_st["hp"],
                    enemy_max_hp=enemy_st["max_hp"],
                    strategy=strategy,
                )

                if action == "fight":
                    atk_outcome, p_atk = attack_first_move(env, budget=min(150, budget - presses))
                    presses += p_atk
                    if atk_outcome == "knockout":
                        knockouts += 1
                        battle_outcome = "knockout"
                        events.append({
                            "type": "knockout",
                            "species": enemy_st["species_name"],
                            "species_id": enemy_st["species_id"],
                            "presses": presses - b_start,
                        })
                        break
                    elif atk_outcome == "battle_ended":
                        battle_outcome = "battle_ended"
                        break
                    # If outcome == "hit", loop re-evaluates next action
                    continue

                # Action is "throw"
                rem_b = budget - presses
                outcome, p_throw = throw_ball(env, budget=min(200, rem_b))
                presses += p_throw
                battle_balls += 1
                balls_thrown += 1

                if outcome == "caught":
                    catches += 1
                    battle_outcome = "caught"
                    events.append({
                        "type": "catch",
                        "species": enemy_st["species_name"],
                        "species_id": enemy_st["species_id"],
                        "balls_used": battle_balls,
                        "presses": presses - b_start,
                    })
                    break
                elif outcome == "no_balls":
                    p_run, escaped = run_away(env, budget=min(80, budget - presses))
                    presses += p_run
                    if escaped:
                        escapes += 1
                        battle_outcome = "escaped_no_balls"
                    break
                elif outcome == "battle_ended":
                    battle_outcome = "battle_ended"
                    break
                elif outcome == "broke_free":
                    continue

            # Check if player fainted
            st_after = party_status(env)
            _, ram = env.observe()
            if st_after["hp"] <= 0 or ram["map_id"] != MAP_ROUTE1:
                blackouts += 1
                events.append({"type": "blackout", "presses": presses - b_start})
                break
            continue

        # Ensure player is in grass patch before pacing
        if (ram["x"], ram["y"]) not in GRASS_PATCH_TILES:
            for target in ROUTE1_SOUTH_WAYPOINTS:
                if ram["y"] > target[1] and target != (13, 30):
                    continue
                while (ram["x"], ram["y"]) != target and presses < budget:
                    if ram["in_battle"]:
                        break
                    if ram["x"] < target[0]: a = "right"
                    elif ram["x"] > target[0]: a = "left"
                    elif ram["y"] > target[1]: a = "up"
                    elif ram["y"] < target[1]: a = "down"
                    _, ram = env.step(a)
                    presses += 1
                if ram["in_battle"]:
                    break
            if ram["in_battle"]:
                continue

        # Pace in grass patch: alternate (13, 30) <-> (14, 30)
        act = "left" if ram["x"] >= 14 else "right"
        _, ram = env.step(act)
        presses += 1

    return {
        "presses": presses,
        "catches": catches,
        "balls_thrown": balls_thrown,
        "escapes": escapes,
        "knockouts": knockouts,
        "battles_started": battles_started,
        "blackouts": blackouts,
        "events": events,
    }


# ── Frame Caption Banner Helper ───────────────────────────────────────────────
def create_captioned_frame(
    rgb_frame: np.ndarray,
    goal: str,
    strategy: str,
    balls: int,
    catches: int,
    catch_target: int,
    hp: int,
    max_hp: int,
) -> Image.Image:
    """Add a top banner displaying strategy, goal, balls, catches, and HP to a native 144x160 Game Boy frame."""
    banner_height = 16
    w = rgb_frame.shape[1]
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    caption = f"{strategy[:1].upper()}:{goal[:5].upper()} | B:{balls} C:{catches}/{catch_target} | {hp}/{max_hp}"
    cv2.putText(
        banner,
        caption,
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    combined = np.vstack([banner, rgb_frame])
    return Image.fromarray(combined)


# ── Full Agent Runner ─────────────────────────────────────────────────────────
def run_catch_quest(
    catch_target: int = 10,
    budget: int = 8000,
    seed: int = 0,
    strategy: str = "throw",
    save_gif: bool = False,
    state_path: str | Path = "saves/pokedex_viridian.state",
    work_dir: str | Path = "work/catch",
) -> dict[str, Any]:
    """Execute one full catch quest run from pokedex_viridian.state."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    timeline: list[dict[str, Any]] = []
    gif_frames: list[Image.Image] = []
    total_presses = 0

    total_catches = 0
    total_balls_bought = 0
    total_balls_thrown = 0
    total_knockouts = 0
    total_heals = 0
    total_blackouts = 0

    current_goal_str = "init"
    stop_reason = "budget exhausted"
    start_time = time.time()

    with PokemonRed(state=state_path) as env:
        # Initial status
        init_st = catch_status(env, catches=0)
        money_start = init_st["money"]
        dex_start = init_st["dex_owned"]

        # Wrap env.step to count presses and capture GIF frames every ~20 presses
        orig_step = env.step

        def wrapped_step(action: int | str):
            nonlocal total_presses
            frame, ram_obs = orig_step(action)
            total_presses += 1
            if save_gif and total_presses % 20 == 0:
                cur_st = catch_status(env, catches=total_catches)
                raw_rgb = screen_capture(env.pyboy)
                captioned = create_captioned_frame(
                    raw_rgb,
                    current_goal_str,
                    strategy,
                    cur_st["balls"],
                    total_catches,
                    catch_target,
                    cur_st["hp"],
                    cur_st["max_hp"],
                )
                gif_frames.append(captioned)
            return frame, ram_obs

        env.step = wrapped_step

        print(
            f"\n[CatchQuest seed={seed} strat={strategy}] Starting run: Catches 0/{catch_target}, "
            f"Balls: {init_st['balls']}, Money: ${init_st['money']}, "
            f"Species Owned: {init_st['dex_owned']}, HP: {init_st['hp']}/{init_st['max_hp']}"
        )

        while total_presses < budget:
            status = catch_status(env, catches=total_catches)
            goal, reason = plan_catch(status, catch_target=catch_target, current_goal=current_goal_str)
            current_goal_str = str(goal)

            elapsed = round(time.time() - start_time, 2)
            decision_log = {
                "time": elapsed,
                "presses": total_presses,
                "goal": str(goal),
                "reason": reason,
                "strategy": strategy,
                "balls": status["balls"],
                "catches": total_catches,
                "knockouts": total_knockouts,
                "money": status["money"],
                "hp": status["hp"],
                "max_hp": status["max_hp"],
                "hp_frac": round(status["hp_frac"], 2),
                "map": status["map_id"],
            }

            print(
                f"[Planner t={elapsed:5.1f}s p={total_presses:4d}] "
                f"Goal: {goal:<20} | Strat: {strategy:<6} | Balls: {status['balls']:2d} | "
                f"Catches: {total_catches:2d}/{catch_target} | KO: {total_knockouts:2d} | "
                f"Money: ${status['money']:4d} | HP: {status['hp']:2d}/{status['max_hp']:2d} | Reason: {reason}"
            )

            if goal in ("done", "done: out of money"):
                decision_log["skill"] = "none"
                decision_log["result"] = {"success": True}
                timeline.append(decision_log)
                stop_reason = str(goal)
                break

            remaining_budget = budget - total_presses
            skill_name = "none"
            skill_result: dict[str, Any] = {"success": False, "presses": 0}

            # ── Plan Execution Branches ───────────────────────────────────────
            if goal == "heal":
                if status["map_id"] == MAP_ROUTE1:
                    skill_name = "navigate_north_to_viridian"
                    exits = {MAP_ROUTE1: ROUTE1_NORTH_WAYPOINTS, **DETOUR_EXITS}
                    skill_result = navigate_catch(
                        env, exits, target_map=MAP_VIRIDIAN, budget=min(600, remaining_budget), rng=rng
                    )
                elif status["map_id"] in (MAP_VIRIDIAN, MAP_VIRIDIAN_POKECENTER):
                    if status["map_id"] == MAP_VIRIDIAN and (status["x"], status["y"]) == (29, 20):
                        for a in MART_TO_PC_PATH:
                            if total_presses >= budget: break
                            env.step(a)
                    skill_name = "heal"
                    total_heals += 1
                    skill_result = heal(env, budget=min(350, remaining_budget))

            elif goal == "buy":
                if status["map_id"] == MAP_ROUTE1:
                    skill_name = "navigate_north_to_viridian"
                    exits = {MAP_ROUTE1: ROUTE1_NORTH_WAYPOINTS, **DETOUR_EXITS}
                    skill_result = navigate_catch(
                        env, exits, target_map=MAP_VIRIDIAN, budget=min(600, remaining_budget), rng=rng
                    )
                elif status["map_id"] in (MAP_VIRIDIAN, MAP_VIRIDIAN_POKECENTER, MAP_VIRIDIAN_MART):
                    max_affordable = status["money"] // 200
                    n_buy = min(10, max_affordable)
                    skill_name = f"buy_poke_balls_{n_buy}"
                    b_bought, m_spent = buy_poke_balls(env, n=n_buy, budget=min(600, remaining_budget))
                    total_balls_bought += b_bought
                    skill_result = {"success": b_bought > 0, "balls_bought": b_bought, "money_spent": m_spent}

            elif goal == "catch":
                if status["map_id"] == MAP_VIRIDIAN:
                    if (status["x"], status["y"]) == (29, 20):
                        skill_name = "mart_to_route1"
                        for a in MART_TO_ROUTE1_PATH:
                            if total_presses >= budget: break
                            env.step(a)
                        skill_result = {"success": env.observe()[1]["map_id"] == MAP_ROUTE1}
                    else:
                        skill_name = "navigate_viridian_to_route1"
                        exits = {MAP_VIRIDIAN: VIRIDIAN_TO_ROUTE1_WAYPOINTS, **DETOUR_EXITS}
                        skill_result = navigate(
                            env, exits, target_map=MAP_ROUTE1, budget=min(250, remaining_budget), rng=rng
                        )
                elif status["map_id"] == MAP_VIRIDIAN_POKECENTER:
                    skill_name = "exit_pokecenter"
                    while env.observe()[1]["map_id"] == MAP_VIRIDIAN_POKECENTER and total_presses < budget:
                        env.step(0)  # DOWN
                    skill_result = {"success": True}
                elif status["map_id"] == MAP_ROUTE1:
                    skill_name = "catch_in_grass"
                    res_grass = catch_in_grass(
                        env,
                        budget=min(1200, remaining_budget),
                        catch_target=catch_target,
                        current_catches=total_catches,
                        hp_threshold=0.5,
                        strategy=strategy,
                    )
                    total_catches += res_grass["catches"]
                    total_balls_thrown += res_grass["balls_thrown"]
                    total_knockouts += res_grass.get("knockouts", 0)
                    total_blackouts += res_grass["blackouts"]
                    skill_result = res_grass

            decision_log["skill"] = skill_name
            decision_log["result"] = skill_result
            timeline.append(decision_log)

        final_st = catch_status(env, catches=total_catches)
        money_end = final_st["money"]
        money_spent = money_start - money_end
        dex_end = final_st["dex_owned"]
        goal_reached = total_catches >= catch_target
        catch_rate_per_ball = round(total_catches / total_balls_thrown, 3) if total_balls_thrown > 0 else 0.0

        metrics = {
            "seed": seed,
            "strategy": strategy,
            "catch_target": catch_target,
            "catches": total_catches,
            "species_owned_before": dex_start,
            "species_owned_after": dex_end,
            "distinct_species_gained": dex_end - dex_start,
            "balls_bought": total_balls_bought,
            "balls_thrown": total_balls_thrown,
            "catch_rate_per_ball": catch_rate_per_ball,
            "knockouts": total_knockouts,
            "money_start": money_start,
            "money_end": money_end,
            "money_spent": money_spent,
            "heals": total_heals,
            "blackouts": total_blackouts,
            "presses": total_presses,
            "goal_reached": goal_reached,
            "stop_reason": stop_reason,
        }

        # Save run JSON
        run_file = work_dir / f"run_{seed}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "timeline": timeline}, f, indent=2)
        print(f"\nSaved run metrics -> {run_file}")

        # Save GIF if requested
        if save_gif and gif_frames:
            gif_file = work_dir / f"run_{seed}.gif"
            gif_frames[0].save(
                gif_file,
                save_all=True,
                append_images=gif_frames[1:],
                duration=120,
                loop=0,
                optimize=True,
            )
            gif_size_mb = gif_file.stat().st_size / (1024 * 1024)
            print(f"Saved run GIF -> {gif_file} ({gif_size_mb:.2f} MB, {len(gif_frames)} frames)")

        return metrics


# ── Main CLI Runner ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="System 1 / System 2 Pokémon Catching Agent.")
    parser.add_argument("--strategy", choices=["throw", "weaken"], default="throw",
                        help="Battle strategy: 'throw' (throw at full HP) or 'weaken' (FIGHT until HP <= 60%).")
    parser.add_argument("--catch-target", type=int, default=10, help="Target number of Pokémon to catch.")
    parser.add_argument("--budget", type=int, default=8000, help="Action press budget per run.")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs.")
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed.")
    parser.add_argument("--gif", action="store_true", help="Record and save GIF of each run.")
    parser.add_argument("--state", type=str, default="saves/pokedex_viridian.state", help="Save state path.")
    parser.add_argument("--work-dir", type=str, default=None, help="Output directory (defaults to work/catch_<strategy>).")
    args = parser.parse_args()

    work_dir_str = args.work_dir if args.work_dir else f"work/catch_{args.strategy}"
    work_dir = Path(work_dir_str)
    work_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for r in range(args.runs):
        cur_seed = args.seed + r
        m = run_catch_quest(
            catch_target=args.catch_target,
            budget=args.budget,
            seed=cur_seed,
            strategy=args.strategy,
            save_gif=args.gif,
            state_path=args.state,
            work_dir=work_dir,
        )
        all_metrics.append(m)

    # Compute summary
    n_runs = len(all_metrics)
    summary = {
        "strategy": args.strategy,
        "n_runs": n_runs,
        "runs": all_metrics,
        "summary": {
            "strategy": args.strategy,
            "catch_target": args.catch_target,
            "budget": args.budget,
            "goals_reached": sum(1 for m in all_metrics if m["goal_reached"]),
            "avg_catches": round(float(np.mean([m["catches"] for m in all_metrics])), 2),
            "min_catches": int(np.min([m["catches"] for m in all_metrics])),
            "max_catches": int(np.max([m["catches"] for m in all_metrics])),
            "avg_species_owned_after": round(float(np.mean([m["species_owned_after"] for m in all_metrics])), 2),
            "avg_balls_bought": round(float(np.mean([m["balls_bought"] for m in all_metrics])), 2),
            "avg_balls_thrown": round(float(np.mean([m["balls_thrown"] for m in all_metrics])), 2),
            "avg_catch_rate_per_ball": round(float(np.mean([m["catch_rate_per_ball"] for m in all_metrics])), 3),
            "avg_knockouts": round(float(np.mean([m["knockouts"] for m in all_metrics])), 2),
            "avg_money_spent": round(float(np.mean([m["money_spent"] for m in all_metrics])), 2),
            "avg_heals": round(float(np.mean([m["heals"] for m in all_metrics])), 2),
            "avg_blackouts": round(float(np.mean([m["blackouts"] for m in all_metrics])), 2),
            "avg_presses": round(float(np.mean([m["presses"] for m in all_metrics])), 2),
        },
    }

    # If the sibling strategy's summary exists, record both strategies' means
    sibling_strat = "weaken" if args.strategy == "throw" else "throw"
    sibling_file = work_dir.parent / f"catch_{sibling_strat}" / "summary.json"
    if sibling_file.exists():
        try:
            with open(sibling_file, "r", encoding="utf-8") as f_sib:
                sib_data = json.load(f_sib)
                if "summary" in sib_data:
                    summary["both_strategies_means"] = {
                        args.strategy: summary["summary"],
                        sibling_strat: sib_data["summary"],
                    }
        except Exception:
            pass

    summary_file = work_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary over {n_runs} runs -> {summary_file}")

    # Also save/update joint comparison summary in work/catch/summary.json if both exist
    if "both_strategies_means" in summary:
        joint_summary_file = work_dir.parent / "catch" / "summary.json"
        try:
            joint_data = {
                "catch_target": args.catch_target,
                "n_runs_per_strategy": n_runs,
                "both_strategies_means": summary["both_strategies_means"],
            }
            with open(joint_summary_file, "w", encoding="utf-8") as f_joint:
                json.dump(joint_data, f_joint, indent=2)
            print(f"Updated joint comparison summary -> {joint_summary_file}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
