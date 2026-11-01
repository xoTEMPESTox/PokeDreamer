"""System 1 / System 2 loop for leveling up and healing in real Pokémon Red.

System 2 v0:
    Rule-based planner reading Game Boy RAM to choose high-level goals:
    - If hp_frac < 0.35 -> "heal"
    - Elif level < target_level -> "grind"
    - Else -> "done"
    Maps each goal to a sequence of System 1 skills based on current map and state.
    Re-plans after every skill execution.

System 1 skills v0:
    Reflexive and greedy skills with bounded press budgets:
    - navigate: greedy navigation toward exit tiles using privileged RAM, with battle reflex.
    - grind: pacing in tall grass on Route 1, fighting wild battles with battle reflex.
    - heal: enters Viridian Pokémon Center, talks to Nurse Joy, clears dialogue, and exits.
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

from lab.agent.mpc import DIRECTIONS, greedy_action
from lab.agent.tasks import _PALLET_DETOURS
from lab.env import ROM_PATH, PokemonRed
from src.game_state import screen_capture
from src.ram_addresses import (
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

# ── Map IDs ───────────────────────────────────────────────────────────────────
MAP_PALLET = 0
MAP_VIRIDIAN = 1
MAP_ROUTE1 = 12
MAP_REDS_HOUSE_1F = 37
MAP_REDS_HOUSE_2F = 38
MAP_RIVALS_HOUSE = 39
MAP_OAKS_LAB = 40
MAP_VIRIDIAN_POKECENTER = 41
MAP_VIRIDIAN_MART = 42

# ── Detour building exits (known exits to overworld) ──────────────────────────
DETOUR_EXITS = {
    **_PALLET_DETOURS,
    MAP_VIRIDIAN_POKECENTER: [(3, 7), (4, 7)],
    MAP_VIRIDIAN_MART: [(3, 7), (4, 7)],
}

# ── Waypoints on Route 1 ───────────────────────────────────────────────────────
# Route 1 has one-way ledges going south. To navigate NORTH from Pallet to Viridian,
# the player must follow a winding corridor around the ledges.
# Discovered by BFS on Route 1 collision/passable tiles:
ROUTE1_NORTH_WAYPOINTS = [
    (10, 29),  # From entrance, walk north to corner
    (7, 29),   # Turn west around southern ledge
    (7, 24),   # Walk north past west side of ledge
    (12, 24),  # Cut east into middle lane
    (12, 20),  # North through middle lane
    (9, 20),   # Cut west around middle ledge
    (9, 14),   # North past middle ledge
    (15, 14),  # Cut east to eastern bypass lane
    (15, 2),   # North along eastern lane past upper ledges
    (11, 2),   # Cut west towards Viridian gate
    (11, 0),   # Viridian City entrance gate
]

# Going SOUTH on Route 1 from Viridian City to the grinding patch:
# Ledges can be jumped downwards, so south route is fast and direct:
ROUTE1_SOUTH_WAYPOINTS = [
    (11, 3),
    (15, 3),
    (15, 14),
    (13, 18),
    (13, 30),  # Grinding patch anchor
]

# ── Viridian City Waypoints ───────────────────────────────────────────────────
# Viridian entrance from Route 1 is at (21, 35). The Pokémon Center door is at (23, 26).
# A small hedge blocks direct walking at (21, 28) / (23, 28), requiring stepping west to x=19.
VIRIDIAN_TO_PC_WAYPOINTS = [
    (21, 30),
    (19, 30),
    (19, 26),
    (23, 26),  # Directly in front of Pokémon Center door
]

# Southbound from Pokémon Center door (23, 26) to Route 1 border (21, 35):
VIRIDIAN_TO_ROUTE1_WAYPOINTS = [
    (21, 26),
    (21, 35),
    (21, 36),  # Warp tile to Route 1
]

# ── Route 1 Tall Grass Grinding Patch ─────────────────────────────────────────
# Explored coordinates: (13, 30) and (14, 30).
# Discovered by systematic coordinate exploration on Route 1: this patch reliably triggers
# wild battles (Rattata, Pidgey) and pacing between (13, 30) and (14, 30) has zero obstacle collisions.
GRASS_PATCH_TILES = [(13, 30), (14, 30)]

# ── Viridian Pokémon Center Nurse Coordinates ─────────────────────────────────
# Standing tile: (3, 3) facing UP (facing=4).
# Nurse Joy is behind the counter at (3, 2).
NURSE_STAND_TILE = (3, 3)


# ── RAM Helpers ───────────────────────────────────────────────────────────────
def party_status(env: PokemonRed) -> dict[str, Any]:
    """Read starter Pokémon party stats and overworld status from RAM."""
    _, ram = env.observe()
    mem = env.pyboy.memory
    party_size = min(int(mem[PARTY_SIZE]), 6)
    if party_size > 0:
        hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
        max_hp = (int(mem[MAX_HP_ADDRESSES[0]]) << 8) | int(mem[MAX_HP_ADDRESSES[0] + 1])
        level = int(mem[LEVEL_ADDRESSES[0]])
    else:
        hp, max_hp, level = 0, 0, 0

    hp_frac = (hp / max_hp) if max_hp > 0 else 0.0
    return {
        "level": level,
        "hp": hp,
        "max_hp": max_hp,
        "hp_frac": hp_frac,
        "party_size": party_size,
        "map_id": int(ram["map_id"]),
        "map": int(ram["map_id"]),
        "x": int(ram["x"]),
        "y": int(ram["y"]),
        "facing": int(ram.get("facing", 0)),
        "in_battle": bool(ram["in_battle"]),
    }


# ── System 2 v0 Planner ───────────────────────────────────────────────────────
class Goal(str):
    """String subclass that allows unpacking (goal, reason) or direct string equality."""

    def __new__(cls, goal: str, reason: str = ""):
        obj = super().__new__(cls, goal)
        obj.reason = reason
        return obj

    def __iter__(self):
        yield str(self)
        yield self.reason


def plan(
    status: dict[str, Any],
    target_level: int = 10,
    current_goal: Optional[str] = None,
    heal_threshold: float = 0.35,
) -> Goal:
    """System 2 v0 rule-based planner.

    Rules:
    - if hp_frac < 0.35 -> "heal"
    - elif level < target_level -> "grind"
    - else -> "done"

    If a heal goal is actively in progress in Viridian / Pokémon Center and HP is not yet full,
    continues healing to prevent prematurely aborting on a mid-trip level up.
    """
    hp_frac = status["hp_frac"]
    level = status["level"]
    hp = status["hp"]
    max_hp = status["max_hp"]
    map_id = status.get("map_id", 0)

    # If currently executing heal in Viridian or PC and not yet at max HP, stay on heal
    if current_goal == "heal" and hp_frac < 1.0 and map_id in (MAP_VIRIDIAN, MAP_VIRIDIAN_POKECENTER):
        return Goal("heal", f"Continuing heal in Viridian: HP {hp}/{max_hp}")

    if hp_frac < heal_threshold:
        return Goal("heal", f"HP low: {hp}/{max_hp} ({hp_frac:.2f} < {heal_threshold})")
    elif level < target_level:
        return Goal("grind", f"Level {level} < target {target_level}")
    else:
        return Goal("done", f"Target level {target_level} reached (current: {level})")


# ── System 1 Skills v0 ────────────────────────────────────────────────────────
def navigate(
    env: PokemonRed,
    exits: dict[int, list[Tuple[int, int]]],
    target_map: int | list[int] | tuple[int, ...],
    budget: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """System 1 navigate skill: greedy navigation toward exit tiles with battle reflex.

    Reuses greedy_action from lab.agent.mpc. If exits[cur_map] has sequential waypoints,
    tracks and advances through waypoints to bypass obstacles.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    presses = 0
    prev_ram, prev_a = None, None
    _, ram = env.observe()
    events: list[dict[str, Any]] = []
    wp_indices: dict[int, int] = {}
    target_maps = [target_map] if isinstance(target_map, int) else list(target_map)

    battles_started = 0
    battles_won = 0
    blackouts = 0

    while ram["map_id"] not in target_maps and presses < budget:
        if ram["in_battle"]:
            # Battle reflex: press A (selects FIGHT, first move; clears text)
            b_start = presses
            battles_started += 1
            battle_map = ram["map_id"]
            while ram["in_battle"] and presses < budget:
                _, ram = env.step(4)  # A
                presses += 1

            st_after = party_status(env)
            if ram["map_id"] == battle_map and st_after["hp"] > 0:
                battles_won += 1
                events.append({
                    "type": "battle_won",
                    "presses": presses - b_start,
                    "level": st_after["level"],
                    "hp": st_after["hp"],
                    "map": ram["map_id"],
                })
            else:
                blackouts += 1
                events.append({
                    "type": "blackout",
                    "presses": presses - b_start,
                    "level": st_after["level"],
                    "hp": st_after["hp"],
                    "map": ram["map_id"],
                })
                break  # Blackout warped player to safety; break out of navigate

            prev_ram, prev_a = None, None
            continue

        cur_map = ram["map_id"]
        if cur_map not in exits:
            break

        target_list = exits[cur_map]
        # Sequential waypoints handling if list has > 2 coordinates:
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
        prev_ram = ram
        prev_a = a
        _, ram = env.step(a)
        presses += 1

    success = (ram["map_id"] in target_maps)
    return {
        "presses": presses,
        "success": success,
        "events": events,
        "battles_started": battles_started,
        "battles_won": battles_won,
        "blackouts": blackouts,
    }


def grind(
    env: PokemonRed,
    budget: int = 500,
    target_level: int = 10,
    rng: Optional[np.random.Generator] = None,
    hp_threshold: float = 0.35,
) -> dict[str, Any]:
    """System 1 grind skill: walks back and forth in tall grass patch on Route 1.

    Triggers wild battles and uses battle reflex (A presses).
    Tracks battles started, battles won, and blackouts.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    presses = 0
    events: list[dict[str, Any]] = []
    battles_started = 0
    battles_won = 0
    blackouts = 0
    _, ram = env.observe()

    # If on Route 1 but outside the grass patch, walk to the anchor (13, 30)
    if ram["map_id"] == MAP_ROUTE1 and (ram["x"], ram["y"]) not in GRASS_PATCH_TILES:
        if ram["y"] <= 5:  # North entrance from Viridian
            for target in ROUTE1_SOUTH_WAYPOINTS:
                while (ram["x"], ram["y"]) != target and presses < budget:
                    if ram["in_battle"]:
                        b_start = presses
                        battles_started += 1
                        while ram["in_battle"] and presses < budget:
                            _, ram = env.step(4)  # A
                            presses += 1
                        st = party_status(env)
                        if ram["map_id"] == MAP_ROUTE1 and st["hp"] > 0:
                            battles_won += 1
                            events.append({"type": "battle_won", "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                        else:
                            blackouts += 1
                            events.append({"type": "blackout", "map": ram["map_id"], "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                            return {
                                "presses": presses,
                                "success": False,
                                "events": events,
                                "battles_started": battles_started,
                                "battles_won": battles_won,
                                "blackouts": blackouts,
                            }
                        continue
                    if ram["x"] < target[0]: a = 2
                    elif ram["x"] > target[0]: a = 1
                    elif ram["y"] > target[1]: a = 3
                    elif ram["y"] < target[1]: a = 0
                    _, ram = env.step(a)
                    presses += 1
        elif ram["y"] >= 32:  # South entrance from Pallet Town
            while ram["y"] > 30 and presses < budget:
                if ram["in_battle"]:
                    b_start = presses
                    battles_started += 1
                    while ram["in_battle"] and presses < budget:
                        _, ram = env.step(4)
                        presses += 1
                    st = party_status(env)
                    if ram["map_id"] == MAP_ROUTE1 and st["hp"] > 0:
                        battles_won += 1
                        events.append({"type": "battle_won", "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                    else:
                        blackouts += 1
                        events.append({"type": "blackout", "map": ram["map_id"], "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                        return {
                            "presses": presses,
                            "success": False,
                            "events": events,
                            "battles_started": battles_started,
                            "battles_won": battles_won,
                            "blackouts": blackouts,
                        }
                    continue
                _, ram = env.step(3)  # UP
                presses += 1
            while ram["x"] < 13 and presses < budget:
                if ram["in_battle"]:
                    b_start = presses
                    battles_started += 1
                    while ram["in_battle"] and presses < budget:
                        _, ram = env.step(4)
                        presses += 1
                    st = party_status(env)
                    if ram["map_id"] == MAP_ROUTE1 and st["hp"] > 0:
                        battles_won += 1
                        events.append({"type": "battle_won", "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                    else:
                        blackouts += 1
                        events.append({"type": "blackout", "map": ram["map_id"], "level": st["level"], "hp": st["hp"], "presses": presses - b_start})
                        return {
                            "presses": presses,
                            "success": False,
                            "events": events,
                            "battles_started": battles_started,
                            "battles_won": battles_won,
                            "blackouts": blackouts,
                        }
                    continue
                _, ram = env.step(2)  # RIGHT
                presses += 1

    # Pace back and forth inside the tall-grass patch: (13, 30) <-> (14, 30)
    while presses < budget:
        st = party_status(env)
        if st["hp_frac"] < hp_threshold:
            break
        if st["level"] >= target_level:
            break

        if ram["in_battle"]:
            battles_started += 1
            b_start = presses
            while ram["in_battle"] and presses < budget:
                _, ram = env.step(4)  # A
                presses += 1

            st_after = party_status(env)
            if ram["map_id"] == MAP_ROUTE1 and st_after["hp"] > 0:
                battles_won += 1
                events.append({
                    "type": "battle_won",
                    "level": st_after["level"],
                    "hp": st_after["hp"],
                    "presses": presses - b_start,
                })
            else:
                blackouts += 1
                events.append({
                    "type": "blackout",
                    "map": ram["map_id"],
                    "level": st_after["level"],
                    "hp": st_after["hp"],
                    "presses": presses - b_start,
                })
                break
            continue

        # Alternate walking left and right between (13, 30) and (14, 30)
        act = 1 if ram["x"] >= 14 else 2  # 1 = left, 2 = right
        _, ram = env.step(act)
        presses += 1

    return {
        "presses": presses,
        "success": (battles_won > 0),
        "events": events,
        "battles_started": battles_started,
        "battles_won": battles_won,
        "blackouts": blackouts,
    }


def heal(env: PokemonRed, budget: int = 300) -> dict[str, Any]:
    """System 1 heal skill: from Viridian City, enter Pokémon Center, heal at nurse, and leave.

    Success = HP == max_hp and player returns to Viridian City (map 1).
    Nurse Joy position: counter at (3, 3) facing UP (nurse at (3, 2)).
    """
    presses = 0
    events: list[dict[str, Any]] = []
    _, ram = env.observe()

    # Step 1: If in Viridian City (1), navigate to Pokémon Center door (23, 26)
    if ram["map_id"] == MAP_VIRIDIAN:
        for target in VIRIDIAN_TO_PC_WAYPOINTS:
            while (ram["x"], ram["y"]) != target and presses < budget:
                if ram["x"] < target[0]: a = 2
                elif ram["x"] > target[0]: a = 1
                elif ram["y"] > target[1]: a = 3
                elif ram["y"] < target[1]: a = 0
                _, ram = env.step(a)
                presses += 1

        if (ram["x"], ram["y"]) == (23, 26) and presses < budget:
            _, ram = env.step(3)  # UP into Pokémon Center
            presses += 1

    # Wait for Pokémon Center map 41 to load
    while ram["map_id"] != MAP_VIRIDIAN_POKECENTER and presses < budget:
        _, ram = env.step(7)  # pass
        presses += 1

    if ram["map_id"] != MAP_VIRIDIAN_POKECENTER:
        return {"presses": presses, "success": False, "events": events}

    # Step 2: Inside Pokémon Center, walk to nurse counter at (3, 3)
    while (ram["x"], ram["y"]) != NURSE_STAND_TILE and presses < budget:
        if ram["x"] < NURSE_STAND_TILE[0]: a = 2
        elif ram["x"] > NURSE_STAND_TILE[0]: a = 1
        elif ram["y"] > NURSE_STAND_TILE[1]: a = 3
        elif ram["y"] < NURSE_STAND_TILE[1]: a = 0
        _, ram = env.step(a)
        presses += 1

    # Face UP towards Nurse Joy
    while ram["facing"] != 4 and presses < budget:
        _, ram = env.step(3)  # UP
        presses += 1

    # Step 3: Talk to nurse (press A through dialog, automatically selects YES)
    mem = env.pyboy.memory
    max_hp = (int(mem[MAX_HP_ADDRESSES[0]]) << 8) | int(mem[MAX_HP_ADDRESSES[0] + 1])
    healed = False
    while presses < budget:
        cur_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
        if cur_hp == max_hp:
            healed = True
            break
        _, ram = env.step(4)  # A
        presses += 1

    events.append({"type": "nurse_heal", "healed": healed, "hp": cur_hp, "max_hp": max_hp})

    # Step 4: Clear post-heal dialogue ("We hope to see you again!")
    # Step down; if blocked by dialog, press B, until player steps down to (3, 4)
    while (ram["x"], ram["y"]) == NURSE_STAND_TILE and presses < budget:
        _, ram = env.step(0)  # DOWN
        presses += 1
        if (ram["x"], ram["y"]) != NURSE_STAND_TILE:
            break
        _, ram = env.step(5)  # B
        presses += 1

    # Step 5: Walk south to exit tiles (3, 7)/(4, 7) and step DOWN to Viridian City
    while ram["map_id"] == MAP_VIRIDIAN_POKECENTER and presses < budget:
        _, ram = env.step(0)  # DOWN
        presses += 1

    # Allow map transition fade to finish
    for _ in range(3):
        if presses < budget:
            _, ram = env.step(7)  # pass
            presses += 1

    mem = env.pyboy.memory
    final_hp = (int(mem[HP_ADDRESSES[0]]) << 8) | int(mem[HP_ADDRESSES[0] + 1])
    success = (final_hp == max_hp) and (ram["map_id"] == MAP_VIRIDIAN)
    return {
        "presses": presses,
        "success": success,
        "events": events,
    }


# ── GIF Caption Helper ────────────────────────────────────────────────────────
def create_captioned_frame(
    rgb_frame: np.ndarray, goal: str, level: int, hp: int, max_hp: int
) -> Image.Image:
    """Add a small top banner displaying goal, level, and HP to a native 144x160 Game Boy frame."""
    banner_height = 16
    w = rgb_frame.shape[1]
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    caption = f"{goal.upper()} | Lv{level} | HP {hp}/{max_hp}"
    cv2.putText(
        banner,
        caption,
        (4, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    combined = np.vstack([banner, rgb_frame])
    return Image.fromarray(combined)


# ── Full Agent Runner ─────────────────────────────────────────────────────────
def run_agent(
    target_level: int = 10,
    budget: int = 4000,
    seed: int = 0,
    save_gif: bool = False,
    state_path: str | Path = "saves/starter_pallet.state",
    work_dir: str | Path = "work/system12",
    heal_threshold: float = 0.35,
) -> dict[str, Any]:
    """Execute one full System 1 / System 2 run from starter save state."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    timeline: list[dict[str, Any]] = []
    gif_frames: list[Image.Image] = []
    total_presses = 0

    total_battles_started = 0
    total_battles_won = 0
    total_blackouts = 0
    total_heals_done = 0
    total_heals_succeeded = 0

    current_goal_str = "init"
    start_time = time.time()

    with PokemonRed(state=state_path) as env:
        # Wrap env.step to count total presses and capture GIF frames every ~20 presses
        orig_step = env.step

        def wrapped_step(action: int | str):
            nonlocal total_presses
            frame, ram_obs = orig_step(action)
            total_presses += 1
            if save_gif and total_presses % 20 == 0:
                st_gif = party_status(env)
                raw_rgb = screen_capture(env.pyboy)
                captioned = create_captioned_frame(
                    raw_rgb,
                    current_goal_str,
                    st_gif["level"],
                    st_gif["hp"],
                    st_gif["max_hp"],
                )
                gif_frames.append(captioned)
            return frame, ram_obs

        env.step = wrapped_step

        initial_status = party_status(env)
        level_start = initial_status["level"]
        print(f"\n[Run seed={seed}] Starting run: Level {level_start}, HP {initial_status['hp']}/{initial_status['max_hp']}, Map {initial_status['map_id']}")

        while total_presses < budget:
            status = party_status(env)
            goal, reason = plan(status, target_level=target_level, current_goal=current_goal_str,
                                heal_threshold=heal_threshold)
            current_goal_str = str(goal)

            elapsed = round(time.time() - start_time, 2)
            decision_log = {
                "time": elapsed,
                "presses": total_presses,
                "goal": str(goal),
                "reason": reason,
                "level": status["level"],
                "hp": status["hp"],
                "max_hp": status["max_hp"],
                "map": status["map_id"],
            }
            print(
                f"[Planner t={elapsed:5.1f}s p={total_presses:4d}] "
                f"Goal: {goal:<6} | Level: {status['level']} | HP: {status['hp']:2d}/{status['max_hp']:2d} | "
                f"Map: {status['map_id']:2d} | Reason: {reason}"
            )

            if goal == "done":
                decision_log["skill"] = "none"
                decision_log["result"] = {"success": True}
                timeline.append(decision_log)
                break

            skill_name = "none"
            skill_result: dict[str, Any] = {"success": False, "presses": 0}
            remaining_budget = budget - total_presses

            if goal == "heal":
                if status["map_id"] == MAP_ROUTE1:
                    skill_name = "navigate_north_to_viridian"
                    exits = {MAP_ROUTE1: ROUTE1_NORTH_WAYPOINTS, **DETOUR_EXITS}
                    skill_result = navigate(
                        env, exits, target_map=MAP_VIRIDIAN, budget=min(600, remaining_budget), rng=rng
                    )
                elif status["map_id"] in (MAP_VIRIDIAN, MAP_VIRIDIAN_POKECENTER):
                    skill_name = "heal"
                    total_heals_done += 1
                    skill_result = heal(env, budget=min(300, remaining_budget))
                    if skill_result["success"]:
                        total_heals_succeeded += 1
                elif status["map_id"] == MAP_PALLET:
                    skill_name = "navigate_pallet_to_route1"
                    exits = {MAP_PALLET: [(10, 0), (11, 0)], **DETOUR_EXITS}
                    skill_result = navigate(
                        env, exits, target_map=MAP_ROUTE1, budget=min(200, remaining_budget), rng=rng
                    )
                elif status["map_id"] in DETOUR_EXITS:
                    skill_name = "navigate_detour_exit"
                    skill_result = navigate(
                        env, DETOUR_EXITS, target_map=[MAP_PALLET, MAP_VIRIDIAN, MAP_ROUTE1], budget=min(200, remaining_budget), rng=rng
                    )

            elif goal == "grind":
                if status["map_id"] == MAP_PALLET:
                    skill_name = "navigate_pallet_to_route1"
                    exits = {MAP_PALLET: [(10, 0), (11, 0)], **DETOUR_EXITS}
                    skill_result = navigate(
                        env, exits, target_map=MAP_ROUTE1, budget=min(200, remaining_budget), rng=rng
                    )
                elif status["map_id"] == MAP_VIRIDIAN:
                    skill_name = "navigate_viridian_to_route1"
                    exits = {MAP_VIRIDIAN: VIRIDIAN_TO_ROUTE1_WAYPOINTS, **DETOUR_EXITS}
                    skill_result = navigate(
                        env, exits, target_map=MAP_ROUTE1, budget=min(200, remaining_budget), rng=rng
                    )
                elif status["map_id"] == MAP_VIRIDIAN_POKECENTER:
                    skill_name = "exit_pokecenter"
                    # Step down to exit
                    while env.observe()[1]["map_id"] == MAP_VIRIDIAN_POKECENTER and total_presses < budget:
                        env.step(0)  # DOWN
                    skill_result = {"success": True, "presses": 0}
                elif status["map_id"] in DETOUR_EXITS:
                    skill_name = "navigate_detour_exit"
                    skill_result = navigate(
                        env, DETOUR_EXITS, target_map=[MAP_PALLET, MAP_VIRIDIAN, MAP_ROUTE1], budget=min(200, remaining_budget), rng=rng
                    )
                elif status["map_id"] == MAP_ROUTE1:
                    skill_name = "grind"
                    skill_result = grind(
                        env,
                        budget=min(500, remaining_budget),
                        target_level=target_level,
                        rng=rng,
                        hp_threshold=heal_threshold,
                    )

            # Accumulate event statistics
            total_battles_started += skill_result.get("battles_started", 0)
            total_battles_won += skill_result.get("battles_won", 0)
            total_blackouts += skill_result.get("blackouts", 0)

            decision_log["skill"] = skill_name
            decision_log["result"] = {
                "success": skill_result.get("success", False),
                "presses": skill_result.get("presses", 0),
            }
            timeline.append(decision_log)

        final_status = party_status(env)
        level_end = final_status["level"]
        levels_gained = level_end - level_start
        goal_reached = level_end >= target_level
        presses_per_level = round(total_presses / levels_gained, 2) if levels_gained > 0 else None

        metrics = {
            "seed": seed,
            "target_level": target_level,
            "level_start": level_start,
            "level_end": level_end,
            "levels_gained": levels_gained,
            "presses_used": total_presses,
            "battles_started": total_battles_started,
            "battles_won": total_battles_won,
            "blackouts": total_blackouts,
            "heals_done": total_heals_done,
            "heals_succeeded": total_heals_succeeded,
            "presses_per_level": presses_per_level,
            "goal_reached": goal_reached,
        }

        # Save run JSON
        run_file = work_dir / f"run_{seed}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "timeline": timeline}, f, indent=2)
        print(f"Saved run metrics -> {run_file}")

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


def main():
    parser = argparse.ArgumentParser(description="System 1 / System 2 loop for leveling up & healing.")
    parser.add_argument("--target-level", type=int, default=10, help="Target level to stop grinding.")
    parser.add_argument("--budget", type=int, default=4000, help="Total action press budget per run.")
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed.")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs.")
    parser.add_argument("--gif", action="store_true", help="Record and save GIF of each run.")
    parser.add_argument("--state", type=str, default="saves/starter_pallet.state", help="Path to starter state.")
    parser.add_argument("--work-dir", type=str, default="work/system12", help="Directory for output files.")
    parser.add_argument("--heal-threshold", type=float, default=0.35, help="System 2: heal when HP fraction drops below this.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for r in range(args.runs):
        cur_seed = args.seed + r
        m = run_agent(
            target_level=args.target_level,
            budget=args.budget,
            seed=cur_seed,
            save_gif=args.gif,
            state_path=args.state,
            work_dir=work_dir,
            heal_threshold=args.heal_threshold,
        )
        all_metrics.append(m)

    # Compute summary
    n_runs = len(all_metrics)
    summary = {
        "n_runs": n_runs,
        "runs": all_metrics,
        "summary": {
            "target_level": args.target_level,
            "budget": args.budget,
            "heal_threshold": args.heal_threshold,
            "goals_reached": sum(1 for m in all_metrics if m["goal_reached"]),
            "avg_levels_gained": round(float(np.mean([m["levels_gained"] for m in all_metrics])), 2),
            "avg_presses_used": round(float(np.mean([m["presses_used"] for m in all_metrics])), 2),
            "avg_battles_started": round(float(np.mean([m["battles_started"] for m in all_metrics])), 2),
            "avg_battles_won": round(float(np.mean([m["battles_won"] for m in all_metrics])), 2),
            "avg_blackouts": round(float(np.mean([m["blackouts"] for m in all_metrics])), 2),
            "avg_heals_done": round(float(np.mean([m["heals_done"] for m in all_metrics])), 2),
            "avg_heals_succeeded": round(float(np.mean([m["heals_succeeded"] for m in all_metrics])), 2),
            "avg_presses_per_level": round(float(np.mean([m["presses_per_level"] for m in all_metrics if m["presses_per_level"] is not None])), 2) if any(m["presses_per_level"] is not None for m in all_metrics) else None,
        },
    }

    summary_file = work_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary over {n_runs} runs -> {summary_file}")


if __name__ == "__main__":
    main()
