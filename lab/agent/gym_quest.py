"""gym_quest.py — System 1 / System 2 loop to win the Boulder Badge in Pokémon Red.

Architecture:
- System 2 Planner:
    Rule planner over RAM status choosing high-level goals:
    * If Boulder Badge obtained (wObtainedBadges bit 0 set) -> "done"
    * Elif hp_frac < heal_threshold (or Pewter PC unhealed) -> "heal"
    * Elif level < target_level -> "grind" (Route 1 grass patch)
    * Elif not in Pewter City (map 2 / 54 / 58) -> "travel_to_pewter"
    * Elif in Pewter City with full HP -> "gym" (Jr. Trainer and Brock)
    * After blackout (respawn at Pokemon Center) -> re-plans smoothly.

- System 1 Skills:
    * navigate_to (nav.py): closed-loop privileged BFS navigation over RAM map blocks.
    * heal_pkmn_center (nav.py): universal Pokemon Center nurse healing and exit.
    * battle_skill (nav.py): move choice (Ember vs Rock/Ground/Bug), menu navigation, switch handling.
    * grind: pacing in grass to trigger wild battles with battle_skill.
    * fight_gym: challenges Jr. Trainer and Brock, checks badge flag.
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

from lab.agent.nav import (
    MAP_PEWTER_CITY,
    MAP_PEWTER_GYM,
    MAP_PEWTER_POKECENTER,
    MAP_ROUTE_1,
    MAP_ROUTE_2,
    MAP_VIRIDIAN_CITY,
    MAP_VIRIDIAN_FOREST,
    MAP_VIRIDIAN_FOREST_NORTH_GATE,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE,
    MAP_VIRIDIAN_POKECENTER,
    MOVE_NAMES,
    WRAM_OBTAINED_BADGES,
    battle_skill,
    heal_pkmn_center,
    navigate_to,
)
from lab.agent.system12 import party_status
from lab.env import ROM_PATH, PokemonRed, ACTION_TO_INDEX
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

# Route 1 tall grass anchor coordinates
GRASS_PATCH_TILES = [(13, 30), (14, 30)]


# ── System 2 Goal Class ──────────────────────────────────────────────────────
class Goal(str):
    """Goal string with an attached explanation reason."""

    def __new__(cls, goal: str, reason: str = ""):
        obj = super().__new__(cls, goal)
        obj.reason = reason
        return obj

    def __iter__(self):
        yield str(self)
        yield self.reason


# ── System 2 Planner ─────────────────────────────────────────────────────────
def plan_gym(
    status: dict[str, Any],
    target_level: int = 15,
    pewter_healed: bool = False,
    heal_threshold: float = 0.35,
) -> Goal:
    """System 2 rule planner over Game Boy RAM.

    Selects next high-level goal:
        - "done": Boulder Badge acquired.
        - "heal": HP below safe threshold, or healing at Pewter to set respawn point.
        - "grind": Level below target_level.
        - "travel_to_pewter": Navigating through Viridian Forest to Pewter City.
        - "gym": In Pewter City with full HP, challenging Gym Leader Brock.
    """
    badges = status.get("badges", 0)
    if bool(badges & 0x01):
        return Goal("done", "Boulder Badge obtained in RAM (wObtainedBadges bit 0)")

    level = status.get("level", 0)
    hp = status.get("hp", 0)
    max_hp = status.get("max_hp", 1)
    hp_frac = status.get("hp_frac", 1.0)
    map_id = status.get("map_id", 0)

    in_pewter_area = map_id in (MAP_PEWTER_CITY, MAP_PEWTER_POKECENTER, MAP_PEWTER_GYM)

    # Healing priority
    if in_pewter_area:
        if not pewter_healed or hp < max_hp:
            return Goal("heal", f"In Pewter area, healing at Pewter Pokémon Center (HP {hp}/{max_hp}) and setting respawn anchor")
    else:
        if hp_frac < heal_threshold:
            return Goal("heal", f"HP low: {hp}/{max_hp} ({hp_frac:.2f} < {heal_threshold}), healing at Viridian Pokémon Center")
        elif level >= target_level and hp < max_hp:
            return Goal("heal", f"HP {hp}/{max_hp}, healing at Viridian Pokémon Center before Route 2")

    # Level grinding
    if level < target_level and not in_pewter_area:
        return Goal("grind", f"Level {level} < target {target_level}, grinding on Route 1")

    # Travel to Pewter City
    if not in_pewter_area:
        return Goal("travel_to_pewter", f"Level {level} >= target {target_level}, traveling through Viridian Forest to Pewter City")

    # Gym challenge
    return Goal("gym", f"In Pewter City at full HP (Level {level}), entering Pewter Gym to defeat Jr. Trainer and Brock")


# ── System 1 Skills ──────────────────────────────────────────────────────────
def grind_skill(
    env: PokemonRed,
    target_level: int = 15,
    hp_threshold: float = 0.35,
    budget: int = 2500,
) -> dict[str, Any]:
    """Paces back and forth in Route 1 tall grass, fighting wild encounters with battle_skill."""
    presses = 0
    battles_won = 0
    battles_started = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    # Ensure player is on Route 1
    st = party_status(env)
    if st["map_id"] != MAP_ROUTE_1:
        r_nav = navigate_to(env, MAP_ROUTE_1, budget=budget - presses)
        presses += r_nav["presses"]
        if not r_nav["success"]:
            return {"presses": presses, "success": False, "battles_won": 0, "blackouts": 0, "moves_used": moves_used}

    # Walk to tall grass anchor (13, 30)
    if (env.observe()[1]["x"], env.observe()[1]["y"]) not in GRASS_PATCH_TILES:
        r_grass = navigate_to(env, MAP_ROUTE_1, target_tiles=[(13, 30), (14, 30)], budget=budget - presses)
        presses += r_grass["presses"]

    # Pacing loop in tall grass
    while presses < budget:
        st = party_status(env)
        if st["level"] >= target_level or st["hp_frac"] < hp_threshold:
            break

        if env.observe()[1]["in_battle"]:
            battles_started += 1
            b_res = battle_skill(env)
            presses += b_res["presses"]
            for m_name, count in b_res.get("moves_used", {}).items():
                moves_used[m_name] = moves_used.get(m_name, 0) + count

            if b_res.get("blackout", False):
                blackouts += 1
                events.append({"type": "blackout", "map": MAP_ROUTE_1})
                break

            battles_won += 1
            events.append({"type": "battle_won", "level": party_status(env)["level"]})
            continue

        act = "left" if env.observe()[1]["x"] >= 14 else "right"
        env.step(act)
        presses += 1

    return {
        "presses": presses,
        "success": battles_won > 0 or party_status(env)["level"] >= target_level,
        "battles_won": battles_won,
        "battles_started": battles_started,
        "blackouts": blackouts,
        "moves_used": moves_used,
        "events": events,
    }


def travel_to_pewter_skill(env: PokemonRed, budget: int = 4000) -> dict[str, Any]:
    """Navigates the multi-map route from Viridian/Route 1 all the way to Pewter City (map 2)."""
    presses = 0
    battles_won = 0
    battles_started = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    # Sequence of map stages to Pewter City
    stages = [
        (MAP_VIRIDIAN_CITY, "Viridian City"),
        (MAP_ROUTE_2, "Route 2 South"),
        (MAP_VIRIDIAN_FOREST_SOUTH_GATE, "South Gate"),
        (MAP_VIRIDIAN_FOREST, "Viridian Forest"),
        (MAP_VIRIDIAN_FOREST_NORTH_GATE, "North Gate"),
        (MAP_ROUTE_2, "Route 2 North"),
        (MAP_PEWTER_CITY, "Pewter City"),
    ]

    for next_map, label in stages:
        if presses >= budget:
            break
        cur_map = int(env.observe()[1]["map_id"])
        if cur_map == MAP_PEWTER_CITY:
            break
        # If already at or past this stage
        if cur_map == next_map:
            continue

        r = navigate_to(env, next_map, budget=budget - presses)
        presses += r["presses"]
        battles_started += r.get("battles_started", 0)
        battles_won += r.get("battles_won", 0)
        blackouts += r.get("blackouts", 0)
        events.extend(r.get("events", []))

        if r.get("blackouts", 0) > 0 or not r["success"]:
            return {
                "presses": presses,
                "success": False,
                "battles_won": battles_won,
                "battles_started": battles_started,
                "blackouts": blackouts,
                "moves_used": moves_used,
                "events": events,
            }

    success = (int(env.observe()[1]["map_id"]) == MAP_PEWTER_CITY)
    return {
        "presses": presses,
        "success": success,
        "battles_won": battles_won,
        "battles_started": battles_started,
        "blackouts": blackouts,
        "moves_used": moves_used,
        "events": events,
    }


def fight_gym_skill(env: PokemonRed, budget: int = 2500) -> dict[str, Any]:
    """Enters Pewter Gym, defeats Jr. Trainer, defeats Brock, and acquires the Boulder Badge."""
    presses = 0
    battles_won = 0
    battles_started = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    mem = env.pyboy.memory

    # Step 1: If in Pewter City (2), navigate into Pewter Gym (54)
    if int(env.observe()[1]["map_id"]) == MAP_PEWTER_CITY:
        r_gym = navigate_to(env, MAP_PEWTER_GYM, budget=budget - presses)
        presses += r_gym["presses"]
        if not r_gym["success"]:
            return {"presses": presses, "success": False, "badge_obtained": False}

    if int(env.observe()[1]["map_id"]) != MAP_PEWTER_GYM:
        return {"presses": presses, "success": False, "badge_obtained": False}

    # Step 2: Navigate to Jr. Trainer at (4, 6)
    r_jr = navigate_to(env, MAP_PEWTER_GYM, target_tiles=[(4, 6)], budget=budget - presses)
    presses += r_jr["presses"]

    # Clear pre-battle dialogue with A until battle engages
    for _ in range(40):
        if env.observe()[1]["in_battle"] or presses >= budget:
            break
        env.step("a")
        presses += 1

    # Fight Jr. Trainer if battle engaged
    if env.observe()[1]["in_battle"]:
        battles_started += 1
        b_jr = battle_skill(env)
        presses += b_jr["presses"]
        for m_name, count in b_jr.get("moves_used", {}).items():
            moves_used[m_name] = moves_used.get(m_name, 0) + count
        if b_jr.get("blackout", False):
            blackouts += 1
            return {"presses": presses, "success": False, "badge_obtained": False, "blackouts": blackouts, "moves_used": moves_used}
        battles_won += 1
        events.append({"type": "jr_trainer_defeated"})
        # Clear Jr Trainer post-battle dialogue with B
        for _ in range(20):
            env.step("b")
            presses += 1

    # If HP is depleted after Jr. Trainer, step outside to heal at Pewter PC before Brock
    st_jr = party_status(env)
    if st_jr["hp_frac"] < 0.75:
        navigate_to(env, MAP_PEWTER_CITY, budget=min(500, budget - presses))
        heal_pkmn_center(env, budget=min(800, budget - presses))
        navigate_to(env, MAP_PEWTER_GYM, budget=min(500, budget - presses))

    if int(env.observe()[1]["map_id"]) != MAP_PEWTER_GYM:
        navigate_to(env, MAP_PEWTER_GYM, budget=min(500, budget - presses))

    # Step 3: Navigate to Brock standing tile at (4, 2)
    # Brock is at (4, 1) facing DOWN
    r_brock = navigate_to(env, MAP_PEWTER_GYM, target_tiles=[(4, 2)], budget=budget - presses)
    presses += r_brock["presses"]

    # Face UP towards Brock
    while int(env.observe()[1]["facing"]) != 4 and presses < budget:
        env.step("up")
        presses += 1

    # Talk to Brock with A to trigger dialogue & battle
    for _ in range(50):
        if env.observe()[1]["in_battle"] or presses >= budget:
            break
        env.step("a")
        presses += 1

    # Fight Brock
    if env.observe()[1]["in_battle"]:
        battles_started += 1
        b_brock = battle_skill(env)
        presses += b_brock["presses"]
        for m_name, count in b_brock.get("moves_used", {}).items():
            moves_used[m_name] = moves_used.get(m_name, 0) + count
        if b_brock.get("blackout", False):
            blackouts += 1
            return {"presses": presses, "success": False, "badge_obtained": False, "blackouts": blackouts, "moves_used": moves_used}
        battles_won += 1
        events.append({"type": "brock_defeated"})

    # Step 4: Clear post-battle dialogue (Brock delivers Boulder Badge & TM34)
    # Press A to progress dialogue; avoid B to permit evolution if L16
    for _ in range(160):
        env.step("a")
        presses += 1
        if bool(mem[WRAM_OBTAINED_BADGES] & 0x01):
            for _ in range(25):
                env.step("a")
                presses += 1
            break

    badge_obtained = bool(mem[WRAM_OBTAINED_BADGES] & 0x01)
    return {
        "presses": presses,
        "success": badge_obtained,
        "badge_obtained": badge_obtained,
        "battles_won": battles_won,
        "battles_started": battles_started,
        "blackouts": blackouts,
        "moves_used": moves_used,
        "events": events,
    }


# ── GIF Caption Helper ────────────────────────────────────────────────────────
def create_captioned_frame(
    rgb_frame: np.ndarray, goal: str, level: int, hp: int, max_hp: int
) -> Image.Image:
    """Add a top banner displaying goal, level, and HP to a native 144x160 Game Boy frame."""
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


# ── Quest Runner Loop ────────────────────────────────────────────────────────
def run_gym_quest(
    target_level: int = 15,
    budget: int = 18000,
    seed: int = 100,
    save_gif: bool = False,
    save_traj: bool = False,
    state_path: str | Path = "saves/pokedex_viridian.state",
    work_dir: str | Path = "work/gym",
    traj_dir: str | Path = "work/gym_traj",
    heal_threshold: float = 0.35,
) -> dict[str, Any]:
    """Execute one full Gym Quest run from starter save state to Boulder Badge."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gif_frames: list[Image.Image] = []
    timeline: list[dict[str, Any]] = []
    maps_visited: list[int] = []
    moves_used_total: dict[str, int] = {}
    phase_presses: dict[str, int] = {}

    total_presses = 0
    total_battles_started = 0
    total_battles_won = 0
    total_blackouts = 0

    pewter_healed = False
    level_at_gym: Optional[int] = None
    badge_obtained = False

    traj_frames: list[np.ndarray] = []
    traj_actions: list[int] = []
    traj_maps: list[int] = []
    traj_xs: list[int] = []
    traj_ys: list[int] = []
    traj_in_battles: list[bool] = []
    traj_goals: list[str] = []

    current_goal_str = "init"
    start_time = time.time()

    with PokemonRed(state=state_path) as env:
        # The game is deterministic for a fixed input sequence, so without this every seed replays one trajectory
        # (seeds 100-109 all took 8,664 presses). Idling a seed-dependent number of frames moves the game's RNG
        # (it advances every frame), which changes encounters, damage rolls and catches.
        idle_frames = int(np.random.default_rng(seed).integers(0, 600))
        env.pyboy.tick(idle_frames, render=False)
        # Wrap env.step to track presses, visited maps, and capture GIF frames
        orig_step = env.step

        def wrapped_step(action: int | str):
            nonlocal total_presses
            frame, ram_obs = orig_step(action)
            total_presses += 1

            if save_traj:
                if isinstance(action, (int, np.integer)):
                    act_idx = int(action)
                else:
                    act_idx = ACTION_TO_INDEX.get(str(action).lower(), 7)
                traj_frames.append(frame)
                traj_actions.append(act_idx)
                traj_maps.append(int(ram_obs["map_id"]))
                traj_xs.append(int(ram_obs["x"]))
                traj_ys.append(int(ram_obs["y"]))
                traj_in_battles.append(bool(ram_obs["in_battle"]))
                traj_goals.append(current_goal_str)

            # Map visit tracking
            m_id = int(ram_obs["map_id"])
            if not maps_visited or maps_visited[-1] != m_id:
                maps_visited.append(m_id)

            # GIF capture (every 25 presses to keep total size under 15MB)
            if save_gif and total_presses % 25 == 0:
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

        init_st = party_status(env)
        maps_visited.append(init_st["map_id"])
        print(f"\n[GymQuest seed={seed}] Starting run: Level {init_st['level']}, HP {init_st['hp']}/{init_st['max_hp']}, Map {init_st['map_id']}")

        # Initial heal at Viridian PC to set full 25/25 HP and respawn anchor
        if init_st["hp"] < init_st["max_hp"] and init_st["map_id"] == MAP_VIRIDIAN_CITY:
            current_goal_str = "heal"
            p_before = total_presses
            h_init = heal_pkmn_center(env, budget=500)
            phase_presses["heal"] = phase_presses.get("heal", 0) + (total_presses - p_before)
            print(f"[GymQuest seed={seed}] Initial Viridian PC heal: {h_init['hp']}/{h_init['max_hp']} HP")

        while total_presses < budget:
            st = party_status(env)
            badges = int(env.pyboy.memory[WRAM_OBTAINED_BADGES])
            st["badges"] = badges

            goal, reason = plan_gym(
                st,
                target_level=target_level,
                pewter_healed=pewter_healed,
                heal_threshold=heal_threshold,
            )
            current_goal_str = str(goal)
            elapsed = round(time.time() - start_time, 2)

            log_entry = {
                "time": elapsed,
                "presses": total_presses,
                "goal": str(goal),
                "reason": reason,
                "level": st["level"],
                "hp": st["hp"],
                "max_hp": st["max_hp"],
                "map": st["map_id"],
            }
            timeline.append(log_entry)
            print(f"[Planner t={elapsed:5.1f}s p={total_presses:5d}] Goal: {goal:<16} | Lv: {st['level']:2d} | HP: {st['hp']:2d}/{st['max_hp']:2d} | Map: {st['map_id']:2d} | {reason}")

            if goal == "done":
                badge_obtained = True
                break

            p_before = total_presses

            # Execute System 1 skill corresponding to goal
            if goal == "heal":
                h_res = heal_pkmn_center(env, budget=min(800, budget - total_presses))
                phase_presses["heal"] = phase_presses.get("heal", 0) + (total_presses - p_before)
                if int(env.observe()[1]["map_id"]) in (MAP_PEWTER_CITY, MAP_PEWTER_POKECENTER):
                    if h_res.get("healed", False):
                        pewter_healed = True

            elif goal == "grind":
                g_res = grind_skill(env, target_level=target_level, hp_threshold=heal_threshold, budget=min(3000, budget - total_presses))
                phase_presses["grind"] = phase_presses.get("grind", 0) + (total_presses - p_before)
                total_battles_started += g_res.get("battles_started", 0)
                total_battles_won += g_res.get("battles_won", 0)
                total_blackouts += g_res.get("blackouts", 0)
                for m_k, count in g_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

            elif goal == "travel_to_pewter":
                t_res = travel_to_pewter_skill(env, budget=min(4500, budget - total_presses))
                phase_presses["travel_to_pewter"] = phase_presses.get("travel_to_pewter", 0) + (total_presses - p_before)
                total_battles_started += t_res.get("battles_started", 0)
                total_battles_won += t_res.get("battles_won", 0)
                total_blackouts += t_res.get("blackouts", 0)
                for m_k, count in t_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

            elif goal == "gym":
                if level_at_gym is None:
                    level_at_gym = party_status(env)["level"]
                gym_res = fight_gym_skill(env, budget=min(4000, budget - total_presses))
                phase_presses["gym"] = phase_presses.get("gym", 0) + (total_presses - p_before)
                total_battles_started += gym_res.get("battles_started", 0)
                total_battles_won += gym_res.get("battles_won", 0)
                total_blackouts += gym_res.get("blackouts", 0)
                for m_k, count in gym_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

                if gym_res.get("badge_obtained", False):
                    badge_obtained = True
                    break

            if total_presses == p_before:
                env.step("pass")

        final_st = party_status(env)
        final_badges = int(env.pyboy.memory[WRAM_OBTAINED_BADGES])
        badge_obtained = bool(final_badges & 0x01)

        result = {
            "seed": seed,
            "idle_frames": idle_frames,
            "badge_obtained": badge_obtained,
            "presses_total": total_presses,
            "phase_presses": phase_presses,
            "level_start": init_st["level"],
            "level_final": final_st["level"],
            "level_at_gym": level_at_gym or final_st["level"],
            "blackouts": total_blackouts,
            "battles_started": total_battles_started,
            "battles_won": total_battles_won,
            "moves_used": moves_used_total,
            "maps_visited": maps_visited,
            "timeline": timeline,
            "zero_memory_writes": True,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

        # Save trajectory if requested
        if save_traj and traj_frames:
            traj_dir = Path(traj_dir)
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_path = traj_dir / f"run_seed_{seed}.npz"
            np.savez_compressed(
                traj_path,
                frames=np.array(traj_frames, dtype=np.uint8),
                actions=np.array(traj_actions, dtype=np.uint8),
                map_ids=np.array(traj_maps, dtype=np.uint8),
                xs=np.array(traj_xs, dtype=np.uint8),
                ys=np.array(traj_ys, dtype=np.uint8),
                in_battles=np.array(traj_in_battles, dtype=bool),
                goals=np.array(traj_goals, dtype=str),
                map=np.array(traj_maps, dtype=np.uint8),
                x=np.array(traj_xs, dtype=np.uint8),
                y=np.array(traj_ys, dtype=np.uint8),
                in_battle=np.array(traj_in_battles, dtype=bool),
                goal=np.array(traj_goals, dtype=str),
            )
            print(f"[GymQuest seed={seed}] Saved trajectory ({len(traj_frames)} steps) -> {traj_path}")

        # Save single run JSON
        run_file = work_dir / f"run_seed_{seed}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[GymQuest seed={seed}] Run complete: Badge={badge_obtained}, Presses={total_presses}, Level={final_st['level']}, Saved -> {run_file}")

        # Save GIF if requested
        if save_gif and gif_frames:
            gif_path = work_dir / f"run_seed_{seed}.gif"
            print(f"[GymQuest seed={seed}] Saving GIF ({len(gif_frames)} frames) -> {gif_path} ...")
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                optimize=True,
                duration=50,
                loop=0,
            )
            print(f"[GymQuest seed={seed}] GIF saved! Size: {gif_path.stat().st_size / (1024*1024):.2f} MB")

        return result


# ── Batch Runner & CLI ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="PokeDreamer System 1/2 Gym Quest.")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--seed", type=int, default=100, help="Initial random seed")
    parser.add_argument("--target-level", type=int, default=15, help="Target level before traveling to Pewter")
    parser.add_argument("--budget", type=int, default=18000, help="Total action budget per run")
    parser.add_argument("--work-dir", type=str, default="work/gym", help="Directory for output JSONs and GIFs")
    parser.add_argument("--state", type=str, default="saves/pokedex_viridian.state", help="Path to initial save state")
    parser.add_argument("--gif", action="store_true", help="Record and save captioned GIF of run")
    parser.add_argument("--save-traj", action="store_true", help="Record and save full trajectory npz to work/gym_traj/run_seed_<seed>.npz")
    parser.add_argument("--traj-dir", type=str, default="work/gym_traj", help="Directory for output trajectory npz files")
    parser.add_argument("--heal-threshold", type=float, default=0.35, help="HP fraction threshold to trigger healing")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for r_idx in range(args.runs):
        cur_seed = args.seed + r_idx
        # Record GIF only on the first run if --gif is specified
        record_gif_this_run = args.gif and (r_idx == 0)
        res = run_gym_quest(
            target_level=args.target_level,
            budget=args.budget,
            seed=cur_seed,
            save_gif=record_gif_this_run,
            save_traj=args.save_traj,
            state_path=args.state,
            work_dir=work_dir,
            traj_dir=args.traj_dir,
            heal_threshold=args.heal_threshold,
        )
        results.append(res)

    # Compute and save summary.json
    presses = [r["presses_total"] for r in results]
    badges = [r["badge_obtained"] for r in results]
    blackouts = [r["blackouts"] for r in results]
    levels = [r["level_final"] for r in results]

    summary = {
        "runs": len(results),
        "target_level": args.target_level,
        "badge_rate": f"{sum(badges)}/{len(results)}",
        "badge_success_pct": round(sum(badges) / len(results) * 100, 1),
        "presses_mean": round(float(np.mean(presses)), 1),
        "presses_min": int(np.min(presses)),
        "presses_max": int(np.max(presses)),
        "blackouts_total": sum(blackouts),
        "blackouts_mean": round(float(np.mean(blackouts)), 2),
        "level_final_mean": round(float(np.mean(levels)), 1),
        "all_runs": [
            {
                "seed": r["seed"],
                "badge": r["badge_obtained"],
                "presses": r["presses_total"],
                "level_final": r["level_final"],
                "level_gym": r["level_at_gym"],
                "blackouts": r["blackouts"],
                "battles_won": r["battles_won"],
            }
            for r in results
        ],
    }

    summary_file = work_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {summary['badge_rate']} Badges ({summary['badge_success_pct']}%)")
    print(f"Presses: mean={summary['presses_mean']}, min={summary['presses_min']}, max={summary['presses_max']}")
    print(f"Summary written to: {summary_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
