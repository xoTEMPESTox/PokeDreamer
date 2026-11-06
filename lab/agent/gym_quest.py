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
    * Navigation modes (--nav bfs|learned):
        - bfs (default): privileged BFS over map layout / collision data read from Game Boy RAM/ROM (nav.py navigate_to).
        - learned: frames-only policies trained in the diffusion world model (mpc.py learned_action on CPU).
          Drives travel legs (Route 1 -> Viridian, Viridian -> Route 2, Route 2 -> South Gate, Viridian Forest,
          Route 2 North -> Pewter, Pewter -> Gym door). Falls back to navigate_to if policy file missing or budget exhausted.
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
import torch

from lab.agent.imagine_rl import Policy
from lab.agent.mpc import in_dialog, learned_action

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
from lab.game.game_state import screen_capture
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

# Route 1 tall grass anchor coordinates
GRASS_PATCH_TILES = [(13, 30), (14, 30)]

DEFAULT_LEG_POLICIES: dict[str, str] = {
    "leave_viridian_pc": "work/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt",
    "to_grass": "work/agent_rl/to_grass_wm31_gate_s0/policy.pt",
    "to_viridian_pc": "work/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt",
    "viridian": "work/agent_rl/viridian_geo_clip/policy.pt",
    "to_route2": "work/agent_rl/to_route2_wm31/policy.pt",
    "to_gate": "work/agent_rl/to_gate_wm31/policy.pt",
    "forest": "work/agent_rl/forest_wm31/policy.pt",
    "to_pewter": "work/agent_rl/to_pewter_wm31/policy.pt",
    "to_pewter_pc": "work/agent_rl/to_pewter_pc_wm31_gate_s0/policy.pt",
    "leave_pewter_pc": "work/agent_rl/leave_pewter_pc_wm31_gate_s0/policy.pt",
    "to_gym": "work/agent_rl/to_gym_wm31/policy.pt",
}

TRAVEL_LEGS: list[tuple[str, int, list[int]]] = [
    ("viridian", MAP_VIRIDIAN_CITY, [MAP_VIRIDIAN_CITY]),
    ("to_route2", MAP_ROUTE_2, [MAP_ROUTE_2]),
    ("to_gate", MAP_VIRIDIAN_FOREST_SOUTH_GATE, [MAP_VIRIDIAN_FOREST_SOUTH_GATE]),
    ("forest", MAP_VIRIDIAN_FOREST_NORTH_GATE, [MAP_VIRIDIAN_FOREST, MAP_VIRIDIAN_FOREST_NORTH_GATE]),
    ("to_pewter", MAP_PEWTER_CITY, [MAP_ROUTE_2, MAP_PEWTER_CITY]),
]

STAGE_TO_LEG: dict[int, str] = {
    MAP_VIRIDIAN_CITY: "viridian",
    MAP_ROUTE_2: "to_route2",
    MAP_VIRIDIAN_FOREST_SOUTH_GATE: "to_gate",
    MAP_VIRIDIAN_FOREST_NORTH_GATE: "forest",
    MAP_PEWTER_CITY: "to_pewter",
    MAP_PEWTER_GYM: "to_gym",
}


ROUTE2_SOUTH_Y = 43  # Route 2 rows with an observed path to the South Gate: 43-71; to Pewter City: 0-11 (ext train split)


def first_travel_leg(map_id: int, y: int) -> int:
    """Index in TRAVEL_LEGS where learned travel starts, so it resumes mid-route (e.g. after a leg ran out of presses)
    instead of restarting with the viridian leg. Route 2 is two regions split by the forest, told apart by the row.
    Anywhere else it is 0 (on Viridian City the viridian leg returns at once)."""
    names = [leg for leg, _, _ in TRAVEL_LEGS]
    if map_id == MAP_ROUTE_2:
        return names.index("to_gate" if y >= ROUTE2_SOUTH_Y else "to_pewter")
    if map_id in (MAP_VIRIDIAN_FOREST_SOUTH_GATE, MAP_VIRIDIAN_FOREST):
        return names.index("forest")
    if map_id == MAP_VIRIDIAN_FOREST_NORTH_GATE:
        return names.index("to_pewter")
    return 0


def leg_for_stage(target_map: int) -> str:
    """Return the policy leg corresponding to a navigation target stage map."""
    if target_map not in STAGE_TO_LEG:
        raise ValueError(f"Unknown leg for target stage map {target_map}")
    return STAGE_TO_LEG[target_map]


_POLICY_CACHE: dict[str, Any] = {}


def load_leg_policy(path: str | Path, device: str = "cpu") -> Optional[Any]:
    """Loads a Policy checkpoint on CPU if the file exists, returning None otherwise."""
    p = Path(path)
    if not p.is_file():
        return None
    p_str = str(p)
    if p_str in _POLICY_CACHE:
        return _POLICY_CACHE[p_str]
    net = Policy().to(device).eval()
    ckpt = torch.load(p, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["policy"])
    _POLICY_CACHE[p_str] = net
    return net


def run_leg(
    env: PokemonRed,
    leg: str,
    target_map: int,
    policy_path: str | Path,
    fallback_maps: list[int],
    budget: int = 1000,
    policy_net: Optional[Any] = None,
    device: str = "cpu",
    fallback_budget: Optional[int] = None,
    target_tiles: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Any]:
    """Execute a single navigation leg using learned policy, falling back to navigate_to on failure.

    Loop until RAM map == target_map (and in target_tiles if specified) or budget runs out:
        - in_battle -> battle_skill(env)
        - in_dialog(env.pyboy) -> press A
        - else -> learned_action(net, frames, device)
    If policy file missing or budget runs out: fall back to navigate_to.
    Already on target_map (e.g. after a heal trip): reached_by_policy is None, and the leg is not counted as attempted.
    """
    def _is_target(r: dict[str, Any]) -> bool:
        if int(r["map_id"]) != target_map:
            return False
        if target_tiles is not None:
            return (int(r["x"]), int(r["y"])) in target_tiles
        return True

    _, ram0 = env.observe()
    start = [int(ram0["map_id"]), int(ram0["x"]), int(ram0["y"])]
    if _is_target(ram0):
        return {
            "leg": leg,
            "start": start,
            "policy": str(policy_path),
            "reached_by_policy": None,
            "policy_presses": 0,
            "fallback_presses": 0,
            "battles": 0,
            "success": True,
            "blackouts": 0,
            "moves_used": {},
            "events": [],
        }

    net = policy_net
    if net is None and Path(policy_path).is_file():
        net = load_leg_policy(policy_path, device=device)

    policy_presses = 0
    fallback_presses = 0
    battles = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    # If policy exists, run learned policy loop
    if net is not None:
        env._nav_leg, env._nav_by = leg, "policy"  # read by the --save-color recorder
        cur_frame, ram = env.observe()
        frames = [cur_frame] * 4  # CTX = 4

        while not _is_target(ram) and policy_presses < budget:
            # Overworld blackout check
            st_party = party_status(env)
            if st_party["hp"] == 0 and st_party["party_size"] > 0:
                blackouts += 1
                for _ in range(50):
                    env.step("a")
                    policy_presses += 1
                    if party_status(env)["map_id"] in (MAP_VIRIDIAN_POKECENTER, MAP_PEWTER_POKECENTER):
                        break
                events.append({"type": "blackout", "map": int(ram["map_id"]), "presses": policy_presses})
                break

            if ram["in_battle"]:
                battles += 1
                b_res = battle_skill(env)
                p_battle = b_res.get("presses", 0)
                policy_presses += p_battle
                for m_k, count in b_res.get("moves_used", {}).items():
                    moves_used[m_k] = moves_used.get(m_k, 0) + count
                if b_res.get("blackout", False):
                    blackouts += 1
                    events.append({"type": "blackout", "map": int(ram["map_id"]), "presses": policy_presses})
                    break
                events.append({"type": "battle_won", "map": int(ram["map_id"]), "moves": b_res.get("moves_used", {})})
                cur_frame, ram = env.observe()
                frames.append(cur_frame)
                continue

            if in_dialog(env.pyboy):
                frame, ram = env.step("a")
                policy_presses += 1
                frames.append(frame)
            else:
                a = learned_action(net, frames, device)
                frame, ram = env.step(a)
                policy_presses += 1
                frames.append(frame)

        if _is_target(ram):
            for _ in range(4):  # settle: RAM updates x, y a few frames after the map id (as navigate_to does)
                env.step("pass")
                policy_presses += 1

    reached_by_policy = _is_target(env.observe()[1])

    # Fallback to navigate_to if not reached by policy (and no blackout)
    if not reached_by_policy and blackouts == 0:
        env._nav_leg, env._nav_by = leg, "bfs"
        fb_limit = fallback_budget if fallback_budget is not None else budget
        for f_map in fallback_maps:
            cur_map = int(env.observe()[1]["map_id"])
            if cur_map == f_map or (cur_map == target_map and target_tiles is None):
                continue
            r_nav = navigate_to(env, f_map, budget=max(0, fb_limit - fallback_presses))
            fallback_presses += r_nav["presses"]
            battles += r_nav.get("battles_started", 0)
            blackouts += r_nav.get("blackouts", 0)
            events.extend(r_nav.get("events", []))
            for m_k, count in r_nav.get("moves_used", {}).items():
                moves_used[m_k] = moves_used.get(m_k, 0) + count
            if not r_nav["success"] or r_nav.get("blackouts", 0) > 0:
                break

        if target_tiles is not None and blackouts == 0:
            cur_ram = env.observe()[1]
            if int(cur_ram["map_id"]) == target_map and (int(cur_ram["x"]), int(cur_ram["y"])) not in target_tiles:
                r_nav = navigate_to(env, target_map, target_tiles=target_tiles, budget=max(0, fb_limit - fallback_presses))
                fallback_presses += r_nav["presses"]
                battles += r_nav.get("battles_started", 0)
                blackouts += r_nav.get("blackouts", 0)
                events.extend(r_nav.get("events", []))
                for m_k, count in r_nav.get("moves_used", {}).items():
                    moves_used[m_k] = moves_used.get(m_k, 0) + count

    env._nav_leg, env._nav_by = "", ""
    success = _is_target(env.observe()[1])
    return {
        "leg": leg,
        "start": start,
        "policy": str(policy_path),
        "reached_by_policy": reached_by_policy,
        "policy_presses": policy_presses,
        "fallback_presses": fallback_presses,
        "battles": battles,
        "success": success,
        "blackouts": blackouts,
        "moves_used": moves_used,
        "events": events,
    }


# ── System 2 Goal Class ──────────────────────────────────────────────────────
class Goal(str):
    """Goal string with an attached explanation reason."""

    def __new__(cls, goal: str, reason: str = "", target_level: Optional[int] = None):
        obj = super().__new__(cls, goal)
        obj.reason = reason
        obj.target_level = target_level
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
    # navigate_to cannot route from here back to the Viridian Pokecenter, so no top-up heal once on the way
    past_viridian = map_id in (MAP_ROUTE_2, MAP_VIRIDIAN_FOREST_SOUTH_GATE, MAP_VIRIDIAN_FOREST,
                               MAP_VIRIDIAN_FOREST_NORTH_GATE)

    # Healing priority
    if in_pewter_area:
        if not pewter_healed or hp < max_hp:
            return Goal("heal", f"In Pewter area, healing at Pewter Pokémon Center (HP {hp}/{max_hp}) and setting respawn anchor")
    else:
        if hp_frac < heal_threshold:
            return Goal("heal", f"HP low: {hp}/{max_hp} ({hp_frac:.2f} < {heal_threshold}), healing at Viridian Pokémon Center")
        elif level >= target_level and hp < max_hp and not past_viridian:
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
    nav: str = "bfs",
    leg_policies: Optional[dict[str, str]] = None,
    run_legs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Paces back and forth in Route 1 tall grass, fighting wild encounters with battle_skill."""
    presses = 0
    battles_won = 0
    battles_started = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    if nav == "bfs":
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
    else:
        # Learned navigation mode: trip to grass goes through run_leg with leg to_grass
        cur_obs = env.observe()[1]
        if int(cur_obs["map_id"]) != MAP_ROUTE_1 or (int(cur_obs["x"]), int(cur_obs["y"])) not in GRASS_PATCH_TILES:
            active_policies = dict(DEFAULT_LEG_POLICIES)
            if leg_policies:
                active_policies.update(leg_policies)
            pol_path = active_policies.get("to_grass", DEFAULT_LEG_POLICIES["to_grass"])
            leg_res = run_leg(
                env=env,
                leg="to_grass",
                target_map=MAP_ROUTE_1,
                target_tiles=GRASS_PATCH_TILES,
                policy_path=pol_path,
                fallback_maps=[MAP_ROUTE_1],
                budget=min(1000, max(0, budget - presses)),
            )
            leg_presses = leg_res["policy_presses"] + leg_res["fallback_presses"]
            presses += leg_presses
            battles_started += leg_res["battles"]
            blackouts += leg_res.get("blackouts", 0)
            events.extend(leg_res.get("events", []))
            battles_won += sum(1 for e in leg_res.get("events", []) if e.get("type") == "battle_won")
            for m_k, count in leg_res.get("moves_used", {}).items():
                moves_used[m_k] = moves_used.get(m_k, 0) + count

            if run_legs is not None:
                run_legs.append({
                    "leg": leg_res["leg"],
                    "start": leg_res["start"],
                    "policy": leg_res["policy"],
                    "reached_by_policy": leg_res["reached_by_policy"],
                    "policy_presses": leg_res["policy_presses"],
                    "fallback_presses": leg_res["fallback_presses"],
                    "battles": leg_res["battles"],
                })

            if leg_res.get("blackouts", 0) > 0 or not leg_res["success"]:
                return {
                    "presses": presses,
                    "success": False,
                    "battles_won": battles_won,
                    "battles_started": battles_started,
                    "blackouts": blackouts,
                    "moves_used": moves_used,
                    "events": events,
                }

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


def heal_skill(
    env: PokemonRed,
    budget: int = 800,
    nav: str = "bfs",
    leg_policies: Optional[dict[str, str]] = None,
    run_legs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Heal at the local Pokemon Center (Viridian or Pewter).

    In BFS mode: calls heal_pkmn_center directly (with exit_after=True).
    In learned mode:
      - reaches the PC via run_leg ('to_viridian_pc' or 'to_pewter_pc', with
        a learned 'PC -> city' first if inside the other PC);
      - heals at Nurse Joy counter via heal_pkmn_center(exit_after=False);
      - exits PC via run_leg ('leave_viridian_pc' or 'leave_pewter_pc').
    In learned mode each step gets the whole `budget`, not what is left of it: a policy that used its full allowance
    otherwise left the counter walk 0 presses (gym_learned_moves_s0: unhealed on the Pewter PC mat).
    """
    if nav == "bfs":
        return heal_pkmn_center(env, budget=budget, exit_after=True)

    presses = 0
    battles_started = 0
    battles_won = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    active_policies = dict(DEFAULT_LEG_POLICIES)
    if leg_policies:
        active_policies.update(leg_policies)

    def _execute_leg(leg_name: str, target_map: int, fallback_maps: list[int]) -> bool:
        nonlocal presses, battles_started, battles_won, blackouts
        pol_path = active_policies.get(leg_name, DEFAULT_LEG_POLICIES[leg_name])
        leg_res = run_leg(
            env=env,
            leg=leg_name,
            target_map=target_map,
            policy_path=pol_path,
            fallback_maps=fallback_maps,
            budget=budget,
        )
        leg_presses = leg_res["policy_presses"] + leg_res["fallback_presses"]
        presses += leg_presses
        battles_started += leg_res["battles"]
        blackouts += leg_res.get("blackouts", 0)
        events.extend(leg_res.get("events", []))
        battles_won += sum(1 for e in leg_res.get("events", []) if e.get("type") == "battle_won")
        for m_k, count in leg_res.get("moves_used", {}).items():
            moves_used[m_k] = moves_used.get(m_k, 0) + count

        if run_legs is not None:
            run_legs.append({
                "leg": leg_res["leg"],
                "start": leg_res["start"],
                "policy": leg_res["policy"],
                "reached_by_policy": leg_res["reached_by_policy"],
                "policy_presses": leg_res["policy_presses"],
                "fallback_presses": leg_res["fallback_presses"],
                "battles": leg_res["battles"],
            })
        return bool(leg_res["success"])

    _, ram = env.observe()
    cur_map = int(ram["map_id"])
    in_pewter = cur_map in (MAP_PEWTER_CITY, MAP_PEWTER_POKECENTER, MAP_PEWTER_GYM)
    target_pc = MAP_PEWTER_POKECENTER if in_pewter else MAP_VIRIDIAN_POKECENTER
    target_city = MAP_PEWTER_CITY if in_pewter else MAP_VIRIDIAN_CITY
    pc_leg = "to_pewter_pc" if in_pewter else "to_viridian_pc"
    leave_leg = "leave_pewter_pc" if in_pewter else "leave_viridian_pc"

    # Step 1: Reaching the PC
    # A learned "PC -> city" first if the agent is inside the other building
    other_pc = MAP_VIRIDIAN_POKECENTER if in_pewter else MAP_PEWTER_POKECENTER
    other_leave_leg = "leave_viridian_pc" if in_pewter else "leave_pewter_pc"
    other_city = MAP_VIRIDIAN_CITY if in_pewter else MAP_PEWTER_CITY
    if cur_map == other_pc:
        _execute_leg(other_leave_leg, other_city, [other_city])
        cur_map = int(env.observe()[1]["map_id"])

    if cur_map != target_pc:
        fallback_maps = [target_pc] if in_pewter else [MAP_VIRIDIAN_CITY, target_pc]
        _execute_leg(pc_leg, target_pc, fallback_maps)
        cur_map = int(env.observe()[1]["map_id"])

    # Fall back to BFS for anything unusual
    if cur_map != target_pc:
        r_nav = navigate_to(env, target_pc, budget=budget)
        presses += r_nav["presses"]
        cur_map = int(env.observe()[1]["map_id"])

    if cur_map != target_pc:
        return {
            "presses": presses,
            "success": False,
            "healed": False,
            "battles_started": battles_started,
            "battles_won": battles_won,
            "blackouts": blackouts,
            "moves_used": moves_used,
            "events": events,
        }

    # Step 2: Nurse Joy counter walk + dialogue
    h_res = heal_pkmn_center(env, budget=budget, exit_after=False)
    presses += h_res["presses"]
    healed = h_res.get("healed", False)
    final_hp = h_res.get("hp", 0)
    max_hp = h_res.get("max_hp", 0)

    # Step 3: Leaving PC via run_leg
    cur_map = int(env.observe()[1]["map_id"])
    if cur_map == target_pc:
        _execute_leg(leave_leg, target_city, [target_city])
        cur_map = int(env.observe()[1]["map_id"])
        if cur_map != target_city:
            r_exit = navigate_to(env, target_city, budget=budget)
            presses += r_exit["presses"]

    return {
        "presses": presses,
        "success": healed,
        "healed": healed,
        "hp": final_hp,
        "max_hp": max_hp,
        "battles_started": battles_started,
        "battles_won": battles_won,
        "blackouts": blackouts,
        "moves_used": moves_used,
        "events": events,
    }


def travel_to_pewter_skill(
    env: PokemonRed,
    budget: int = 4000,
    nav: str = "bfs",
    leg_policies: Optional[dict[str, str]] = None,
    run_legs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Navigates the multi-map route from Viridian/Route 1 all the way to Pewter City (map 2)."""
    presses = 0
    battles_won = 0
    battles_started = 0
    blackouts = 0
    moves_used: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    if nav == "bfs":
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

    # Learned navigation mode
    travel_legs = TRAVEL_LEGS
    active_policies = dict(DEFAULT_LEG_POLICIES)
    if leg_policies:
        active_policies.update(leg_policies)

    _, ram = env.observe()
    for leg_name, target_map, fallback_maps in travel_legs[first_travel_leg(int(ram["map_id"]), int(ram["y"])):]:
        if presses >= budget:
            break
        cur_map = int(env.observe()[1]["map_id"])
        if cur_map == MAP_PEWTER_CITY:
            break

        pol_path = active_policies.get(leg_name, DEFAULT_LEG_POLICIES.get(leg_name, ""))
        leg_res = run_leg(
            env=env,
            leg=leg_name,
            target_map=target_map,
            policy_path=pol_path,
            fallback_maps=fallback_maps,
            budget=min(1000, budget - presses),
        )
        leg_presses = leg_res["policy_presses"] + leg_res["fallback_presses"]
        presses += leg_presses
        battles_started += leg_res["battles"]
        blackouts += leg_res.get("blackouts", 0)
        events.extend(leg_res.get("events", []))
        for m_k, count in leg_res.get("moves_used", {}).items():
            moves_used[m_k] = moves_used.get(m_k, 0) + count

        if run_legs is not None:
            run_legs.append({
                "leg": leg_res["leg"],
                "start": leg_res["start"],
                "policy": leg_res["policy"],
                "reached_by_policy": leg_res["reached_by_policy"],
                "policy_presses": leg_res["policy_presses"],
                "fallback_presses": leg_res["fallback_presses"],
                "battles": leg_res["battles"],
            })

        if leg_res.get("blackouts", 0) > 0 or not leg_res["success"]:
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


def fight_gym_skill(
    env: PokemonRed,
    budget: int = 2500,
    nav: str = "bfs",
    leg_policies: Optional[dict[str, str]] = None,
    run_legs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
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
        if nav == "learned":
            active_policies = dict(DEFAULT_LEG_POLICIES)
            if leg_policies:
                active_policies.update(leg_policies)
            leg = leg_for_stage(MAP_PEWTER_GYM)
            pol_path = active_policies.get(leg, DEFAULT_LEG_POLICIES[leg])
            leg_res = run_leg(
                env=env,
                leg=leg,
                target_map=MAP_PEWTER_GYM,
                policy_path=pol_path,
                fallback_maps=[MAP_PEWTER_GYM],
                budget=min(1000, budget - presses),
            )
            leg_presses = leg_res["policy_presses"] + leg_res["fallback_presses"]
            presses += leg_presses
            battles_started += leg_res["battles"]
            blackouts += leg_res.get("blackouts", 0)
            events.extend(leg_res.get("events", []))
            for m_k, count in leg_res.get("moves_used", {}).items():
                moves_used[m_k] = moves_used.get(m_k, 0) + count

            if run_legs is not None:
                run_legs.append({
                    "leg": leg_res["leg"],
                    "start": leg_res["start"],
                    "policy": leg_res["policy"],
                    "reached_by_policy": leg_res["reached_by_policy"],
                    "policy_presses": leg_res["policy_presses"],
                    "fallback_presses": leg_res["fallback_presses"],
                    "battles": leg_res["battles"],
                })

            if not leg_res["success"] or leg_res.get("blackouts", 0) > 0:
                return {"presses": presses, "success": False, "badge_obtained": False}
        else:
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
        if nav == "learned":
            h_jr = heal_skill(env, budget=min(800, budget - presses), nav=nav, leg_policies=leg_policies, run_legs=run_legs)
            presses += h_jr["presses"]
            battles_started += h_jr.get("battles_started", 0)
            battles_won += h_jr.get("battles_won", 0)
            blackouts += h_jr.get("blackouts", 0)
            for m_k, count in h_jr.get("moves_used", {}).items():
                moves_used[m_k] = moves_used.get(m_k, 0) + count
            events.extend(h_jr.get("events", []))
            active_policies = dict(DEFAULT_LEG_POLICIES)
            if leg_policies:
                active_policies.update(leg_policies)
            leg = leg_for_stage(MAP_PEWTER_GYM)
            pol_path = active_policies.get(leg, DEFAULT_LEG_POLICIES[leg])
            leg_res = run_leg(
                env=env,
                leg=leg,
                target_map=MAP_PEWTER_GYM,
                policy_path=pol_path,
                fallback_maps=[MAP_PEWTER_GYM],
                budget=min(500, budget - presses),
            )
            leg_presses = leg_res["policy_presses"] + leg_res["fallback_presses"]
            presses += leg_presses
            battles_started += leg_res["battles"]
            blackouts += leg_res.get("blackouts", 0)
            events.extend(leg_res.get("events", []))
            for m_k, count in leg_res.get("moves_used", {}).items():
                moves_used[m_k] = moves_used.get(m_k, 0) + count
            if run_legs is not None:
                run_legs.append({
                    "leg": leg_res["leg"],
                    "start": leg_res["start"],
                    "policy": leg_res["policy"],
                    "reached_by_policy": leg_res["reached_by_policy"],
                    "policy_presses": leg_res["policy_presses"],
                    "fallback_presses": leg_res["fallback_presses"],
                    "battles": leg_res["battles"],
                })
        else:
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
    save_color: bool = False,
    state_path: str | Path = "saves/pokedex_viridian.state",
    work_dir: str | Path = "work/gym",
    traj_dir: str | Path = "work/gym_traj",
    heal_threshold: float = 0.35,
    nav: str = "bfs",
    leg_policies: Optional[dict[str, str]] = None,
    planner: str = "rules",
    llm_model: str = "gemma4:26b",
    llm_think: bool = True,
    llm_planner: Optional[Any] = None,
    max_llm_decisions: int = 60,
) -> dict[str, Any]:
    """Execute one full Gym Quest run from starter save state to Boulder Badge."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gif_frames: list[Image.Image] = []
    timeline: list[dict[str, Any]] = []
    maps_visited: list[int] = []
    moves_used_total: dict[str, int] = {}
    phase_presses: dict[str, int] = {}
    run_legs: list[dict[str, Any]] = []

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

    # --save-color: the stage-1 colour frame (lab/color) and the active navigation leg at every trajectory step
    palette = None
    traj_colors: list[np.ndarray] = []
    traj_legs: list[str] = []
    traj_bys: list[str] = []
    if save_color:
        from lab.color.native import pixel_tiles
        from lab.color.rules import build_tile_map, colorize_frame
        labels = json.loads(Path("lab/color/labels.json").read_text(encoding="utf-8"))
        keys = [f"{k.split(':')[0]}:{layer}:{k.split(':')[1]}" for k in labels for layer in (0, 1)]
        palette = {"classes": json.loads(Path("lab/color/palette.json").read_text(encoding="utf-8"))["classes"],
                   "tiles": build_tile_map(labels, keys)}

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
                if palette is not None:
                    traj_colors.append(colorize_frame(pixel_tiles(env.pyboy), palette))
                    traj_legs.append(getattr(env, "_nav_leg", ""))
                    traj_bys.append("" if ram_obs["in_battle"] else getattr(env, "_nav_by", ""))

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
            if nav == "learned":
                h_init = heal_skill(
                    env,
                    budget=500,
                    nav=nav,
                    leg_policies=leg_policies,
                    run_legs=run_legs,
                )
            else:
                h_init = heal_pkmn_center(env, budget=500)
            phase_presses["heal"] = phase_presses.get("heal", 0) + (total_presses - p_before)
            print(f"[GymQuest seed={seed}] Initial Viridian PC heal: {h_init['hp']}/{h_init['max_hp']} HP")

        effective_target_level = target_level
        consecutive_stalls = 0
        stopped_reason: Optional[str] = None
        if planner == "llm":
            if llm_planner is None:
                from lab.agent.llm_planner import LLMPlanner
                llm_planner = LLMPlanner(model=llm_model, think=llm_think)
            if hasattr(llm_planner, "warmup"):
                load_sec = llm_planner.warmup()
                print(f"[GymQuest seed={seed}] Model {llm_model} warmed up in {load_sec:.2f}s (load_seconds={load_sec})")

        while total_presses < budget:
            st = party_status(env)
            badges = int(env.pyboy.memory[WRAM_OBTAINED_BADGES])
            st["badges"] = badges

            # LLM Stall Guard: check decision cap
            if planner == "llm" and getattr(llm_planner, "total_decisions", 0) >= max_llm_decisions:
                stopped_reason = "decision_cap"
                print(f"[GymQuest seed={seed}] Stopped by LLM stall guard: decision_cap ({max_llm_decisions} decisions reached)")
                break

            if planner == "llm":
                st["presses_used"], st["press_budget"] = total_presses, budget  # the LLM must know the time limit
                goal_obj = llm_planner(
                    st,
                    target_level=effective_target_level,
                    pewter_healed=pewter_healed,
                    heal_threshold=heal_threshold,
                )
                goal = str(goal_obj)
                reason = goal_obj.reason
                new_target = getattr(goal_obj, "target_level", None)
                if new_target is not None and new_target != effective_target_level:
                    print(f"[LLMPlanner] Target level updated from {effective_target_level} to {new_target}")
                    effective_target_level = new_target
            else:
                goal, reason = plan_gym(
                    st,
                    target_level=effective_target_level,
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
            if planner == "llm":
                last_rec = getattr(llm_planner, "last_decision_record", {})
                log_entry.update({
                    "source": last_rec.get("source", "llm"),
                    "latency": last_rec.get("latency", 0.0),
                    "prompt_tokens": last_rec.get("prompt_tokens", 0),
                    "eval_tokens": last_rec.get("eval_tokens", 0),
                    "raw_output": last_rec.get("raw_output"),
                    "thinking": last_rec.get("thinking", ""),
                    "rejection_reason": last_rec.get("rejection_reason"),
                    "target_level": effective_target_level,
                })

            timeline.append(log_entry)
            print(f"[Planner t={elapsed:5.1f}s p={total_presses:5d}] Goal: {goal:<16} | Lv: {st['level']:2d} | HP: {st['hp']:2d}/{st['max_hp']:2d} | Map: {st['map_id']:2d} | {reason}")

            if goal == "done":
                badge_obtained = True
                break

            p_before = total_presses

            # Execute System 1 skill corresponding to goal
            if goal == "heal":
                if nav == "learned":
                    h_res = heal_skill(
                        env,
                        budget=min(800, budget - total_presses),
                        nav=nav,
                        leg_policies=leg_policies,
                        run_legs=run_legs,
                    )
                else:
                    h_res = heal_pkmn_center(env, budget=min(800, budget - total_presses))
                phase_presses["heal"] = phase_presses.get("heal", 0) + (total_presses - p_before)
                total_battles_started += h_res.get("battles_started", 0)
                total_battles_won += h_res.get("battles_won", 0)
                total_blackouts += h_res.get("blackouts", 0)
                for m_k, count in h_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count
                if int(env.observe()[1]["map_id"]) in (MAP_PEWTER_CITY, MAP_PEWTER_POKECENTER):
                    if h_res.get("healed", False):
                        pewter_healed = True

            elif goal == "grind":
                g_res = grind_skill(
                    env,
                    target_level=effective_target_level,
                    hp_threshold=heal_threshold,
                    budget=min(3000, budget - total_presses),
                    nav=nav,
                    leg_policies=leg_policies,
                    run_legs=run_legs,
                )
                phase_presses["grind"] = phase_presses.get("grind", 0) + (total_presses - p_before)
                total_battles_started += g_res.get("battles_started", 0)
                total_battles_won += g_res.get("battles_won", 0)
                total_blackouts += g_res.get("blackouts", 0)
                for m_k, count in g_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

            elif goal == "travel_to_pewter":
                t_res = travel_to_pewter_skill(
                    env,
                    budget=min(4500, budget - total_presses),
                    nav=nav,
                    leg_policies=leg_policies,
                    run_legs=run_legs,
                )
                phase_presses["travel_to_pewter"] = phase_presses.get("travel_to_pewter", 0) + (total_presses - p_before)
                total_battles_started += t_res.get("battles_started", 0)
                total_battles_won += t_res.get("battles_won", 0)
                total_blackouts += t_res.get("blackouts", 0)
                for m_k, count in t_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

            elif goal == "gym":
                if level_at_gym is None:
                    level_at_gym = party_status(env)["level"]
                gym_res = fight_gym_skill(
                    env,
                    budget=min(4000, budget - total_presses),
                    nav=nav,
                    leg_policies=leg_policies,
                    run_legs=run_legs,
                )
                phase_presses["gym"] = phase_presses.get("gym", 0) + (total_presses - p_before)
                total_battles_started += gym_res.get("battles_started", 0)
                total_battles_won += gym_res.get("battles_won", 0)
                total_blackouts += gym_res.get("blackouts", 0)
                for m_k, count in gym_res.get("moves_used", {}).items():
                    moves_used_total[m_k] = moves_used_total.get(m_k, 0) + count

                if gym_res.get("badge_obtained", False):
                    badge_obtained = True
                    break

            p_used = total_presses - p_before
            if total_presses == p_before:
                env.step("pass")
                p_used = total_presses - p_before

            if planner == "llm":
                if p_used <= 1:
                    consecutive_stalls += 1
                else:
                    consecutive_stalls = 0

                if hasattr(llm_planner, "record_outcome"):
                    st_after = party_status(env)
                    st_after["badges"] = int(env.pyboy.memory[WRAM_OBTAINED_BADGES])
                    llm_planner.record_outcome(
                        presses_used=p_used,
                        status_after=st_after,
                    )

                if consecutive_stalls >= 5:
                    stopped_reason = "stalled"
                    print(f"[GymQuest seed={seed}] Stopped by LLM stall guard: stalled (5 consecutive decisions used <= 1 press)")
                    break

        final_st = party_status(env)
        final_badges = int(env.pyboy.memory[WRAM_OBTAINED_BADGES])
        badge_obtained = bool(final_badges & 0x01)
        if planner == "llm" and stopped_reason is not None:
            badge_obtained = False

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
        if planner == "llm":
            if stopped_reason is not None:
                result["stopped_reason"] = stopped_reason
            result["planner"] = llm_planner.get_stats()
        else:
            result["planner"] = {
                "type": "rules",
                "model": None,
                "decisions": len(timeline),
                "llm_decisions": 0,
                "hallucinations": {},
                "fallbacks": 0,
                "mean_latency": 0.0,
                "max_latency": 0.0,
                "total_tokens": 0,
                "notebook": [],
            }
        if nav == "learned" or run_legs:
            result["legs"] = run_legs

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
            if traj_colors:
                color_path = traj_dir / f"run_seed_{seed}_color.npz"
                np.savez_compressed(color_path, color=np.array(traj_colors, dtype=np.uint8),
                                    map_ids=np.array(traj_maps, dtype=np.uint8), goals=np.array(traj_goals, dtype=str),
                                    in_battles=np.array(traj_in_battles, dtype=bool),
                                    leg=np.array(traj_legs, dtype=str), by=np.array(traj_bys, dtype=str))
                print(f"[GymQuest seed={seed}] Saved colour trajectory ({len(traj_colors)} steps) -> {color_path}")

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
    # LLM reasons can hold any Unicode (gemma4:26b wrote "≈"); the Windows console/log default (cp1252) crashed a run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="PokeDreamer System 1/2 Gym Quest.")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--seed", type=int, default=100, help="Initial random seed")
    parser.add_argument("--target-level", type=int, default=15, help="Target level before traveling to Pewter")
    parser.add_argument("--budget", type=int, default=18000, help="Total action budget per run")
    parser.add_argument("--work-dir", type=str, default="work/gym", help="Directory for output JSONs and GIFs")
    parser.add_argument("--state", type=str, default="saves/pokedex_viridian.state", help="Path to initial save state")
    parser.add_argument("--gif", action="store_true", help="Record and save captioned GIF of run")
    parser.add_argument("--save-traj", action="store_true", help="Record and save full trajectory npz to work/gym_traj/run_seed_<seed>.npz")
    parser.add_argument("--save-color", action="store_true",
                        help="with --save-traj: also save the stage-1 colour frames and the active leg to <traj-dir>/run_seed_<seed>_color.npz")
    parser.add_argument("--traj-dir", type=str, default="work/gym_traj", help="Directory for output trajectory npz files")
    parser.add_argument("--heal-threshold", type=float, default=0.35, help="HP fraction threshold to trigger healing")
    parser.add_argument("--nav", choices=["bfs", "learned"], default="bfs", help="Navigation mode (bfs or learned)")
    parser.add_argument("--leg-policies", type=str, default="", help="Override leg policies (leg=path,...)")
    parser.add_argument("--planner", choices=["rules", "llm"], default="rules", help="System 2 planner type (rules or llm)")
    parser.add_argument("--llm-model", type=str, default="gemma4:26b", help="Ollama model name for LLM planner")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking in LLM planner")
    parser.add_argument("--max-llm-decisions", type=int, default=60, help="Max LLM planner decisions before stopping run")
    args = parser.parse_args()
    if args.save_color and not args.save_traj:
        parser.error("--save-color requires --save-traj")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    policies_override = dict(DEFAULT_LEG_POLICIES)
    if args.leg_policies:
        for item in args.leg_policies.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                policies_override[k.strip()] = v.strip()

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
            save_color=args.save_color,
            state_path=args.state,
            work_dir=work_dir,
            traj_dir=args.traj_dir,
            heal_threshold=args.heal_threshold,
            nav=args.nav,
            leg_policies=policies_override,
            planner=args.planner,
            llm_model=args.llm_model,
            llm_think=not args.no_think,
            max_llm_decisions=args.max_llm_decisions,
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

    if any("legs" in r for r in results):
        legs_summary = {}
        for leg_name in [
            "leave_viridian_pc",
            "to_grass",
            "to_viridian_pc",
            "viridian",
            "to_route2",
            "to_gate",
            "forest",
            "to_pewter",
            "to_pewter_pc",
            "leave_pewter_pc",
            "to_gym",
        ]:
            att = sum(1 for r in results for l in r.get("legs", []) if l["leg"] == leg_name and l["reached_by_policy"] is not None)
            if att > 0:
                rec = sum(1 for r in results for l in r.get("legs", []) if l["leg"] == leg_name and l["reached_by_policy"])
                legs_summary[leg_name] = f"{rec}/{att}"
        summary["legs"] = legs_summary

    summary_file = work_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {summary['badge_rate']} Badges ({summary['badge_success_pct']}%)")
    print(f"Presses: mean={summary['presses_mean']}, min={summary['presses_min']}, max={summary['presses_max']}")
    print(f"Summary written to: {summary_file}")
    print("=" * 80 + "\n")

    if args.planner == "llm":
        try:
            from lab.agent.llm_planner import unload_model
            unload_model(model=args.llm_model)
            print(f"[LLMPlanner] Unloaded {args.llm_model} from Ollama.")
        except Exception as err:
            print(f"[LLMPlanner] Note: could not unload model {args.llm_model}: {err}")


if __name__ == "__main__":
    main()
