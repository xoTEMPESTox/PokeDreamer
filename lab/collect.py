"""Exploration data collection beyond Viridian City with parallel emulators (PokeDreamer).

Two-phase Go-Explore architecture:
- Phase A (Explore, unrecorded):
  Classic Go-Explore with short iterations (~100 steps) pushing the exploration frontier
  through Route 2, South Gate (50), Viridian Forest (51), North Gate (47), into Pewter City (2),
  Pewter Gym (54), and Pewter Pokémon Center (58), or up to --explore-cap steps per worker.
  Selection weight: 1 / sqrt(times_chosen + 1) with x2 boost for recently discovered cells.
  Low HP handling (<30% HP): restores from healthy archive cell (>=80% HP).
  Snapshot replacement: prefer higher party HP / fewer steps when a cell is re-found.
  Exploratory action policy: momentum directions (1-4 steps), bump / re-pick direction,
  5% A, 5% B, NEVER START, battle reflex (press A in battle). No hand weights or direction biases.
  Outputs work/collect_v2/explore.json with merged coverage stats.

- Phase B (Record):
  Records --episodes of 1,000 steps each in the exact dataset format:
  obs (144x160x3 uint8 screen_capture), actions, rewards (0.0), episode_starts, map_ids,
  xs, ys, facings, in_battles, dialog_opens, badges, party_hps, party_max_hps.
  Start cells sampled uniformly over maps in target set (all Phase A maps except Pallet/Route 1/Oak's lab),
  then uniformly over that map's cells in the archive.
  Writes 10 episodes per transitions_NNNN.npz to work/collect_v2/ and coverage.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.env import BUTTONS, ROM_PATH, PokemonRed
from src.game_state import extract_game_state, screen_capture
from src.ram_addresses import HP_ADDRESSES, MAX_HP_ADDRESSES, PARTY_SIZE

# ── Map Constants ─────────────────────────────────────────────────────────────
MAP_PALLET = 0
MAP_VIRIDIAN = 1
MAP_PEWTER = 2
MAP_ROUTE1 = 12
MAP_ROUTE2 = 13
MAP_ROUTE22 = 33
MAP_VIRIDIAN_POKECENTER = 41
MAP_VIRIDIAN_MART = 42
MAP_VIRIDIAN_SCHOOL = 43
MAP_VIRIDIAN_HOUSE = 44
MAP_VIRIDIAN_GYM = 45
MAP_DIGLETTS_CAVE = 46
MAP_VIRIDIAN_FOREST_NORTH_GATE = 47
MAP_ROUTE2_HOUSE = 48
MAP_ROUTE2_GATE = 49
MAP_VIRIDIAN_FOREST_SOUTH_GATE = 50
MAP_VIRIDIAN_FOREST = 51
MAP_PEWTER_MUSEUM_1F = 52
MAP_PEWTER_MUSEUM_2F = 53
MAP_PEWTER_GYM = 54
MAP_PEWTER_NIDORAN_HOUSE = 55
MAP_PEWTER_MART = 56
MAP_PEWTER_SPEECH_HOUSE = 57
MAP_PEWTER_POKECENTER = 58
MAP_ROUTE22_GATE = 193

TARGET_MAP_LABELS: Dict[int, str] = {
    MAP_VIRIDIAN: "Viridian City",
    MAP_ROUTE22: "Route 22",
    MAP_ROUTE2: "Route 2",
    MAP_VIRIDIAN_FOREST_SOUTH_GATE: "Viridian Forest South Gate",
    MAP_VIRIDIAN_FOREST: "Viridian Forest",
    MAP_VIRIDIAN_FOREST_NORTH_GATE: "Viridian Forest North Gate",
    MAP_PEWTER: "Pewter City",
    MAP_PEWTER_MUSEUM_1F: "Pewter Museum 1F",
    MAP_PEWTER_MUSEUM_2F: "Pewter Museum 2F",
    MAP_PEWTER_GYM: "Pewter Gym",
    MAP_PEWTER_NIDORAN_HOUSE: "Pewter Nidoran House",
    MAP_PEWTER_MART: "Pewter Mart",
    MAP_PEWTER_SPEECH_HOUSE: "Pewter Speech House",
    MAP_PEWTER_POKECENTER: "Pewter Pokémon Center",
    MAP_VIRIDIAN_POKECENTER: "Viridian Pokémon Center",
    MAP_VIRIDIAN_MART: "Viridian Mart",
    MAP_VIRIDIAN_SCHOOL: "Viridian School",
    MAP_VIRIDIAN_HOUSE: "Viridian House",
    MAP_ROUTE22_GATE: "Route 22 Gate",
}

# Maps excluded from Phase B start cell sampling
EXCLUDED_START_MAPS = {0, 12, 37, 38, 39, 40, 245, 246, 247, 248}

# Route ranking for reporting the furthest cell reached towards Pewter
ROUTE_RANK = {
    MAP_PALLET: 0,
    MAP_ROUTE1: 1,
    MAP_VIRIDIAN: 2,
    MAP_ROUTE22: 2,
    MAP_ROUTE22_GATE: 2,
    MAP_VIRIDIAN_MART: 2,
    MAP_VIRIDIAN_POKECENTER: 2,
    MAP_VIRIDIAN_SCHOOL: 2,
    MAP_VIRIDIAN_HOUSE: 2,
    MAP_ROUTE2: 3,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE: 4,
    MAP_VIRIDIAN_FOREST: 5,
    MAP_VIRIDIAN_FOREST_NORTH_GATE: 6,
    MAP_PEWTER: 7,
    MAP_PEWTER_MUSEUM_1F: 8,
    MAP_PEWTER_MUSEUM_2F: 8,
    MAP_PEWTER_NIDORAN_HOUSE: 8,
    MAP_PEWTER_MART: 8,
    MAP_PEWTER_SPEECH_HOUSE: 8,
    MAP_PEWTER_POKECENTER: 8,
    MAP_PEWTER_GYM: 9,
}

DEFAULT_STATE_PATH = Path("saves/pokedex_viridian.state")


def cell_key(map_id: int, x: int, y: int) -> Tuple[int, int, int]:
    """Compute archive cell key (map_id, x // 3, y // 3)."""
    return (int(map_id), int(x) // 3, int(y) // 3)


@dataclass
class ArchiveEntry:
    snapshot: bytes
    times_chosen: int = 0
    map_id: int = 0
    x: int = 0
    y: int = 0
    hp_frac: float = 1.0
    first_iter: int = 0
    steps: int = 0
    visit_count: int = 0
    first_episode: int = 0

    def __post_init__(self):
        if self.visit_count > 0 and self.times_chosen == 0:
            self.times_chosen = self.visit_count
        elif self.times_chosen > 0 and self.visit_count == 0:
            self.visit_count = self.times_chosen
        if self.first_episode > 0 and self.first_iter == 0:
            self.first_iter = self.first_episode
        elif self.first_iter > 0 and self.first_episode == 0:
            self.first_episode = self.first_iter


def get_party_hp_frac(env: PokemonRed) -> float:
    """Read starter HP fraction from RAM."""
    mem = env.pyboy.memory
    party_sz = mem[PARTY_SIZE]
    if party_sz == 0:
        return 0.0
    cur_hp = (mem[HP_ADDRESSES[0]] << 8) | mem[HP_ADDRESSES[0] + 1]
    max_hp = (mem[MAX_HP_ADDRESSES[0]] << 8) | mem[MAX_HP_ADDRESSES[0] + 1]
    return (cur_hp / max_hp) if max_hp > 0 else 0.0


def choose_exploratory_action(
    ram: Dict[str, Any],
    prev_pos: Optional[Tuple[int, int, int]],
    cur_dir: int,
    steps_left: int,
    rng: np.random.Generator,
    prev_act: int = -1,
) -> Tuple[int, int, int]:
    """Exploratory action policy adhering to Go-Explore specifications:

    - Battle reflex: press A (4) while in_battle.
    - Directional momentum: directions held for 1-4 steps.
    - Bump / re-pick: when a movement attempt did not change position, re-pick a different direction.
    - Occasional A (5%) to talk / advance dialogue and B (5%) to cancel.
    - NEVER START (6).
    - No per-map hand weights and no direction biases.
    Returns: (action, next_dir, next_steps_left).
    """
    if ram["in_battle"]:
        return 4, cur_dir, 0  # Action 4 (A) battle reflex

    cur_pos = (ram["map_id"], ram["x"], ram["y"])
    bumped = (prev_act in (0, 1, 2, 3) and prev_pos == cur_pos)
    if bumped:
        steps_left = 0

    if steps_left > 0:
        return cur_dir, cur_dir, steps_left - 1

    # Decide new action
    roll = rng.random()
    if roll < 0.05:
        return 4, cur_dir, 0  # A
    elif roll < 0.10:
        return 5, cur_dir, 0  # B
    else:
        if bumped:
            choices = [d for d in [0, 1, 2, 3] if d != cur_dir]
            cur_dir = int(rng.choice(choices))
        else:
            cur_dir = int(rng.integers(0, 4))
        duration = int(rng.integers(1, 5))  # 1 to 4 steps
        return cur_dir, cur_dir, duration - 1


def step_fast(env: PokemonRed, action: int) -> None:
    """Execute action without rendering host frames for unrecorded exploration."""
    btn = BUTTONS[action]
    if btn != "pass":
        env.pyboy.button(btn, 8)
    env.pyboy.tick(24, render=False)


def select_explore_cell(
    archive: Dict[Tuple[int, int, int], ArchiveEntry],
    rng: np.random.Generator,
    cur_iter: int,
    prefer_healthy: bool = False,
) -> Tuple[int, int, int]:
    """Select a cell from the archive favoring rarely chosen cells and recent discoveries.

    HP policy: If prefer_healthy is True (e.g. party HP dropped below 30% or blackout),
    candidate cells are filtered to those with hp_frac >= 0.8 (or highest available).
    Weight: 1 / sqrt(times_chosen + 1), multiplied by 2 for cells discovered within last 20 iters.
    """
    candidates = list(archive.keys())
    if prefer_healthy:
        healthy = [c for c in candidates if archive[c].hp_frac >= 0.8]
        if healthy:
            candidates = healthy
        else:
            max_hp = max(archive[c].hp_frac for c in candidates)
            candidates = [c for c in candidates if archive[c].hp_frac >= max_hp - 0.05]

    weights = []
    for c in candidates:
        entry = archive[c]
        w = 1.0 / np.sqrt(entry.times_chosen + 1)
        if (cur_iter - entry.first_iter) < 20:
            w *= 2.0
        weights.append(w)

    tot = sum(weights)
    if tot <= 0:
        p = [1.0 / len(candidates)] * len(candidates)
    else:
        p = [w / tot for w in weights]

    chosen_idx = rng.choice(len(candidates), p=p)
    return candidates[chosen_idx]


def initialize_worker_archive(
    env: PokemonRed,
    archive: Dict[Tuple[int, int, int], ArchiveEntry],
    start_state_bytes: Optional[bytes] = None,
    start_states_path: Optional[Path | str] = None,
    no_default_start: bool = False,
) -> None:
    """Seed the archive with start states.

    If start_states_path is given, loads entries from index.json, keys them by (map_id, x // 3, y // 3),
    computes hp_frac from hp / max_hp, and adds them to archive.
    If not no_default_start, also adds the default start state and runs the Viridian Pokecenter healing sequence.
    """
    if start_states_path is not None:
        p_idx = Path(start_states_path)
        if not p_idx.exists():
            raise FileNotFoundError(f"Start states index not found: {p_idx}")
        with open(p_idx, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            f_path = Path(entry["file"])
            if not f_path.is_absolute() and not f_path.exists():
                f_path = p_idx.parent / f_path.name
            if not f_path.exists():
                f_path = Path.cwd() / entry["file"]
            if not f_path.exists():
                continue
            snapshot_bytes = f_path.read_bytes()
            m_id = int(entry["map_id"])
            x_pos = int(entry["x"])
            y_pos = int(entry["y"])
            hp = float(entry.get("hp", 0))
            max_hp = float(entry.get("max_hp", 0))
            hp_frac = (hp / max_hp) if max_hp > 0 else 1.0

            ck = cell_key(m_id, x_pos, y_pos)
            if ck not in archive or hp_frac > archive[ck].hp_frac:
                archive[ck] = ArchiveEntry(
                    snapshot=snapshot_bytes,
                    times_chosen=0,
                    map_id=m_id,
                    x=x_pos,
                    y=y_pos,
                    hp_frac=hp_frac,
                    first_iter=0,
                    steps=0,
                    visit_count=0,
                    first_episode=0,
                )

    if not no_default_start and start_state_bytes is not None:
        # 1. Start state (Viridian Mart exterior)
        env.restore(start_state_bytes)
        st0 = extract_game_state(env.pyboy)
        hp_frac0 = get_party_hp_frac(env)
        c0 = cell_key(st0.map_id, st0.x, st0.y)
        if c0 not in archive or hp_frac0 > archive[c0].hp_frac:
            archive[c0] = ArchiveEntry(
                snapshot=env.snapshot(),
                times_chosen=0,
                map_id=st0.map_id,
                x=st0.x,
                y=st0.y,
                hp_frac=hp_frac0,
                first_iter=0,
                steps=0,
                visit_count=0,
                first_episode=0,
            )

        # 2. Heal at Pokémon Center and record full-HP state
        try:
            path_to_pc = ["left"] * 10 + ["down"] * 6 + ["right"] * 4 + ["up", "up"]
            for a in path_to_pc:
                env.step(a)
            for _ in range(3):
                env.step("pass")
            # Walk up to Nurse Joy counter at (3, 3)
            for _ in range(4):
                env.step("up")
            # Clear dialogue
            for _ in range(10):
                env.step("a")
            # Step down off counter
            for _ in range(20):
                _, r = env.step("down")
                if (r["x"], r["y"]) != (3, 3):
                    break
                env.step("b")
            while r["map_id"] == MAP_VIRIDIAN_POKECENTER:
                _, r = env.step("down")
            for _ in range(5):
                _, r = env.step("pass")

            # Now outside PC at (23, 26) with full HP
            st_healed = extract_game_state(env.pyboy)
            hp_healed = get_party_hp_frac(env)
            c_healed = cell_key(st_healed.map_id, st_healed.x, st_healed.y)
            if c_healed not in archive or hp_healed > archive[c_healed].hp_frac:
                archive[c_healed] = ArchiveEntry(
                    snapshot=env.snapshot(),
                    times_chosen=0,
                    map_id=st_healed.map_id,
                    x=st_healed.x,
                    y=st_healed.y,
                    hp_frac=hp_healed,
                    first_iter=0,
                    steps=0,
                    visit_count=0,
                    first_episode=0,
                )
        except Exception:
            pass

    # Ensure env is loaded with a valid snapshot from the archive
    if env is not None and archive:
        first_entry = next(iter(archive.values()))
        env.restore(first_entry.snapshot)


def run_phase_a_explore(
    env: PokemonRed,
    archive: Dict[Tuple[int, int, int], ArchiveEntry],
    rng: np.random.Generator,
    worker_id: int,
    explore_cap: int = 400000,
    iter_length: int = 100,
    target_maps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Execute Phase A Go-Explore loop until targets are reached or explore_cap steps taken."""
    total_steps = 0
    cur_iter = 0
    maps_first_iter: Dict[int, int] = {}
    for entry in archive.values():
        if entry.map_id not in maps_first_iter:
            maps_first_iter[entry.map_id] = 0

    stop_reason = "explore_cap_reached"
    low_hp_flag = False

    t0_explore = time.perf_counter()

    while total_steps < explore_cap:
        cur_iter += 1

        # Select cell
        chosen_c = select_explore_cell(archive, rng, cur_iter=cur_iter, prefer_healthy=low_hp_flag)
        archive[chosen_c].times_chosen += 1
        archive[chosen_c].visit_count += 1
        env.restore(archive[chosen_c].snapshot)
        base_steps = archive[chosen_c].steps

        cur_dir = int(rng.integers(0, 4))
        steps_left = 0
        prev_pos = None
        prev_act = -1

        for step_i in range(iter_length):
            total_steps += 1
            st = extract_game_state(env.pyboy)
            mid = st.map_id

            if mid not in maps_first_iter:
                maps_first_iter[mid] = cur_iter
                m_label = TARGET_MAP_LABELS.get(mid, f"Map_{mid}")
                print(
                    f"[Worker {worker_id}] Phase A iter {cur_iter} ({total_steps:,} steps): "
                    f"FIRST REACHED {m_label} ({mid}) at ({st.x}, {st.y})! Total maps: {len(maps_first_iter)}",
                    flush=True,
                )

            # Update archive on peaceful overworld
            if not st.in_battle:
                ck = cell_key(mid, st.x, st.y)
                hp_f = get_party_hp_frac(env)
                path_steps = base_steps + step_i + 1

                if ck not in archive:
                    archive[ck] = ArchiveEntry(
                        snapshot=env.snapshot(),
                        times_chosen=0,
                        map_id=mid,
                        x=st.x,
                        y=st.y,
                        hp_frac=hp_f,
                        first_iter=cur_iter,
                        steps=path_steps,
                        visit_count=0,
                        first_episode=0,
                    )
                else:
                    old = archive[ck]
                    if hp_f > old.hp_frac or (hp_f == old.hp_frac and path_steps < old.steps):
                        old.snapshot = env.snapshot()
                        old.hp_frac = hp_f
                        old.steps = path_steps

            # Choose exploratory action
            ram_dict = {
                "map_id": st.map_id,
                "x": st.x,
                "y": st.y,
                "facing": st.facing,
                "in_battle": st.in_battle,
                "dialog_open": st.dialog_open,
            }
            act, cur_dir, steps_left = choose_exploratory_action(
                ram_dict, prev_pos, cur_dir, steps_left, rng, prev_act=prev_act
            )
            prev_pos = (st.map_id, st.x, st.y)
            prev_act = act

            # Fast step without host frame rendering
            step_fast(env, act)

        # Check ending HP of this iteration
        end_hp_frac = get_party_hp_frac(env)
        low_hp_flag = (end_hp_frac < 0.30)

        # Check stopping targets
        maps_in_archive = {c[0] for c in archive.keys()}
        if target_maps is not None:
            if set(target_maps).issubset(maps_in_archive):
                stop_reason = "targets_reached"
                print(
                    f"[Worker {worker_id}] Phase A SUCCESS: reached all {len(target_maps)} "
                    f"target maps in {total_steps:,} steps ({cur_iter} iters)!",
                    flush=True,
                )
                break
        else:
            if MAP_PEWTER_GYM in maps_in_archive and MAP_PEWTER_POKECENTER in maps_in_archive:
                stop_reason = "targets_reached"
                print(
                    f"[Worker {worker_id}] Phase A SUCCESS: reached both Pewter Gym (54) and "
                    f"Pewter Center (58) in {total_steps:,} steps ({cur_iter} iters)!",
                    flush=True,
                )
                break

    dt_explore = time.perf_counter() - t0_explore
    sps_explore = total_steps / dt_explore if dt_explore > 0 else 0.0

    # Determine furthest cell reached along route
    best_rank = -1
    furthest_cell_info = (1, 0, 0)
    for c, entry in archive.items():
        m = c[0]
        rank = ROUTE_RANK.get(m, 2)
        if rank > best_rank:
            best_rank = rank
            furthest_cell_info = (entry.map_id, entry.x, entry.y)
        elif rank == best_rank:
            if entry.y < furthest_cell_info[2]:
                furthest_cell_info = (entry.map_id, entry.x, entry.y)

    cells_per_map: Dict[int, int] = {}
    for c in archive.keys():
        m = c[0]
        cells_per_map[m] = cells_per_map.get(m, 0) + 1

    worker_unreached = [m for m in target_maps if m not in cells_per_map] if target_maps is not None else []

    print(
        f"[Worker {worker_id}] Phase A completed ({stop_reason}): {total_steps:,} steps, "
        f"{cur_iter} iters in {dt_explore:.1f}s ({sps_explore:.1f} st/s). "
        f"Archive: {len(archive)} cells across {len(cells_per_map)} maps. "
        f"Furthest cell: {furthest_cell_info}",
        flush=True,
    )
    if worker_unreached:
        print(f"[Worker {worker_id}] Target maps NOT reached in Phase A: {worker_unreached}", flush=True)

    targets_reached_flag = (
        set(target_maps).issubset(set(cells_per_map.keys()))
        if target_maps is not None
        else (MAP_PEWTER_GYM in cells_per_map and MAP_PEWTER_POKECENTER in cells_per_map)
    )

    return {
        "worker_id": worker_id,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "total_iters": cur_iter,
        "time_s": dt_explore,
        "steps_per_sec": sps_explore,
        "maps_first_iter": {str(k): v for k, v in maps_first_iter.items()},
        "cells_per_map": {str(k): v for k, v in cells_per_map.items()},
        "furthest_cell": list(furthest_cell_info),
        "targets_reached": targets_reached_flag,
        "unreached_targets": worker_unreached,
    }


def run_episode(
    env: PokemonRed,
    archive: Dict[Tuple[int, int, int], ArchiveEntry],
    rng: np.random.Generator,
    episode_idx: int = 0,
    episode_length: int = 1000,
    target_maps: Optional[List[int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run a single 1,000-step exploration episode and record transitions in dataset format.

    Start cell selection:
    Uniformly sampled over maps in target_maps (only those reached in Phase A),
    then uniformly sampled over that map's cells in the archive.
    """
    avail_maps = {c[0] for c in archive.keys()}
    if target_maps is not None:
        valid_targets = [m for m in target_maps if m in avail_maps]
        if not valid_targets:
            valid_targets = sorted(list(avail_maps))
    else:
        filtered = [m for m in sorted(list(avail_maps)) if m not in EXCLUDED_START_MAPS]
        valid_targets = filtered if filtered else sorted(list(avail_maps))

    # 1. Sample start map uniformly
    chosen_map = int(rng.choice(valid_targets))
    map_cells = [c for c in archive.keys() if c[0] == chosen_map]
    if not map_cells:
        map_cells = list(archive.keys())
    chosen_cell = map_cells[rng.integers(0, len(map_cells))]

    archive[chosen_cell].times_chosen += 1
    archive[chosen_cell].visit_count += 1
    env.restore(archive[chosen_cell].snapshot)

    # Pre-allocate arrays for exact dataset format
    obs_arr = np.empty((episode_length, 144, 160, 3), dtype=np.uint8)
    actions_arr = np.empty((episode_length,), dtype=np.uint8)
    rewards_arr = np.zeros((episode_length,), dtype=np.float32)
    episode_starts_arr = np.zeros((episode_length,), dtype=bool)
    episode_starts_arr[0] = True

    map_ids_arr = np.empty((episode_length,), dtype=np.uint8)
    xs_arr = np.empty((episode_length,), dtype=np.uint8)
    ys_arr = np.empty((episode_length,), dtype=np.uint8)
    facings_arr = np.empty((episode_length,), dtype=np.uint8)
    in_battles_arr = np.empty((episode_length,), dtype=bool)
    dialog_opens_arr = np.empty((episode_length,), dtype=bool)
    badges_arr = np.empty((episode_length,), dtype=np.uint8)
    party_hps_arr = np.empty((episode_length, 6), dtype=np.int16)
    party_max_hps_arr = np.empty((episode_length, 6), dtype=np.int16)

    cur_dir = int(rng.integers(0, 4))
    steps_left = 0
    prev_pos = None
    prev_act = -1

    maps_seen_this_ep: Set[int] = set()
    battle_steps = 0
    blackouts = 0
    was_in_battle = False

    for step_i in range(episode_length):
        # 1. Capture screen & state before action is taken
        raw_frame = screen_capture(env.pyboy)
        st = extract_game_state(env.pyboy)

        obs_arr[step_i] = raw_frame
        map_ids_arr[step_i] = st.map_id
        xs_arr[step_i] = st.x
        ys_arr[step_i] = st.y
        facings_arr[step_i] = st.facing
        in_battles_arr[step_i] = st.in_battle
        dialog_opens_arr[step_i] = st.dialog_open
        badges_arr[step_i] = st.badges

        hps = list(st.party_hp) + [0] * (6 - len(st.party_hp))
        max_hps = list(st.party_max_hp) + [0] * (6 - len(st.party_max_hp))
        party_hps_arr[step_i] = hps
        party_max_hps_arr[step_i] = max_hps

        if st.in_battle:
            battle_steps += 1
            was_in_battle = True

        maps_seen_this_ep.add(st.map_id)

        # Check for blackout
        if was_in_battle and not st.in_battle:
            starter_hp = st.party_hp[0] if st.party_hp else 0
            if starter_hp == 0 or st.map_id == MAP_VIRIDIAN_POKECENTER:
                blackouts += 1
            was_in_battle = False

        # Add newly reached cells to archive if peaceful overworld
        if not st.in_battle:
            ck = cell_key(st.map_id, st.x, st.y)
            hp_frac = (st.party_hp[0] / st.party_max_hp[0]) if (st.party_hp and st.party_max_hp and st.party_max_hp[0] > 0) else 0.0
            if ck not in archive:
                archive[ck] = ArchiveEntry(
                    snapshot=env.snapshot(),
                    times_chosen=0,
                    map_id=st.map_id,
                    x=st.x,
                    y=st.y,
                    hp_frac=hp_frac,
                    first_iter=0,
                    steps=step_i + 1,
                    visit_count=0,
                    first_episode=episode_idx,
                )
            elif hp_frac > archive[ck].hp_frac:
                archive[ck].snapshot = env.snapshot()
                archive[ck].hp_frac = hp_frac

        # 2. Decide action
        ram_dict = {
            "map_id": st.map_id,
            "x": st.x,
            "y": st.y,
            "facing": st.facing,
            "in_battle": st.in_battle,
            "dialog_open": st.dialog_open,
        }
        act, cur_dir, steps_left = choose_exploratory_action(
            ram_dict, prev_pos, cur_dir, steps_left, rng, prev_act=prev_act
        )
        actions_arr[step_i] = act
        prev_pos = (st.map_id, st.x, st.y)
        prev_act = act

        # 3. Advance emulator
        env.step(act)

    ep_data = {
        "obs": obs_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "episode_starts": episode_starts_arr,
        "map_ids": map_ids_arr,
        "xs": xs_arr,
        "ys": ys_arr,
        "facings": facings_arr,
        "in_battles": in_battles_arr,
        "dialog_opens": dialog_opens_arr,
        "badges": badges_arr,
        "party_hps": party_hps_arr,
        "party_max_hps": party_max_hps_arr,
    }

    ep_stats = {
        "start_map": chosen_map,
        "maps_seen": list(maps_seen_this_ep),
        "battle_steps": battle_steps,
        "blackouts": blackouts,
        "archive_size": len(archive),
    }

    return ep_data, ep_stats


def worker_proc(
    worker_id: int,
    num_episodes: int,
    explore_cap: int,
    base_seed: int,
    rom_path: str,
    state_path: str,
    out_dir: str,
    save_interval: int,
    result_queue: mp.Queue,
    start_states_path: Optional[str] = None,
    no_default_start: bool = False,
    target_maps: Optional[List[int]] = None,
) -> None:
    """Worker process running in CPU parallel: runs Phase A, then Phase B, writes chunks."""
    seed = base_seed + worker_id * 1000
    rng = np.random.default_rng(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_state_bytes = None
    if not no_default_start and state_path:
        p_state = Path(state_path)
        if p_state.exists():
            with open(p_state, "rb") as f:
                start_state_bytes = f.read()

    env = PokemonRed(rom=rom_path)
    archive: Dict[Tuple[int, int, int], ArchiveEntry] = {}
    initialize_worker_archive(
        env=env,
        archive=archive,
        start_state_bytes=start_state_bytes,
        start_states_path=start_states_path,
        no_default_start=no_default_start,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: Explore (unrecorded)
    # ══════════════════════════════════════════════════════════════════════════
    phase_a_stats = run_phase_a_explore(
        env=env,
        archive=archive,
        rng=rng,
        worker_id=worker_id,
        explore_cap=explore_cap,
        iter_length=100,
        target_maps=target_maps,
    )

    # Save worker's Phase A explore summary to disk
    worker_explore_file = out_path / f"_explore_worker_{worker_id}.json"
    worker_explore_file.write_text(json.dumps(phase_a_stats, indent=2))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: Record (1,000 steps per episode)
    # ══════════════════════════════════════════════════════════════════════════
    avail_maps = sorted(list(set(c[0] for c in archive.keys())))
    if target_maps is not None:
        phase_b_maps = [m for m in target_maps if m in avail_maps]
        worker_unreached = [m for m in target_maps if m not in avail_maps]
        if worker_unreached:
            print(
                f"[Worker {worker_id}] Target maps NOT reached in Phase A: {worker_unreached}",
                flush=True,
            )
        if not phase_b_maps:
            phase_b_maps = avail_maps
    else:
        filtered = [m for m in avail_maps if m not in EXCLUDED_START_MAPS]
        phase_b_maps = filtered if filtered else avail_maps

    print(
        f"[Worker {worker_id}] Phase B starting: {num_episodes} episodes over "
        f"{len(phase_b_maps)} target maps: {phase_b_maps}",
        flush=True,
    )

    maps_first_reached: Dict[int, int] = {}
    total_steps_per_map: Dict[int, int] = {}
    total_battle_steps = 0
    total_blackouts = 0

    buffered_episodes: List[Dict[str, np.ndarray]] = []
    chunk_counter = 0

    t_record_start = time.perf_counter()

    for ep_i in range(num_episodes):
        t0 = time.perf_counter()
        ep_data, ep_stats = run_episode(
            env=env,
            archive=archive,
            rng=rng,
            episode_idx=ep_i,
            episode_length=1000,
            target_maps=phase_b_maps,
        )
        dt = time.perf_counter() - t0
        sps = 1000.0 / dt if dt > 0 else 0.0

        buffered_episodes.append(ep_data)

        # Track maps and stats
        for m in ep_stats["maps_seen"]:
            if m not in maps_first_reached:
                maps_first_reached[m] = ep_i
                m_label = TARGET_MAP_LABELS.get(m, f"Map_{m}")
                print(
                    f"[Worker {worker_id}] Episode {ep_i + 1}/{num_episodes}: "
                    f"First reached {m_label} ({m})!",
                    flush=True,
                )

        # Step counts per map
        for m_id, count in zip(*np.unique(ep_data["map_ids"], return_counts=True)):
            total_steps_per_map[int(m_id)] = total_steps_per_map.get(int(m_id), 0) + int(count)

        total_battle_steps += ep_stats["battle_steps"]
        total_blackouts += ep_stats["blackouts"]

        start_map_label = TARGET_MAP_LABELS.get(ep_stats["start_map"], f"Map_{ep_stats['start_map']}")
        print(
            f"[Worker {worker_id}] Ep {ep_i + 1}/{num_episodes} done | "
            f"Start: {start_map_label} | Maps seen: {ep_stats['maps_seen']} | "
            f"Speed: {sps:.1f} steps/s",
            flush=True,
        )

        # Save chunk when save_interval is met
        if len(buffered_episodes) >= save_interval:
            chunk_counter += 1
            chunk_file = out_path / f"_worker_{worker_id}_chunk_{chunk_counter:04d}.npz"
            stacked = {
                k: np.concatenate([ep[k] for ep in buffered_episodes], axis=0)
                for k in buffered_episodes[0].keys()
            }
            np.savez_compressed(chunk_file, **stacked)
            buffered_episodes.clear()

    # Save leftover episodes
    if buffered_episodes:
        leftover_file = out_path / f"_worker_{worker_id}_leftover.npz"
        stacked = {
            k: np.concatenate([ep[k] for ep in buffered_episodes], axis=0)
            for k in buffered_episodes[0].keys()
        }
        np.savez_compressed(leftover_file, **stacked)
        buffered_episodes.clear()

    env.close()

    total_record_time = time.perf_counter() - t_record_start
    total_record_steps = num_episodes * 1000

    cells_per_map: Dict[int, int] = {}
    for c in archive.keys():
        m = c[0]
        cells_per_map[m] = cells_per_map.get(m, 0) + 1

    summary = {
        "worker_id": worker_id,
        "phase_a": phase_a_stats,
        "phase_b": {
            "episodes": num_episodes,
            "total_steps": total_record_steps,
            "total_time": total_record_time,
            "steps_per_sec": total_record_steps / total_record_time if total_record_time > 0 else 0.0,
            "maps_first_reached": maps_first_reached,
            "steps_per_map": total_steps_per_map,
            "cells_per_map": cells_per_map,
            "battle_steps": total_battle_steps,
            "blackouts": total_blackouts,
            "archive_size": len(archive),
        },
    }
    result_queue.put(summary)


def merge_and_finalize_chunks(out_dir: Path, save_interval_episodes: int = 10) -> int:
    """Consolidate worker chunk files into standard transitions_NNNN.npz files."""
    chunk_files = sorted(list(out_dir.glob("_worker_*_chunk_*.npz")))
    leftover_files = sorted(list(out_dir.glob("_worker_*_leftover.npz")))

    final_chunks: List[Path] = []
    chunk_idx = 1

    # Rename complete worker chunks
    for cf in chunk_files:
        target = out_dir / f"transitions_{chunk_idx:04d}.npz"
        if cf != target:
            if target.exists():
                target.unlink()
            cf.rename(target)
        final_chunks.append(target)
        chunk_idx += 1

    # Merge leftover files into 10-episode chunks
    if leftover_files:
        leftover_eps: List[Dict[str, np.ndarray]] = []
        for lf in leftover_files:
            with np.load(lf) as data:
                n_steps = len(data["obs"])
                n_eps = n_steps // 1000
                for i in range(n_eps):
                    sl = slice(i * 1000, (i + 1) * 1000)
                    ep_dict = {k: np.array(data[k][sl]) for k in data.files}
                    leftover_eps.append(ep_dict)
            try:
                lf.unlink()
            except Exception:
                pass

        while leftover_eps:
            batch = leftover_eps[:save_interval_episodes]
            leftover_eps = leftover_eps[save_interval_episodes:]
            target = out_dir / f"transitions_{chunk_idx:04d}.npz"
            stacked = {
                k: np.concatenate([ep[k] for ep in batch], axis=0)
                for k in batch[0].keys()
            }
            np.savez_compressed(target, **stacked)
            final_chunks.append(target)
            chunk_idx += 1

    # Remove temporary worker explore files
    for wf in out_dir.glob("_explore_worker_*.json"):
        try:
            wf.unlink()
        except Exception:
            pass

    return len(final_chunks)


def collect(
    rom_path: Path = ROM_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    out_dir: Path = Path("work/collect_v2"),
    workers: int = 8,
    episodes: int = 300,
    explore_cap: int = 400000,
    save_interval: int = 10,
    base_seed: int = 42,
    start_states_path: Optional[Path | str] = None,
    no_default_start: bool = False,
    target_maps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Main two-phase data collection entrypoint."""
    workers = min(workers, 8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribute episodes across workers
    ep_per_worker = [episodes // workers] * workers
    for i in range(episodes % workers):
        ep_per_worker[i] += 1

    active_workers = [(w, ep) for w, ep in enumerate(ep_per_worker) if ep > 0]
    num_active = len(active_workers)

    print("=" * 80)
    print("POKEDREAMER TWO-PHASE EXPLORATION DATA COLLECTION")
    print(f"Workers: {num_active} parallel emulators on CPU")
    print(f"Phase A Cap: {explore_cap:,} exploration steps per worker")
    print(f"Phase B Target: {episodes} recorded episodes (={episodes * 1000:,} steps)")
    print(f"Start State: {state_path}")
    print(f"Start States Index: {start_states_path}")
    print(f"No Default Start: {no_default_start}")
    print(f"Target Maps: {target_maps}")
    print(f"Output Directory: {out_dir.resolve()}")
    print("=" * 80)

    t_start = time.perf_counter()
    result_queue: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []

    for w_id, n_ep in active_workers:
        p = mp.Process(
            target=worker_proc,
            args=(
                w_id,
                n_ep,
                explore_cap,
                base_seed,
                str(rom_path),
                str(state_path) if state_path is not None else "",
                str(out_dir),
                save_interval,
                result_queue,
                str(start_states_path) if start_states_path is not None else None,
                no_default_start,
                target_maps,
            ),
        )
        procs.append(p)
        p.start()

    # Collect stats with timeout
    worker_summaries: List[Dict[str, Any]] = []
    for _ in range(num_active):
        try:
            # 60 minute timeout to allow full exploration
            summary = result_queue.get(timeout=3600)
            worker_summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] Timed out waiting for worker summary: {e}")
            break

    # Join worker processes
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Consolidate chunks into transitions_NNNN.npz
    num_chunks = merge_and_finalize_chunks(out_dir, save_interval_episodes=save_interval)
    total_wall_time = time.perf_counter() - t_start

    # ══════════════════════════════════════════════════════════════════════════
    # MERGE PHASE A STATS -> explore.json
    # ══════════════════════════════════════════════════════════════════════════
    phase_a_steps_per_worker = {}
    phase_a_stop_reasons = {}
    phase_a_cells_per_map: Dict[str, int] = {}
    phase_a_first_iter: Dict[str, int] = {}
    phase_a_furthest_cells = {}
    targets_reached_per_worker = {}

    for s in worker_summaries:
        w_id = str(s["worker_id"])
        pa = s["phase_a"]
        phase_a_steps_per_worker[w_id] = pa["total_steps"]
        phase_a_stop_reasons[w_id] = pa["stop_reason"]
        phase_a_furthest_cells[w_id] = pa["furthest_cell"]
        targets_reached_per_worker[w_id] = pa["targets_reached"]

        for m_str, c_count in pa["cells_per_map"].items():
            phase_a_cells_per_map[m_str] = phase_a_cells_per_map.get(m_str, 0) + c_count

        for m_str, it_val in pa["maps_first_iter"].items():
            if m_str not in phase_a_first_iter or it_val < phase_a_first_iter[m_str]:
                phase_a_first_iter[m_str] = it_val

    all_explore_maps = sorted([int(m) for m in phase_a_first_iter.keys()])

    if target_maps is not None:
        all_unreached_targets = [m for m in target_maps if m not in all_explore_maps]
        all_reached_targets = [m for m in target_maps if m in all_explore_maps]
    else:
        all_unreached_targets = []
        all_reached_targets = []

    maps_to_report = target_maps if target_maps is not None else list(TARGET_MAP_LABELS.keys())
    target_maps_status = {
        TARGET_MAP_LABELS.get(mid, f"Map_{mid}"): (mid in all_explore_maps)
        for mid in maps_to_report
    }

    explore_report = {
        "steps_per_worker": phase_a_steps_per_worker,
        "stopping_reasons": phase_a_stop_reasons,
        "cells_per_map": phase_a_cells_per_map,
        "first_iter_per_map": phase_a_first_iter,
        "furthest_cells": phase_a_furthest_cells,
        "maps_reached": all_explore_maps,
        "target_maps_specified": target_maps,
        "unreached_target_maps": all_unreached_targets,
        "reached_target_maps": all_reached_targets,
        "target_maps_reached": target_maps_status,
    }
    explore_path = out_dir / "explore.json"
    explore_path.write_text(json.dumps(explore_report, indent=2))

    # ══════════════════════════════════════════════════════════════════════════
    # MERGE PHASE B STATS -> coverage.json
    # ══════════════════════════════════════════════════════════════════════════
    merged_steps_per_map: Dict[str, int] = {}
    merged_cells_per_map: Dict[str, int] = {}
    merged_first_reached: Dict[str, int] = {}
    total_record_steps = 0
    total_battle_steps = 0
    total_blackouts = 0

    for s in worker_summaries:
        pb = s["phase_b"]
        total_record_steps += pb["total_steps"]
        total_battle_steps += pb["battle_steps"]
        total_blackouts += pb["blackouts"]

        for m_int, count in pb["steps_per_map"].items():
            k = str(m_int)
            merged_steps_per_map[k] = merged_steps_per_map.get(k, 0) + count

        for m_int, cells in pb["cells_per_map"].items():
            k = str(m_int)
            merged_cells_per_map[k] = max(merged_cells_per_map.get(k, 0), cells)

        for m_int, ep in pb["maps_first_reached"].items():
            k = str(m_int)
            if k not in merged_first_reached or ep < merged_first_reached[k]:
                merged_first_reached[k] = ep

    battle_fraction = (total_battle_steps / total_record_steps) if total_record_steps > 0 else 0.0
    overall_sps = total_record_steps / total_wall_time if total_wall_time > 0 else 0.0

    chunk_files = list(out_dir.glob("transitions_*.npz"))
    total_disk_bytes = sum(f.stat().st_size for f in chunk_files)
    total_disk_mb = total_disk_bytes / (1024 * 1024)

    all_recorded_maps = sorted([int(m) for m in merged_steps_per_map.keys()])
    coverage_report = {
        "steps_per_map": merged_steps_per_map,
        "distinct_cells_per_map": merged_cells_per_map,
        "first_reached_episode": merged_first_reached,
        "battle_fraction": battle_fraction,
        "blackout_count": total_blackouts,
        "total_steps": total_record_steps,
        "total_episodes": episodes,
        "total_chunks": num_chunks,
        "total_disk_mb": round(total_disk_mb, 2),
        "total_wall_time_s": round(total_wall_time, 2),
        "steps_per_second": round(overall_sps, 1),
        "maps_reached": all_recorded_maps,
        "target_maps_specified": target_maps,
        "unreached_target_maps": all_unreached_targets,
        "reached_target_maps": all_reached_targets,
        "target_maps_reached": {
            TARGET_MAP_LABELS.get(mid, f"Map_{mid}"): (mid in all_recorded_maps)
            for mid in maps_to_report
        },
    }

    coverage_path = out_dir / "coverage.json"
    coverage_path.write_text(json.dumps(coverage_report, indent=2))

    # Print summary tables
    print("\n" + "=" * 80)
    print("PHASE A EXPLORATION COVERAGE (Merged)")
    print("=" * 80)
    for m_id in maps_to_report:
        k = str(m_id)
        name = TARGET_MAP_LABELS.get(m_id, f"Map_{m_id}")
        reached = m_id in all_explore_maps
        cells = phase_a_cells_per_map.get(k, 0)
        first_it = phase_a_first_iter.get(k, "N/A")
        status = "REACHED" if reached else "NOT REACHED"
        print(f"  [{status:<11}] Map {m_id:2d} ({name:<30}): {cells:4d} cells | First Iter: {first_it}")
    if target_maps is not None:
        print(f"\nTarget Maps Summary ({len(target_maps)} total):")
        print(f"  Reached     ({len(all_reached_targets)}): {all_reached_targets}")
        print(f"  NOT REACHED ({len(all_unreached_targets)}): {all_unreached_targets}")
    print("\nStopping reasons per worker:")
    for w_id, reason in phase_a_stop_reasons.items():
        fc = phase_a_furthest_cells[w_id]
        print(f"  Worker {w_id}: {reason} | Furthest: map={fc[0]}, pos=({fc[1]}, {fc[2]})")

    print("\n" + "=" * 80)
    print("PHASE B RECORDED DATASET COVERAGE")
    print("=" * 80)
    print(f"Total Steps: {total_record_steps:,} | Chunks: {num_chunks} | Wall Time: {total_wall_time:.1f}s | Speed: {overall_sps:.1f} steps/s")
    print(f"Disk Used: {total_disk_mb:.1f} MB (across {len(chunk_files)} files)")
    print(f"Battle Fraction: {battle_fraction:.3f} | Blackouts: {total_blackouts}")
    print("\nTarget Map Recorded Steps:")
    for m_id in maps_to_report:
        k = str(m_id)
        name = TARGET_MAP_LABELS.get(m_id, f"Map_{m_id}")
        reached = m_id in all_recorded_maps
        steps = merged_steps_per_map.get(k, 0)
        cells = merged_cells_per_map.get(k, 0)
        first_ep = merged_first_reached.get(k, "N/A")
        status = "REACHED" if reached else "NOT REACHED"
        print(f"  [{status:<11}] Map {m_id:2d} ({name:<30}): {steps:6d} steps | {cells:3d} cells | First Ep: {first_ep}")
    print("=" * 80 + "\n")

    return coverage_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase exploration data collection for Pokemon Red.")
    parser.add_argument("--rom", type=Path, default=ROM_PATH, help="Path to Pokemon Red ROM")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH, help="Path to initial save state")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel emulator processes (max 8)")
    parser.add_argument("--episodes", type=int, default=300, help="Total episodes to collect in Phase B (default: 300)")
    parser.add_argument("--explore-cap", type=int, default=400000, help="Max Phase A exploration steps per worker (default: 400000)")
    parser.add_argument("--save-interval", type=int, default=10, help="Episodes per .npz chunk (default: 10)")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, default=Path("work/collect_v2"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--start-states", type=Path, default=None, help="Path to index.json containing initial states")
    parser.add_argument("--no-default-start", action="store_true", help="Do not seed archive with the default start state")
    parser.add_argument("--target-maps", type=int, nargs="*", default=None, help="Target map IDs for Phase B start cell sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_maps = args.target_maps if (args.target_maps is not None and len(args.target_maps) > 0) else None
    collect(
        rom_path=args.rom,
        state_path=args.state,
        out_dir=args.out_dir,
        workers=args.workers,
        episodes=args.episodes,
        explore_cap=args.explore_cap,
        save_interval=args.save_interval,
        base_seed=args.seed,
        start_states_path=args.start_states,
        no_default_start=args.no_default_start,
        target_maps=target_maps,
    )


if __name__ == "__main__":
    main()
