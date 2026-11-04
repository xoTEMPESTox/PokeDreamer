"""make_move_states.py — Generate start states for the grind and heal leg tasks.

Idempotent: reuses existing state files in work/route_states/ unless --force is given.
Start tiles:
  - leave_viridian_pc: Viridian PC in front of Nurse Joy after healing (map 41, x=3, y=4)
  - to_grass: Viridian City outside the PC door (map 1, x=23, y=26)
  - to_viridian_pc: Route 1 tall grass anchor (map 12, x=13, y=30)
  - to_pewter_pc: Pewter City south entrance where travel_to_pewter arrives (map 2, x=18, y=35)
  - leave_pewter_pc: Pewter PC in front of Nurse Joy after healing (map 58, x=3, y=4)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lab.agent.gym_quest import travel_to_pewter_skill
from lab.agent.nav import (
    MAP_ROUTE_1,
    MAP_VIRIDIAN_CITY,
    heal_pkmn_center,
    navigate_to,
)
from lab.env import PokemonRed

START_SAVE = "saves/pokedex_viridian.state"
ROUTE_DIR = Path("work/route_states")

TARGET_TILES = {
    "leave_viridian_pc": (41, 3, 4),
    "to_grass": (1, 23, 26),
    "to_viridian_pc": (12, 13, 30),
    "to_pewter_pc": (2, 18, 35),
    "leave_pewter_pc": (58, 3, 4),
}


def make_states(force: bool = False) -> dict[str, tuple[int, int, int]]:
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)
    states: dict[str, tuple[int, int, int]] = {}

    for task_name, expected in TARGET_TILES.items():
        out_path = ROUTE_DIR / f"{task_name}.state"
        if out_path.exists() and not force:
            with PokemonRed(state=str(out_path)) as env:
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, (
                    f"Existing state {out_path} has {actual}, expected {expected}"
                )
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (reused)")
            continue

        if task_name == "leave_viridian_pc":
            with PokemonRed(state=START_SAVE) as env:
                heal_pkmn_center(env, exit_after=False)
                for _ in range(4):
                    env.step("pass")
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, f"Ended at {actual}, expected {expected}"
                out_path.write_bytes(env.snapshot())
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (saved)")

        elif task_name == "to_grass":
            with PokemonRed(state=START_SAVE) as env:
                navigate_to(env, MAP_VIRIDIAN_CITY, target_tiles=[(23, 26)])
                for _ in range(4):
                    env.step("pass")
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, f"Ended at {actual}, expected {expected}"
                out_path.write_bytes(env.snapshot())
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (saved)")

        elif task_name == "to_viridian_pc":
            with PokemonRed(state=START_SAVE) as env:
                navigate_to(env, MAP_ROUTE_1, target_tiles=[(13, 30)])
                for _ in range(4):
                    env.step("pass")
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, f"Ended at {actual}, expected {expected}"
                out_path.write_bytes(env.snapshot())
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (saved)")

        elif task_name == "to_pewter_pc":
            with PokemonRed(state=START_SAVE) as env:
                travel_to_pewter_skill(env, nav="bfs")
                for _ in range(4):
                    env.step("pass")
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, f"Ended at {actual}, expected {expected}"
                out_path.write_bytes(env.snapshot())
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (saved)")

        elif task_name == "leave_pewter_pc":
            pewter_state = ROUTE_DIR / "to_pewter_pc.state"
            with PokemonRed(state=str(pewter_state)) as env:
                heal_pkmn_center(env, exit_after=False)
                for _ in range(4):
                    env.step("pass")
                _, ram = env.observe()
                actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
                assert actual == expected, f"Ended at {actual}, expected {expected}"
                out_path.write_bytes(env.snapshot())
                states[task_name] = actual
                print(f"{task_name}: map {actual[0]} ({actual[1]}, {actual[2]}) (saved)")

    return states


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing state files")
    args = ap.parse_args()
    make_states(force=args.force)
