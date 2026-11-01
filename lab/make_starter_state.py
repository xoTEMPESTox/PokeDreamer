"""make_starter_state.py — Generate a PyBoy starter save state for Pokemon Red.

Sequence:
1. Start from saves/pallet.state (create it via make_states if missing).
2. Walk north towards Route 1 to trigger Oak cutscene; advance into Oak's Lab (map 40).
3. Clear lab introduction dialogue until player is free to move at (5, 4).
4. Move to starter Poke Ball table, choose starter, decline nickname (press B).
5. Walk towards lab exit to trigger rival battle, spam A to fight and win, clear post-battle dialogue.
   Save saves/starter_lab.state here.
6. Walk out of Oak's lab into Pallet Town (map 0). Clear any text, verify movement & party size.
   Save saves/starter_pallet.state.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lab.env import ROM_PATH, PokemonRed, make_states
from src.game_state import extract_game_state, screen_capture
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
)

SPECIES_NAMES = {
    0xB0: "Charmander",
    0xB1: "Squirtle",
    0x99: "Bulbasaur",
    176: "Charmander",
    177: "Squirtle",
    153: "Bulbasaur",
}

STARTER_COORDS = {
    # Position to stand and face UP towards the ball
    "charmander": (6, 4),
    "squirtle": (7, 4),
    "bulbasaur": (8, 4),
}


def save_debug_frame(env: PokemonRed, out_path: Path) -> None:
    """Save an unscaled RGB screen capture to disk as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = screen_capture(env.pyboy)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def trigger_oak_and_enter_lab(env: PokemonRed, debug_dir: Path | None = None) -> dict[str, Any]:
    """Phase 1: Walk north toward Route 1 to trigger Oak; advance dialog until map 40."""
    print("\n--- Phase 1: Trigger Oak and enter Oak's Lab ---")
    _, ram = env.observe()
    print(f"Start pos: ({ram['x']}, {ram['y']}), map: {ram['map_id']}")

    # Walk right to X=10
    while ram["x"] < 10:
        _, ram = env.step("right")
    while ram["x"] > 10:
        _, ram = env.step("left")

    # Walk up to Y=1 to step into the grass trigger
    for _ in range(10):
        _, ram = env.step("up")
        if ram["y"] <= 1:
            break

    print(f"Triggered Oak at pos: ({ram['x']}, {ram['y']}), facing: {ram['facing']}")

    # Advance dialogue / cutscene until in Oak's lab (map 40)
    steps = 0
    max_steps = 150
    while ram["map_id"] != 40 and steps < max_steps:
        _, ram = env.step("a")
        steps += 1

    if ram["map_id"] != 40:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase1_failed.png")
        raise RuntimeError(
            f"Phase 1 failed: Oak cutscene did not reach map 40 within {max_steps} steps (current map={ram['map_id']})"
        )

    print(f"Reached Oak's Lab (map 40) after {steps} steps! RAM: {ram}")
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase1_lab_entry.png")
    return ram


def clear_lab_intro_dialog(env: PokemonRed, debug_dir: Path | None = None) -> dict[str, Any]:
    """Phase 2: In Oak's Lab, advance cutscene & dialog until player is free to move."""
    print("\n--- Phase 2: Clear Oak's Lab introduction dialogue ---")
    # Oak walks Red up to (5, 3)
    steps = 0
    max_walk_steps = 100
    _, ram = env.observe()
    while (ram["x"], ram["y"]) != (5, 3) and steps < max_walk_steps:
        _, ram = env.step("a")
        steps += 1

    if (ram["x"], ram["y"]) != (5, 3):
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase2_walk_failed.png")
        raise RuntimeError(
            f"Phase 2 failed: Oak did not walk to (5, 3) within {max_walk_steps} steps (pos=({ram['x']},{ram['y']}))"
        )

    print(f"Oak and Red at desk (5, 3) after {steps} steps.")

    # Oak and Blue talk; advance dialogue until player can move (try pressing 'down')
    dialog_steps = 0
    max_dialog_steps = 150
    freed = False
    while dialog_steps < max_dialog_steps:
        _, ram = env.step("down")
        dialog_steps += 1
        if (ram["x"], ram["y"]) != (5, 3):
            freed = True
            break
        _, ram = env.step("a")
        dialog_steps += 1

    if not freed:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase2_dialog_failed.png")
        raise RuntimeError(
            f"Phase 2 failed: Dialog at desk did not finish within {max_dialog_steps} steps (pos=({ram['x']},{ram['y']}))"
        )

    print(f"Player free to move at pos: ({ram['x']}, {ram['y']}) after {dialog_steps} steps.")
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase2_free_in_lab.png")
    return ram


def choose_starter(
    env: PokemonRed,
    starter: str = "charmander",
    debug_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 3: Move to starter ball, press A to take it, and decline nickname with B."""
    print(f"\n--- Phase 3: Choose starter ({starter}) and decline nickname ---")
    starter_key = starter.lower()
    if starter_key not in STARTER_COORDS:
        raise ValueError(f"Unknown starter '{starter}'. Choose from: {list(STARTER_COORDS.keys())}")

    target_x, target_y = STARTER_COORDS[starter_key]
    _, ram = env.observe()

    # Navigate to target position
    while ram["x"] < target_x:
        _, ram = env.step("right")
    while ram["x"] > target_x:
        _, ram = env.step("left")
    while ram["y"] < target_y:
        _, ram = env.step("down")
    while ram["y"] > target_y:
        _, ram = env.step("up")

    # Turn to face UP towards the table ball at Y=3
    _, ram = env.step("up")
    print(f"Standing at ball position: ({ram['x']}, {ram['y']}), facing: {ram['facing']}")

    # Press A to inspect the ball and confirm YES
    ball_steps = 0
    max_ball_steps = 60
    while env.pyboy.memory[PARTY_SIZE] == 0 and ball_steps < max_ball_steps:
        _, ram = env.step("a")
        ball_steps += 1

    if env.pyboy.memory[PARTY_SIZE] == 0:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase3_ball_failed.png")
        raise RuntimeError(f"Phase 3 failed: Did not obtain {starter} within {max_ball_steps} steps")

    print(f"Obtained starter! Party size: {env.pyboy.memory[PARTY_SIZE]} (after {ball_steps} steps)")

    # Decline nickname: press B repeatedly
    for _ in range(20):
        env.step("b")

    # Clear rival's response text ("I'll take this one then!")
    for _ in range(5):
        env.step("b")

    # Verify starter info
    raw_species = env.pyboy.memory[PARTY_SPECIES_ADDRESSES[0]]
    species_name = SPECIES_NAMES.get(raw_species, f"Species_{raw_species}")
    cur_hp = (env.pyboy.memory[HP_ADDRESSES[0]] << 8) | env.pyboy.memory[HP_ADDRESSES[0] + 1]
    max_hp = (env.pyboy.memory[MAX_HP_ADDRESSES[0]] << 8) | env.pyboy.memory[MAX_HP_ADDRESSES[0] + 1]
    level = env.pyboy.memory[LEVEL_ADDRESSES[0]]
    starter_info = {
        "species_id": raw_species,
        "species_name": species_name,
        "level": level,
        "hp": cur_hp,
        "max_hp": max_hp,
    }
    print(f"Starter chosen: {species_name} (Lv {level}, HP {cur_hp}/{max_hp})")

    if debug_dir:
        save_debug_frame(env, debug_dir / "phase3_starter_chosen.png")
    return env.observe()[1], starter_info


def fight_rival_battle(
    env: PokemonRed,
    lab_state_out: Path | None = None,
    debug_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 4: Walk towards exit to trigger battle, spam A through battle, clear dialog."""
    print("\n--- Phase 4: Trigger and resolve rival battle ---")
    trigger_steps = 0
    max_trigger_steps = 100

    # Walk towards exit door (X=5, Y=11) until IN_BATTLE != 0
    while env.pyboy.memory[IN_BATTLE] == 0 and trigger_steps < max_trigger_steps:
        ram = env.observe()[1]
        x = ram["x"]
        if x > 5:
            env.step("left")
        elif x < 4:
            env.step("right")
        else:
            env.step("down")
        # Also press A to advance rival's approach dialog
        env.step("a")
        trigger_steps += 2

    if env.pyboy.memory[IN_BATTLE] == 0:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase4_trigger_failed.png")
        raise RuntimeError(f"Phase 4 failed: Rival battle was not triggered within {max_trigger_steps} steps")

    print(f"Rival battle triggered after ~{trigger_steps} steps!")

    # Spam A through the battle until IN_BATTLE == 0
    battle_steps = 0
    max_battle_steps = 500
    while env.pyboy.memory[IN_BATTLE] != 0 and battle_steps < max_battle_steps:
        env.step("a")
        battle_steps += 1

    if env.pyboy.memory[IN_BATTLE] != 0:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase4_battle_failed.png")
        raise RuntimeError(f"Phase 4 failed: Battle did not terminate within {max_battle_steps} steps")

    # Check battle outcome
    cur_hp = (env.pyboy.memory[HP_ADDRESSES[0]] << 8) | env.pyboy.memory[HP_ADDRESSES[0] + 1]
    max_hp = (env.pyboy.memory[MAX_HP_ADDRESSES[0]] << 8) | env.pyboy.memory[MAX_HP_ADDRESSES[0] + 1]
    level = env.pyboy.memory[LEVEL_ADDRESSES[0]]
    outcome = "Won" if cur_hp > 0 else "Lost"
    battle_info = {
        "outcome": outcome,
        "battle_steps": battle_steps,
        "final_hp": cur_hp,
        "max_hp": max_hp,
        "final_level": level,
    }
    print(f"Battle concluded! Outcome: {outcome} ({battle_steps} 'a' steps), HP: {cur_hp}/{max_hp}, Level: {level}")

    # Clear post-battle dialogue: wait until player is free to move
    post_steps = 0
    max_post_steps = 150
    freed = False
    pos_initial = (env.observe()[1]["x"], env.observe()[1]["y"])
    while post_steps < max_post_steps:
        _, ram = env.step("down")
        post_steps += 1
        if (ram["x"], ram["y"]) != pos_initial:
            freed = True
            break
        _, ram = env.step("a")
        post_steps += 1

    if not freed:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase4_post_dialog_failed.png")
        raise RuntimeError(f"Phase 4 failed: Post-battle dialogue did not finish within {max_post_steps} steps")

    print(f"Player free in Oak's Lab at pos: ({ram['x']}, {ram['y']})")

    # Save saves/starter_lab.state if requested
    if lab_state_out is not None:
        lab_state_out.parent.mkdir(parents=True, exist_ok=True)
        with open(lab_state_out, "wb") as f:
            env.pyboy.save_state(f)
        print(f"Saved Oak's Lab state -> {lab_state_out.resolve()}")

    if debug_dir:
        save_debug_frame(env, debug_dir / "phase4_lab_state.png")
    return env.observe()[1], battle_info


def exit_lab_to_pallet(env: PokemonRed, debug_dir: Path | None = None) -> dict[str, Any]:
    """Phase 5: Walk out of Oak's Lab into Pallet Town (map 0) and clear text."""
    print("\n--- Phase 5: Walk out of Oak's Lab into Pallet Town ---")
    exit_steps = 0
    max_exit_steps = 60
    ram = env.observe()[1]

    while ram["map_id"] != 0 and exit_steps < max_exit_steps:
        if ram["x"] > 5:
            _, ram = env.step("left")
        elif ram["x"] < 4:
            _, ram = env.step("right")
        else:
            _, ram = env.step("down")
        exit_steps += 1

    if ram["map_id"] != 0:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase5_exit_failed.png")
        raise RuntimeError(f"Phase 5 failed: Did not reach Pallet Town (map 0) within {max_exit_steps} steps")

    # Clear any text boxes and let warp fade settle
    for _ in range(5):
        _, ram = env.step("b")

    print(f"Successfully arrived in Pallet Town after {exit_steps} steps! RAM: {ram}")
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase5_pallet_town.png")
    return ram


def verify_and_save(
    env: PokemonRed,
    pallet_state_out: Path,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """Phase 6: Verify invariants (map 0, party 1, not in battle, free movement) and save state."""
    print("\n--- Phase 6: Verify invariants and save final state ---")
    _, ram = env.observe()

    # 1. Map ID must be 0 (Pallet Town)
    if ram["map_id"] != 0:
        raise AssertionError(f"Verification failed: Expected map_id=0, got {ram['map_id']}")

    # 2. Party size must be exactly 1
    party_size = env.pyboy.memory[PARTY_SIZE]
    if party_size != 1:
        raise AssertionError(f"Verification failed: Expected party size 1, got {party_size}")

    # 3. Must not be in battle
    if ram["in_battle"] or env.pyboy.memory[IN_BATTLE] != 0:
        raise AssertionError("Verification failed: Player is currently marked in battle")

    # 4. Movement test: verify player can move freely without changing the saved state
    snap = env.snapshot()
    moved = False
    for direction in ["up", "left", "right"]:
        env.restore(snap)
        _, r_test = env.step(direction)
        if (r_test["x"], r_test["y"]) != (ram["x"], ram["y"]):
            moved = True
            print(f"Movement verified: pressing '{direction}' changed pos from ({ram['x']},{ram['y']}) to ({r_test['x']},{r_test['y']})")
            break

    env.restore(snap)
    if not moved:
        raise AssertionError("Verification failed: Player could not move in any tested direction (dialog locked)")

    # 5. Route 1 reachability test: verify player can reach Route 1 without Oak stopping
    snap_before_route1 = env.snapshot()
    # Walk left 3 times, up 10 times, right 1 time, up 3 times to Route 1 (map 12)
    path = ["left"] * 3 + ["up"] * 10 + ["right"] + ["up"] * 3
    r_route = ram
    for move in path:
        _, r_route = env.step(move)
    if r_route["map_id"] == 12:
        print(f"Route 1 reachability confirmed! Final pos: ({r_route['x']}, {r_route['y']}), map={r_route['map_id']}")
    else:
        print(f"Warning: Walk path ended at map {r_route['map_id']}, pos ({r_route['x']}, {r_route['y']})")

    env.restore(snap_before_route1)

    # Save state to pallet_state_out
    pallet_state_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pallet_state_out, "wb") as f:
        env.pyboy.save_state(f)
    print(f"Saved Pallet Town starter state -> {pallet_state_out.resolve()}")

    if debug_dir:
        save_debug_frame(env, debug_dir / "phase6_final_pallet.png")

    final_ram = env.observe()[1]
    return final_ram


def make_starter_state(
    rom: str | Path = ROM_PATH,
    pallet_state: str | Path = Path("saves/pallet.state"),
    starter_pallet_out: str | Path = Path("saves/starter_pallet.state"),
    starter_lab_out: str | Path = Path("saves/starter_lab.state"),
    starter: str = "charmander",
    debug_dir: str | Path | None = Path("work/starter_debug"),
) -> dict[str, Any]:
    """Execute end-to-end starter state generation."""
    rom_path = Path(rom)
    pallet_path = Path(pallet_state)
    starter_pallet_path = Path(starter_pallet_out)
    starter_lab_path = Path(starter_lab_out)
    debug_path = Path(debug_dir) if debug_dir else None

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path.resolve()}")

    # Ensure pallet.state exists
    if not pallet_path.exists():
        print(f"[make_starter_state] {pallet_path} missing; generating initial states first...")
        make_states(rom=rom_path, out_dir=pallet_path.parent)

    phase_logs = {}

    print("=" * 80)
    print("GENERATING POKEMON RED STARTER SAVE STATE")
    print(f"ROM:           {rom_path.resolve()}")
    print(f"Base State:    {pallet_path.resolve()}")
    print(f"Starter:       {starter}")
    print(f"Pallet Output: {starter_pallet_path.resolve()}")
    print(f"Lab Output:    {starter_lab_path.resolve()}")
    print("=" * 80)

    with PokemonRed(rom=rom_path, state=pallet_path) as env:
        # Phase 1: Trigger Oak and enter lab
        r1 = trigger_oak_and_enter_lab(env, debug_path)
        phase_logs["Phase 1 (Enter Lab)"] = r1

        # Phase 2: Clear intro dialog
        r2 = clear_lab_intro_dialog(env, debug_path)
        phase_logs["Phase 2 (Free in Lab)"] = r2

        # Phase 3: Choose starter
        r3, starter_info = choose_starter(env, starter=starter, debug_dir=debug_path)
        phase_logs["Phase 3 (Starter Chosen)"] = r3

        # Phase 4: Rival battle & post-battle dialog
        r4, battle_info = fight_rival_battle(env, lab_state_out=starter_lab_path, debug_dir=debug_path)
        phase_logs["Phase 4 (Post Rival Battle)"] = r4

        # Phase 5: Exit lab to Pallet Town
        r5 = exit_lab_to_pallet(env, debug_path)
        phase_logs["Phase 5 (Pallet Town Arrive)"] = r5

        # Phase 6: Verify and save
        r6 = verify_and_save(env, pallet_state_out=starter_pallet_path, debug_dir=debug_path)
        phase_logs["Phase 6 (Final Pallet Town)"] = r6

    print("\n" + "=" * 80)
    print("RUN COMPLETE — SUMMARY")
    print("=" * 80)
    print("Phase Log (RAM after each phase):")
    for phase_name, ram_state in phase_logs.items():
        print(f"  {phase_name:<30}: {ram_state}")

    print("\nStarter & Battle Details:")
    print(f"  Chosen Starter: {starter_info['species_name']} (Species ID: {starter_info['species_id']})")
    print(f"  Battle Outcome: {battle_info['outcome']} (after {battle_info['battle_steps']} steps)")
    print(f"  Final HP:       {battle_info['final_hp']}/{battle_info['max_hp']}")
    print(f"  Final Level:    {battle_info['final_level']}")

    print("\nFinal Pallet Town RAM Summary:")
    print(f"  Map ID:    {r6['map_id']} (Pallet Town)")
    print(f"  Position:  ({r6['x']}, {r6['y']})")
    print(f"  Facing:    {r6['facing']}")
    print(f"  In Battle: {r6['in_battle']}")
    print(f"  Party Sz:  1")
    print("=" * 80 + "\n")

    return {
        "phase_logs": phase_logs,
        "starter_info": starter_info,
        "battle_info": battle_info,
        "final_ram": r6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pokemon Red starter save state.")
    parser.add_argument("--rom", type=Path, default=ROM_PATH, help="Path to Pokemon Red ROM")
    parser.add_argument("--pallet-state", type=Path, default=Path("saves/pallet.state"), help="Path to initial pallet.state")
    parser.add_argument("--starter-pallet-out", type=Path, default=Path("saves/starter_pallet.state"), help="Output starter_pallet.state")
    parser.add_argument("--starter-lab-out", type=Path, default=Path("saves/starter_lab.state"), help="Output starter_lab.state")
    parser.add_argument("--starter", type=str, default="charmander", choices=["charmander", "squirtle", "bulbasaur"], help="Starter Pokemon to select")
    parser.add_argument("--debug-dir", type=Path, default=Path("work/starter_debug"), help="Debug directory for screenshots")
    args = parser.parse_args()

    make_starter_state(
        rom=args.rom,
        pallet_state=args.pallet_state,
        starter_pallet_out=args.starter_pallet_out,
        starter_lab_out=args.starter_lab_out,
        starter=args.starter,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
