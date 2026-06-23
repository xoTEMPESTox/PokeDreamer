"""
make_save_state.py — Create a PyBoy 2.7-compatible save state.

Ticks through the game intro (title screen + Oak dialogue) and executes a
hardcoded walk sequence to walk down the stairs and out the door of the player's
house to save the state on the Pallet Town overworld.

Usage:
    conda activate pokemon-rl
    python scripts/make_save_state.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from src.ram_addresses import MAP_ID as MAP_ID_ADDR, PLAYER_X, PLAYER_Y, MAP_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a PyBoy 2.7 save state past the intro")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"))
    p.add_argument("--out", type=Path, default=Path("saves/intro_done.state"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.rom.exists():
        print(f"[ERROR] ROM not found: {args.rom.resolve()}")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[make_save_state] ROM:    {args.rom.resolve()}")
    print(f"[make_save_state] Output: {args.out.resolve()}\n")

    pyboy = PyBoy(str(args.rom), window="null")

    # 1. Spam A + Start for 8000 ticks to clear Oak's dialogue
    print("Clearing Oak's dialogue (8000 frames)...")
    for tick in range(1, 8001):
        if tick % 6 in (0, 2, 4):
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            pyboy.tick(1, render=False)
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        elif tick % 30 == 0:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            pyboy.tick(1, render=False)
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        else:
            pyboy.tick(1, render=False)

    # Let the transition fade-in finish
    pyboy.tick(100, render=False)

    init_map = pyboy.memory[MAP_ID_ADDR]
    init_x = pyboy.memory[PLAYER_X]
    init_y = pyboy.memory[PLAYER_Y]
    print(f"Start bedroom state: map={hex(init_map)} pos=({init_x},{init_y})")

    # Walk execution helper (like red_gym_env.py)
    def walk_step(press_event, release_event):
        pyboy.send_input(press_event)
        for i in range(24):
            if i == 8:
                pyboy.send_input(release_event)
            pyboy.tick(1, render=False)

    # 2. Walk from Bedroom to Stairs (X=3, Y=6) -> (7, 1)
    print("Walking to Bedroom stairs...")
    # Walk right (X=3 -> X=5)
    for _ in range(2):
        walk_step(WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
    # Walk up (Y=6 -> Y=1)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
    # Walk right (X=5 -> X=7)
    for _ in range(2):
        walk_step(WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)

    # Wait for warp transition
    pyboy.tick(100, render=False)
    
    mid_map = pyboy.memory[MAP_ID_ADDR]
    mid_x = pyboy.memory[PLAYER_X]
    mid_y = pyboy.memory[PLAYER_Y]
    print(f"Post-stairs 1F state: map={hex(mid_map)} pos=({mid_x},{mid_y})")

    if mid_map != 0x25:
        print("[ERROR] Failed to reach Living Room (0x25)")
        pyboy.stop()
        sys.exit(1)

    # 3. Walk from Living Room stairs to front door (X=7, Y=1) -> (2, 7) -> exit
    print("Walking to Living Room exit...")
    # Walk down (Y=1 -> Y=6)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
    # Walk left (X=7 -> X=2)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
    # Walk down (Y=6 -> Y=7/exit)
    for _ in range(3):
        walk_step(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

    # Wait for overworld transition
    pyboy.tick(100, render=False)

    final_map = pyboy.memory[MAP_ID_ADDR]
    final_x = pyboy.memory[PLAYER_X]
    final_y = pyboy.memory[PLAYER_Y]
    print(f"Final state: map={hex(final_map)} pos=({final_x},{final_y})")

    if final_map == 0x00:
        print("\n[make_save_state] OK Overworld: Pallet Town")
        with open(args.out, "wb") as f:
            pyboy.save_state(f)
        print(f"[make_save_state] Saved → {args.out.resolve()}")
        pyboy.stop()
        print("[make_save_state] Save state generated successfully!")
    else:
        print(f"\n[ERROR] Failed to reach Pallet Town. Final map={hex(final_map)}")
        pyboy.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
