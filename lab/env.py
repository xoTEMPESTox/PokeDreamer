"""Pokemon Red environment wrapper around PyBoy 2.7.

Matches the exact frame preprocessing and action timing of recorded dataset
(collect_data.py):
- 1 step = button pressed for 8 frames, tick 24 frames with render=True
- Observation = (72, 80) uint8 greyscale frame + RAM state dict
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from lab.data import load, to_small
from src.game_state import extract_game_state, screen_capture

ROM_PATH = Path("Pokemon - Red Version (USA, Europe).gb")
BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]
ACTION_NAMES = BUTTONS
ACTION_TO_INDEX = {btn: idx for idx, btn in enumerate(BUTTONS)}


class PokemonRed:
    """Headless PyBoy wrapper for Pokemon Red matching dataset collection."""

    def __init__(self, rom: str | Path = ROM_PATH, state: str | Path | bytes | None = None):
        self.rom_path = Path(rom)
        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM not found at {self.rom_path.resolve()}")

        self.pyboy = PyBoy(str(self.rom_path), window="null")
        if state is not None:
            self.restore(state)

    def observe(self) -> Tuple[np.ndarray, dict[str, Any]]:
        """Observe current Game Boy screen and RAM state.

        Returns:
            frame72: (72, 80) uint8 greyscale frame (downsampled via cv2.INTER_AREA)
            ram: dict with map_id, x, y, facing, in_battle
        """
        raw = screen_capture(self.pyboy)
        frame72 = to_small(raw[None])[0]
        st = extract_game_state(self.pyboy)
        ram = {
            "map_id": int(st.map_id),
            "x": int(st.x),
            "y": int(st.y),
            "facing": int(st.facing),
            "in_battle": bool(st.in_battle),
        }
        return frame72, ram

    def step(self, action: int | str) -> Tuple[np.ndarray, dict[str, Any]]:
        """Execute one 24-frame action step and observe the result.

        Matches collect_data.py:
        - If button != 'pass', hold for 8 frames
        - Advance emulator by 24 ticks with render=True
        """
        if isinstance(action, (int, np.integer)):
            btn = BUTTONS[int(action)]
        else:
            btn = str(action).lower()

        if btn != "pass":
            self.pyboy.button(btn, 8)
        self.pyboy.tick(24, render=True)
        return self.observe()

    def snapshot(self) -> bytes:
        """Capture emulator state as bytes."""
        bio = io.BytesIO()
        self.pyboy.save_state(bio)
        return bio.getvalue()

    def restore(self, state: bytes | bytearray | str | Path | io.BytesIO) -> None:
        """Restore emulator from bytes, file path, or stream."""
        if isinstance(state, (bytes, bytearray)):
            bio = io.BytesIO(state)
            self.pyboy.load_state(bio)
        elif isinstance(state, (str, Path)):
            with open(state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            self.pyboy.load_state(state)

    def close(self) -> None:
        """Stop PyBoy emulator session."""
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None

    def __enter__(self) -> PokemonRed:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def make_states(rom: str | Path = ROM_PATH, out_dir: str | Path = Path("saves")) -> None:
    """Generate PyBoy save states for Red's bedroom (map 38) and Pallet Town (map 0).

    Copies the sequence of v1/scripts/make_save_state.py without editing or importing it.
    """
    rom_path = Path(rom)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path.resolve()}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bedroom_path = out_path / "bedroom.state"
    pallet_path = out_path / "pallet.state"

    print(f"[make_states] Initializing PyBoy with {rom_path}...")
    pyboy = PyBoy(str(rom_path), window="null")

    print("[make_states] Clearing Oak's dialogue (8000 frames)...")
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

    ram_bedroom = extract_game_state(pyboy)
    print(
        f"[make_states] Bedroom state: map_id={ram_bedroom.map_id} "
        f"({hex(ram_bedroom.map_id)}), pos=({ram_bedroom.x}, {ram_bedroom.y}), "
        f"facing={ram_bedroom.facing_name}"
    )

    with open(bedroom_path, "wb") as f:
        pyboy.save_state(f)
    print(f"[make_states] Saved -> {bedroom_path.resolve()}")

    def walk_step(press_event, release_event):
        pyboy.send_input(press_event)
        for i in range(24):
            if i == 8:
                pyboy.send_input(release_event)
            pyboy.tick(1, render=False)

    print("[make_states] Walking to Bedroom stairs...")
    # Walk right (X=3 -> X=5)
    for _ in range(2):
        walk_step(WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
    # Walk up (Y=6 -> Y=1)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
    # Walk right (X=5 -> X=7)
    for _ in range(2):
        walk_step(WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)

    pyboy.tick(100, render=False)
    ram_1f = extract_game_state(pyboy)
    print(
        f"[make_states] Post-stairs 1F state: map_id={ram_1f.map_id} "
        f"({hex(ram_1f.map_id)}), pos=({ram_1f.x}, {ram_1f.y}), "
        f"facing={ram_1f.facing_name}"
    )

    if ram_1f.map_id != 0x25:
        pyboy.stop()
        raise RuntimeError(f"Failed to reach Living Room (0x25), got {hex(ram_1f.map_id)}")

    print("[make_states] Walking to Living Room exit...")
    # Walk down (Y=1 -> Y=6)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
    # Walk left (X=7 -> X=2)
    for _ in range(5):
        walk_step(WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
    # Walk down (Y=6 -> Y=7/exit)
    for _ in range(3):
        walk_step(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

    pyboy.tick(100, render=False)
    ram_pallet = extract_game_state(pyboy)
    print(
        f"[make_states] Pallet Town state: map_id={ram_pallet.map_id} "
        f"({hex(ram_pallet.map_id)}), pos=({ram_pallet.x}, {ram_pallet.y}), "
        f"facing={ram_pallet.facing_name}"
    )

    if ram_pallet.map_id != 0x00:
        pyboy.stop()
        raise RuntimeError(f"Failed to reach Pallet Town (0x00), got {hex(ram_pallet.map_id)}")

    with open(pallet_path, "wb") as f:
        pyboy.save_state(f)
    print(f"[make_states] Saved -> {pallet_path.resolve()}")

    pyboy.stop()
    print("[make_states] Both save states generated successfully!\n")


def check_fidelity(rom: str | Path = ROM_PATH, state_dir: str | Path = Path("saves")) -> None:
    """Run fidelity checks against recorded dataset:

    a) Confirm Red's-room episodes in dataset start at map 38 (3,6) and bedroom state matches.
    b) Frame fidelity: ~30 random direction steps from bedroom state; report min mean-abs pixel diff.
    c) Movement: 2 right, 5 up from bedroom state; report RAM (x, y) after each step.
    """
    state_dir = Path(state_dir)
    bedroom_path = state_dir / "bedroom.state"
    if not bedroom_path.exists():
        print(f"[check_fidelity] {bedroom_path} missing, generating states first...")
        make_states(rom, state_dir)

    print("=" * 80)
    print("FIDELITY CHECKS")
    print("=" * 80)

    # ── Check a: Red's-room episode starts in dataset ─────────────────────────
    print("\n--- Check (a): Dataset Episode Starts in Red's Room ---")
    data, splits = load()
    starts = np.where(data["episode_starts"])[0]
    bedroom_starts = starts[data["map_ids"][starts] == 38]
    n_bedroom = len(bedroom_starts)
    unique_xs = np.unique(data["xs"][bedroom_starts])
    unique_ys = np.unique(data["ys"][bedroom_starts])
    unique_facings = np.unique(data["facings"][bedroom_starts])

    print(f"Dataset total episodes: {len(starts)}, starting in bedroom (map 38): {n_bedroom}")
    print(f"Dataset bedroom starts: unique x={unique_xs.tolist()}, y={unique_ys.tolist()}, facing={unique_facings.tolist()}")

    # Check our bedroom state RAM
    with PokemonRed(rom=rom, state=bedroom_path) as env:
        f0, ram0 = env.observe()

    bedroom_matches = (
        ram0["map_id"] == 38
        and ram0["x"] == 3
        and ram0["y"] == 6
        and (ram0["x"] in unique_xs)
        and (ram0["y"] in unique_ys)
    )
    print(f"Env bedroom state: map_id={ram0['map_id']}, x={ram0['x']}, y={ram0['y']}, facing={ram0['facing']}")
    print(f"Bedroom state matches dataset start: {bedroom_matches}")

    # ── Check b: Frame fidelity over ~30 random direction steps ──────────────
    print("\n--- Check (b): Frame Fidelity (30 random direction steps) ---")
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, size=30)  # directions 0=down, 1=left, 2=right, 3=up

    min_diffs = []
    with PokemonRed(rom=rom, state=bedroom_path) as env:
        for i, act in enumerate(actions):
            frame, ram = env.step(act)
            btn = BUTTONS[act]

            # Find matching frames in dataset
            matches = np.where(
                (data["map_ids"] == ram["map_id"])
                & (data["xs"] == ram["x"])
                & (data["ys"] == ram["y"])
                & (data["facings"] == ram["facing"])
                & (~data["in_battles"])
            )[0]

            if len(matches) > 0:
                diffs = np.mean(
                    np.abs(data["frames"][matches].astype(float) - frame.astype(float)),
                    axis=(1, 2),
                )
                min_d = float(np.min(diffs))
                min_diffs.append(min_d)
                print(
                    f"  step {i:2d} | act={act} ({btn:<5}) | map={ram['map_id']} pos=({ram['x']:2d},{ram['y']:2d}) "
                    f"facing={ram['facing']:2d} | matches={len(matches):5d} | min abs diff = {min_d:.4f}"
                )
            else:
                print(
                    f"  step {i:2d} | act={act} ({btn:<5}) | map={ram['map_id']} pos=({ram['x']:2d},{ram['y']:2d}) "
                    f"facing={ram['facing']:2d} | NO MATCH in dataset"
                )

    med_diff = float(np.median(min_diffs)) if min_diffs else float("nan")
    max_diff = float(np.max(min_diffs)) if min_diffs else float("nan")
    print(f"\nFrame fidelity over {len(min_diffs)} steps:")
    print(f"  Median min-diff: {med_diff:.4f}")
    print(f"  Max min-diff:    {max_diff:.4f}")
    if med_diff == 0.0 and max_diff == 0.0:
        print("  -> Pixel-identical match (0.0000) to recorded dataset!")
    else:
        print(f"  -> Discrepancy observed: median={med_diff:.4f}, max={max_diff:.4f}")

    # ── Check c: Movement check (right x2, up x5) ─────────────────────────────
    print("\n--- Check (c): Movement Check (2 right, 5 up) ---")
    moves = [2, 2, 3, 3, 3, 3, 3]  # 2 right, 5 up
    with PokemonRed(rom=rom, state=bedroom_path) as env:
        _, r0 = env.observe()
        print(f"  Start: pos=({r0['x']}, {r0['y']})")
        curr_x, curr_y = r0["x"], r0["y"]
        movement_ok = True
        for step_i, act in enumerate(moves, 1):
            _, r = env.step(act)
            btn = BUTTONS[act]
            dx = r["x"] - curr_x
            dy = r["y"] - curr_y
            disp = abs(dx) + abs(dy)
            if disp != 1:
                movement_ok = False
            print(f"  Step {step_i}: action={act} ({btn:<5}) -> pos=({r['x']}, {r['y']}) | delta=({dx:+d}, {dy:+d})")
            curr_x, curr_y = r["x"], r["y"]

    expected_final = (5, 1)
    final_pos = (curr_x, curr_y)
    print(f"Final position: {final_pos} (expected {expected_final})")
    print(f"Movement check passed: {movement_ok and final_pos == expected_final}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon Red Gym-like environment and verification tools.")
    parser.add_argument("--rom", type=Path, default=ROM_PATH, help="Path to Pokemon Red ROM")
    parser.add_argument("--make-states", action="store_true", help="Generate bedroom.state and pallet.state")
    parser.add_argument("--check", action="store_true", help="Run fidelity checks against recorded dataset")
    parser.add_argument("--saves-dir", type=Path, default=Path("saves"), help="Directory for save states")
    args = parser.parse_args()

    if args.make_states:
        make_states(rom=args.rom, out_dir=args.saves_dir)

    if args.check:
        check_fidelity(rom=args.rom, state_dir=args.saves_dir)

    if not args.make_states and not args.check:
        parser.print_help()


if __name__ == "__main__":
    main()
