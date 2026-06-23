"""
run_trajectory.py — Day 1 deliverable script.

Runs the frozen PPO policy for N steps in headless mode, logging
(step, action, map_id, x, y, badges, in_battle, dialog_open, party_hp)
to a JSONL file.  Prints a summary at the end.

Usage:
    conda activate pokemon-rl
    python scripts/run_trajectory.py \\
        --rom PokemonRed.gb \\
        --checkpoint external/PokemonRedExperiments/checkpoints/<name>.zip \\
        --steps 500 \\
        --log-file logs/trajectory.jsonl

Without --checkpoint, runs with random actions (useful for quick env smoke-test).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.game_state import extract_game_state, screen_capture

# ── Optional PPO import (graceful fallback to random) ─────────────────────────
_PPO_AVAILABLE = True
try:
    from src.frozen_ppo import FrozenPPO
except ImportError:
    _PPO_AVAILABLE = False


PYBOY_BUTTONS = ["down", "left", "right", "up", "a", "b", "start"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frozen PPO and log (x,y,map) trajectory to JSONL"
    )
    p.add_argument(
        "--rom",
        type=Path,
        default=Path("Pokemon - Red Version (USA, Europe).gb"),
        help="Path to PokemonRed.gb",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SB3 PPO .zip checkpoint.  Omit to use random actions.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of emulator steps to run (default: 500)",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/trajectory.jsonl"),
        help="Output JSONL log file (default: logs/trajectory.jsonl)",
    )
    p.add_argument(
        "--save-state",
        type=Path,
        default=None,
        help="Optional: load a .state file before running (skip intro)",
    )
    p.add_argument(
        "--ticks-per-action",
        type=int,
        default=24,
        help="Emulator ticks to advance per action (default: 24 = ~0.4s game time)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic PPO actions (default: True)",
    )
    return p.parse_args()


def print_summary(log_entries: list[dict]) -> None:
    """Print a human-readable summary of the trajectory."""
    if not log_entries:
        return

    maps_visited = {e["map_name"] for e in log_entries}
    xs = [e["x"] for e in log_entries]
    ys = [e["y"] for e in log_entries]
    battles = sum(1 for e in log_entries if e["in_battle"])
    dialogs = sum(1 for e in log_entries if e["dialog_open"])

    # Unique tiles visited
    tiles = {(e["map_id"], e["x"], e["y"]) for e in log_entries}

    first = log_entries[0]
    last  = log_entries[-1]

    print("\n" + "=" * 60)
    print("  TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"  Steps logged     : {len(log_entries)}")
    print(f"  Unique tiles     : {len(tiles)}")
    print(f"  Maps visited     : {', '.join(sorted(maps_visited))}")
    print(f"  Start position   : ({first['x']}, {first['y']}) @ {first['map_name']}")
    print(f"  End position     : ({last['x']}, {last['y']}) @ {last['map_name']}")
    print(f"  X range          : [{min(xs)}, {max(xs)}]")
    print(f"  Y range          : [{min(ys)}, {max(ys)}]")
    print(f"  Battle steps     : {battles}")
    print(f"  Dialog steps     : {dialogs}")
    print(f"  Badge count (end): {last['badges']}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    # ── Validate ROM ──────────────────────────────────────────────────────────
    if not args.rom.exists():
        print(f"[ERROR] ROM not found: {args.rom.resolve()}")
        sys.exit(1)

    # ── Set up log file ───────────────────────────────────────────────────────
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Load policy (or fall back to random) ──────────────────────────────────
    policy = None
    if args.checkpoint is not None:
        if not _PPO_AVAILABLE:
            print("[WARN] FrozenPPO import failed — falling back to random actions")
        elif not args.checkpoint.exists():
            print(f"[WARN] Checkpoint not found: {args.checkpoint} — using random actions")
        else:
            policy = FrozenPPO(args.checkpoint)
            print(f"[run_trajectory] Policy: {policy}")

    if policy is None:
        import random
        print("[run_trajectory] Using RANDOM actions (no checkpoint)")

    # ── Start emulator ────────────────────────────────────────────────────────
    print(f"[run_trajectory] ROM:       {args.rom.resolve()}")
    print(f"[run_trajectory] Steps:     {args.steps}")
    print(f"[run_trajectory] Log:       {args.log_file.resolve()}")
    print(f"[run_trajectory] Ticks/act: {args.ticks_per_action}")
    print()

    pyboy = PyBoy(str(args.rom), window="null")

    if args.save_state and args.save_state.exists():
        with open(args.save_state, "rb") as f:
            pyboy.load_state(f)
        print(f"[run_trajectory] Loaded save state: {args.save_state}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    log_entries: list[dict] = []
    t_start = time.perf_counter()

    with open(args.log_file, "w") as log_f:
        for step in range(args.steps):

            # 1. Capture current screen
            frame = screen_capture(pyboy)

            # 2. Choose action
            if policy is not None:
                action = policy.predict(frame, deterministic=args.deterministic)
            else:
                action = random.randint(0, 6)

            # 3. Press button and advance emulator ticks
            pyboy.button(PYBOY_BUTTONS[action])
            pyboy.tick(args.ticks_per_action, render=False)

            # 4. Read RAM state (same tick as frame above → synchronised)
            state = extract_game_state(pyboy)

            # 5. Build log entry
            entry = {
                "step": step,
                "action": action,
                "action_name": PYBOY_BUTTONS[action],
                **state.to_dict(),
            }
            log_entries.append(entry)

            # 6. Write to JSONL immediately (flush every 50 steps)
            log_f.write(json.dumps(entry) + "\n")
            if step % 50 == 0:
                log_f.flush()
                elapsed = time.perf_counter() - t_start
                print(
                    f"  step {step:>5}/{args.steps}  "
                    f"{state}  "
                    f"[{elapsed:.1f}s]"
                )

    pyboy.stop()

    elapsed_total = time.perf_counter() - t_start
    print(f"\n[run_trajectory] Done in {elapsed_total:.1f}s")
    print(f"[run_trajectory] Log written → {args.log_file.resolve()}")

    print_summary(log_entries)


if __name__ == "__main__":
    main()
