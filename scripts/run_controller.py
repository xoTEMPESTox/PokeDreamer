"""
run_controller.py — Day 2 deliverable demo script.

Demonstrates the controller loop: takes a hardcoded (map_id, x, y) goal
and runs the frozen PPO (or random actions) until arrival, stuck, or timeout.

Usage:
    conda activate pokemon-rl

    # With frozen PPO checkpoint:
    python scripts/run_controller.py \\
        --rom "Pokemon Red Version (Colorization).gb" \\
        --checkpoint "external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip" \\
        --save-state saves/intro_done.state \\
        --target-x 5 --target-y 5 \\
        --log-file logs/controller_run.jsonl

    # Without checkpoint (random actions — quick env smoke test):
    python scripts/run_controller.py \\
        --rom "Pokemon Red Version (Colorization).gb" \\
        --save-state saves/intro_done.state \\
        --target-x 5 --target-y 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.controller import Controller, ControllerConfig
from src.goal import Goal
from src.game_state import extract_game_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the external controller toward a hardcoded goal")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="SB3 PPO .zip checkpoint (omit for random actions)")
    p.add_argument("--save-state", type=Path, default=None,
                   help="PyBoy .state file to load (use saves/intro_done.state)")
    p.add_argument("--target-x", type=int, required=True, help="Target tile X coordinate")
    p.add_argument("--target-y", type=int, required=True, help="Target tile Y coordinate")
    p.add_argument("--target-map", type=int, default=None,
                   help="Target map ID (default: same as current map)")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="Max steps before timeout (default: 2000)")
    p.add_argument("--stuck-window", type=int, default=40,
                   help="Steps without improvement before stuck (default: 40)")
    p.add_argument("--ticks-per-action", type=int, default=24,
                   help="Emulator ticks per action (default: 24)")
    p.add_argument("--log-file", type=Path, default=Path("logs/controller_run.jsonl"))
    p.add_argument("--deterministic", action="store_true", help="Use deterministic policy predictions")
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── ROM check ─────────────────────────────────────────────────────────────
    if not args.rom.exists():
        print(f"[ERROR] ROM not found: {args.rom.resolve()}")
        sys.exit(1)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = None
    if args.checkpoint is not None:
        try:
            from src.frozen_ppo import FrozenPPO
            policy = FrozenPPO(args.checkpoint)
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}\nFalling back to random actions.")

    if policy is None:
        print("[run_controller] No checkpoint — using RANDOM actions")

    # ── Start emulator ────────────────────────────────────────────────────────
    print(f"[run_controller] ROM: {args.rom.resolve()}")
    pyboy = PyBoy(str(args.rom), window="null")

    if args.save_state and args.save_state.exists():
        with open(args.save_state, "rb") as f:
            pyboy.load_state(f)
        print(f"[run_controller] Loaded save state: {args.save_state}")
    else:
        if args.save_state:
            print(f"[WARN] Save state not found: {args.save_state}")
            print("  Run: python scripts/make_save_state.py first")
        print("[run_controller] Starting from ROM cold start")

    # ── Print starting position ───────────────────────────────────────────────
    init_state = extract_game_state(pyboy)
    print(f"[run_controller] Start: {init_state}")

    # ── Build goal ────────────────────────────────────────────────────────────
    goal = Goal.goto(
        x=args.target_x,
        y=args.target_y,
        map_id=args.target_map,
        reason="Day 2 hardcoded test",
    )
    print(f"[run_controller] Goal:  {goal}")

    # ── Configure controller ──────────────────────────────────────────────────
    cfg = ControllerConfig(
        ticks_per_action=args.ticks_per_action,
        max_steps_per_goal=args.max_steps,
        stuck_window=args.stuck_window,
        verbose=not args.quiet,
        deterministic=args.deterministic,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    with Controller(pyboy, policy=policy, config=cfg, log_file=args.log_file) as ctrl:
        result = ctrl.run(goal)

    pyboy.stop()

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONTROLLER RUN SUMMARY")
    print("=" * 60)
    print(f"  Goal         : {goal}")
    print(f"  Status       : {result.status.value.upper()}")
    print(f"  Steps taken  : {result.steps_taken}")
    print(f"  Final pos    : ({result.final_x}, {result.final_y}) @ map {result.final_map_id}")
    if result.notes:
        print(f"  Notes        : {result.notes}")
    print(f"  Log file     : {args.log_file.resolve()}")
    print("=" * 60)

    # Exit code: 0 = success, 1 = stuck/timeout
    sys.exit(0 if result.succeeded else 1)


if __name__ == "__main__":
    main()
