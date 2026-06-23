"""
steerability_test.py — Systematic steerability test for Day 3.

Runs 25 trials starting from Pallet Town (5,6) to various nearby walkable target
coordinates on the same map. Logs success rate, average steps, and stuck rate.

Usage:
    conda activate pokemon-rl
    python scripts/steerability_test.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.controller import Controller, ControllerConfig
from src.goal import Goal
from src.frozen_ppo import FrozenPPO

# Walkable targets in Pallet Town at varying distances from (5,6)
# Manhattan distances range from 1 to 6
TEST_TARGETS = [
    # Distance 1
    {"x": 5, "y": 7, "dist": 1},
    {"x": 6, "y": 6, "dist": 1},
    # Distance 2
    {"x": 5, "y": 8, "dist": 2},
    {"x": 6, "y": 7, "dist": 2},
    {"x": 4, "y": 7, "dist": 2},
    # Distance 3
    {"x": 5, "y": 9, "dist": 3},
    {"x": 6, "y": 8, "dist": 3},
    {"x": 7, "y": 7, "dist": 3},
    {"x": 8, "y": 6, "dist": 3},
    # Distance 4
    {"x": 5, "y": 10, "dist": 4},
    {"x": 7, "y": 8, "dist": 4},
    {"x": 9, "y": 6, "dist": 4},
    # Distance 5
    {"x": 8, "y": 8, "dist": 5},
    {"x": 10, "y": 6, "dist": 5},
    # Distance 6
    {"x": 11, "y": 6, "dist": 6},
    {"x": 9, "y": 8, "dist": 6},
]


def main() -> None:
    rom_path = Path("Pokemon - Red Version (USA, Europe).gb")
    checkpoint_path = Path("external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip")
    save_state_path = Path("saves/intro_done.state")

    if not rom_path.exists():
        print(f"[ERROR] ROM not found: {rom_path}")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not save_state_path.exists():
        print(f"[ERROR] Save state not found: {save_state_path}")
        sys.exit(1)

    print("[steerability_test] Loading policy...")
    policy = FrozenPPO(checkpoint_path)

    results = []

    print(f"\nStarting Steerability Test ({len(TEST_TARGETS)} trials)...")
    print("-" * 75)
    print(f"{'Trial':<6} | {'Target':<10} | {'Start Dist':<10} | {'Status':<10} | {'Steps':<8} | {'Final Pos':<10}")
    print("-" * 75)

    for idx, target in enumerate(TEST_TARGETS):
        # Fresh emulator instance for each trial to ensure complete independence
        pyboy = PyBoy(str(rom_path), window="null")
        with open(save_state_path, "rb") as f:
            pyboy.load_state(f)

        goal = Goal.goto(x=target["x"], y=target["y"], map_id=0, reason=f"Steerability Trial {idx+1}")

        cfg = ControllerConfig(
            ticks_per_action=24,
            max_steps_per_goal=120,
            stuck_window=30,
            verbose=False,
            deterministic=True,
        )

        with Controller(pyboy, policy=policy, config=cfg) as ctrl:
            res = ctrl.run(goal)

        pyboy.stop()

        status_str = res.status.value.upper()
        print(f"#{idx+1:<5} | ({target['x']},{target['y']}) | {target['dist']:<10} | {status_str:<10} | {res.steps_taken:<8} | ({res.final_x},{res.final_y})")

        results.append({
            "trial": idx + 1,
            "target": (target["x"], target["y"]),
            "start_dist": target["dist"],
            "status": res.status.value,
            "steps_taken": res.steps_taken,
            "final_pos": (res.final_x, res.final_y),
            "final_map_id": res.final_map_id,
        })

    # Compute stats
    total = len(results)
    successes = [r for r in results if r["status"] == "success"]
    stucks = [r for r in results if r["status"] == "stuck"]
    timeouts = [r for r in results if r["status"] == "timeout"]

    success_rate = (len(successes) / total) * 100
    stuck_rate = (len(stucks) / total) * 100
    timeout_rate = (len(timeouts) / total) * 100
    avg_steps_success = sum(r["steps_taken"] for r in successes) / len(successes) if successes else 0

    print("-" * 75)
    print("SUMMARY STATISTICS")
    print("-" * 75)
    print(f"Total Trials       : {total}")
    print(f"Success Rate       : {success_rate:.1f}% ({len(successes)}/{total})")
    print(f"Stuck Rate         : {stuck_rate:.1f}% ({len(stucks)}/{total})")
    print(f"Timeout Rate       : {timeout_rate:.1f}% ({len(timeouts)}/{total})")
    print(f"Avg Steps (Success): {avg_steps_success:.1f}")
    print("-" * 75)

    # Save to file
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "steerability_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_trials": total,
            "success_rate": success_rate,
            "stuck_rate": stuck_rate,
            "timeout_rate": timeout_rate,
            "avg_steps_success": avg_steps_success,
            "trials": results
        }, f, indent=2)
    print(f"Results saved to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
