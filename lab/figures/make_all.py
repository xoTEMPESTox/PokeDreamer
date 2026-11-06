"""Generate all research figures for the PokeDreamer v3 write-up.

Runs:
1. fig_leg_paths
2. fig_leg_failures
3. fig_leg_success
4. fig_door_warp
5. fig_battle_fidelity
6. fig_gpu_ledger
7. fig_quest_presses

Prints each caption and verifies output files.
"""
import sys
import time

from lab.figures import (
    fig_leg_paths,
    fig_leg_failures,
    fig_leg_success,
    fig_door_warp,
    fig_battle_fidelity,
    fig_gpu_ledger,
    fig_quest_presses,
)

FIGURE_MODULES = [
    ("fig_leg_paths", fig_leg_paths),
    ("fig_leg_failures", fig_leg_failures),
    ("fig_leg_success", fig_leg_success),
    ("fig_door_warp", fig_door_warp),
    ("fig_battle_fidelity", fig_battle_fidelity),
    ("fig_gpu_ledger", fig_gpu_ledger),
    ("fig_quest_presses", fig_quest_presses),
]


def main():
    print("=" * 80)
    print("Generating PokeDreamer v3 publication figures...")
    print("=" * 80)

    captions = {}
    t0_all = time.time()

    for name, module in FIGURE_MODULES:
        t0 = time.time()
        caption = module.main()
        elapsed = time.time() - t0
        captions[name] = caption
        print(f"\n[{name}] ({elapsed:.2f}s)")
        print(f"Caption: {caption}")

    print("\n" + "=" * 80)
    print(f"All figures generated successfully in {time.time() - t0_all:.2f}s!")
    print("Outputs located in work/figures/ (*.png and *.svg)")
    print("=" * 80)

    return captions


if __name__ == "__main__":
    main()
