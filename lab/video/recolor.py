"""Recolour a recorded gym_quest run with the current stage-1 colour rules, by replaying its button presses.

`gym_quest --save-color` stores colour frames made with the colour rules of the day. The game is deterministic for a
fixed input sequence, so replaying the recorded buttons from the same save state, after the same seed-dependent idle
frames, reproduces the run (this also works for LLM-planned runs, whose planner is not deterministic). Every replayed
frame is checked against the recorded greyscale frame; the script stops at the first mismatch.

Usage: python -m lab.video.recolor --traj work/video/traj/run_seed_100.npz --seed 100
Writes <traj stem>_color.npz (the previous one is kept as <traj stem>_color_prev.npz).
Needs the ROM, so it has no CI test; the frame-by-frame check is its verification.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lab.color.native import pixel_tiles
from lab.color.rules import build_tile_map, colorize_frame
from lab.env import PokemonRed


def stage1_palette():
    """The same palette gym_quest --save-color builds."""
    labels = json.loads(Path("lab/color/labels.json").read_text(encoding="utf-8"))
    keys = [f"{k.split(':')[0]}:{layer}:{k.split(':')[1]}" for k in labels for layer in (0, 1)]
    return {"classes": json.loads(Path("lab/color/palette.json").read_text(encoding="utf-8"))["classes"],
            "tiles": build_tile_map(labels, keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="run_seed_<seed>.npz written by gym_quest --save-traj")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--state", default="saves/pokedex_viridian.state")
    args = ap.parse_args()

    traj = Path(args.traj)
    color_path = traj.with_name(traj.stem + "_color.npz")
    rec = np.load(traj)
    old = dict(np.load(color_path))
    frames, actions = rec["frames"], rec["actions"]
    palette = stage1_palette()

    colors = np.empty((len(actions), 144, 160, 3), np.uint8)
    with PokemonRed(state=args.state) as env:
        env.pyboy.tick(int(np.random.default_rng(args.seed).integers(0, 600)), render=False)  # as gym_quest
        for i, a in enumerate(actions):
            frame, _ = env.step(int(a))
            if not np.array_equal(frame, frames[i]):
                raise SystemExit(f"replay diverged at step {i} (action {int(a)}); not writing anything")
            colors[i] = colorize_frame(pixel_tiles(env.pyboy), palette)
            if i % 1000 == 0:
                print(f"step {i}/{len(actions)}", flush=True)

    color_path.rename(color_path.with_name(traj.stem + "_color_prev.npz"))
    old["color"] = colors
    np.savez_compressed(color_path, **old)
    print(f"replayed {len(actions)} steps, all frames identical; wrote {color_path}")


if __name__ == "__main__":
    main()
