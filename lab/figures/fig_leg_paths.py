"""Figure 1: Real-game evaluation traces on each navigation leg's main map.

Plots the 10 navigation legs in a 2x5 grid showing:
- Observed tiles in train split (light grey squares) on the map where the leg spends most steps
- Exit tiles (dark squares)
- 20 random evaluation episodes of the learned policy (green if success, red if failure)
- Per-trace jitter (±0.25 tile) and alpha 0.35 so path spread and overlap are visible
- Y-axis inverted (y grows downward in the game)
"""
from collections import Counter
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import numpy as np

import lab.data
from lab.agent.tasks import TASKS, train_geodesic_fields
from lab.figures import style

# Cache dataset load across train_geodesic_fields calls for fast execution
_CACHED_DATA = None
_ORIG_LOAD = lab.data.load


def _cached_load(*args, **kwargs):
    global _CACHED_DATA
    if _CACHED_DATA is None:
        _CACHED_DATA = _ORIG_LOAD(*args, **kwargs)
    return _CACHED_DATA


lab.data.load = _cached_load

LEGS = [
    # Row 1: Main progression legs
    ("to_route2", "to_route2_wm31_gate_s0"),
    ("to_gate", "to_gate_wm31_gate_s0"),
    ("forest", "forest_wm31_gate_s0"),
    ("to_pewter", "to_pewter_wm31_gate_s0"),
    ("to_gym", "to_gym_wm31_gate_s0"),
    # Row 2: PC detours and grind legs
    ("to_grass", "to_grass_wm31_gate_s0"),
    ("to_viridian_pc", "to_viridian_pc_wm31_gate_s0"),
    ("leave_viridian_pc", "leave_viridian_pc_wm31_gate_s0"),
    ("to_pewter_pc", "to_pewter_pc_wm31_gate_it1800_s0"),
    ("leave_pewter_pc", "leave_pewter_pc_wm31_gate_mc_s0"),
]

MAP_NAMES = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    12: "Route 1",
    13: "Route 2",
    41: "Viridian PC",
    47: "Forest North Gate",
    50: "Forest South Gate",
    51: "Viridian Forest",
    54: "Pewter Gym",
    58: "Pewter PC",
}

CAPTION = (
    "Real-game evaluation traces (20 random episodes per leg, with ±0.25 tile jitter and alpha 0.35) "
    "of the learned navigation policies on each leg's primary traversal map, showing observed training tiles "
    "(light grey), target exits (dark), successful paths (green), and failures (red)."
)


def plot():
    style.apply()

    fig, axes = plt.subplots(2, 5, figsize=(10.2, 5.0), dpi=200)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88, wspace=0.10, hspace=0.24)

    rng = np.random.default_rng(42)

    for idx, (leg, run) in enumerate(LEGS):
        ax = axes[idx // 5, idx % 5]
        task = TASKS[leg]

        # Load evaluation traces
        run_file = Path(f"work/agent/{leg}_learned_{run}_s200.json")
        with open(run_file) as f:
            agent_data = json.load(f)
        results = agent_data["results"]

        # Choose 20 random episodes
        chosen = rng.choice(len(results), size=min(20, len(results)), replace=False)

        # Determine the map where the leg spends the most steps across the 20 episodes
        map_counts = Counter()
        for ep_idx in chosen:
            for step in results[ep_idx].get("trace", []):
                map_counts[step[0]] += 1
        main_map, _ = map_counts.most_common(1)[0]
        map_name = MAP_NAMES.get(main_map, f"Map {main_map}")

        # Compute geodesic fields to identify observed tiles on main_map
        fields = train_geodesic_fields(task["exits"], extended=True)
        arr = fields[main_map]
        obs = arr < arr.max()
        ys, xs = np.where(obs)

        # Draw observed tiles as light grey squares
        obs_patches = [Rectangle((x - 0.5, y - 0.5), 1.0, 1.0) for x, y in zip(xs, ys)]
        pc_obs = PatchCollection(obs_patches, facecolor="#e2e8f0", edgecolor="#cbd5e1", linewidth=0.25, zorder=1)
        ax.add_collection(pc_obs)

        # Draw exit tiles on main map as dark squares
        exit_tiles = task["exits"].get(main_map, [])
        if exit_tiles:
            exit_patches = [Rectangle((x - 0.5, y - 0.5), 1.0, 1.0) for x, y in exit_tiles]
            pc_exit = PatchCollection(exit_patches, facecolor="#1e293b", edgecolor="#0f172a", linewidth=0.4, zorder=2)
            ax.add_collection(pc_exit)

        # Draw traces with fixed per-trace jitter (±0.25 tile) and alpha 0.35
        for ep_idx in chosen:
            r = results[ep_idx]
            trace = r.get("trace", [])

            # Extract points on main_map
            pts = [(step[1], step[2]) for step in trace if step[0] == main_map]
            if len(pts) >= 2:
                # Per-trace fixed jitter
                jx = rng.uniform(-0.25, 0.25)
                jy = rng.uniform(-0.25, 0.25)
                bx, by = style.break_jumps(pts)
                pxs = [x + jx for x in bx]
                pys = [y + jy for y in by]

                c = style.COLOR_SUCCESS if r.get("success", False) else style.COLOR_DANGER
                ax.plot(pxs, pys, color=c, lw=1.2, alpha=0.35, zorder=3)

        # Set map extent (inverted y-axis so y grows downward)
        pad = 1.0
        ax.set_xlim(xs.min() - pad - 0.5, xs.max() + pad + 0.5)
        ax.set_ylim(ys.max() + pad + 0.5, ys.min() - pad - 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{leg}\n({map_name})", fontsize=8.5, fontweight="bold", pad=4)
        ax.set_axis_off()

    # Figure legend at the top
    legend_elements = [
        Patch(facecolor="#e2e8f0", edgecolor="#cbd5e1", label="Observed tile"),
        Patch(facecolor="#1e293b", edgecolor="#0f172a", label="Target exit"),
        Line2D([0], [0], color=style.COLOR_SUCCESS, lw=2.0, alpha=0.7, label="Success trace (jitter ±0.25)"),
        Line2D([0], [0], color=style.COLOR_DANGER, lw=2.0, alpha=0.7, label="Failure trace"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )

    style.save(fig, "fig_leg_paths")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
