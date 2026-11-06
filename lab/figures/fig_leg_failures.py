"""Figure: Real-game failure traces on Viridian Forest and Pewter City.

Plots failure traces from weaker policy seeds:
1. forest seed 1 (forest_wm31_gate_s1, 69/100 successes) on Viridian Forest (map 51)
2. to_pewter_pc seed 1 (to_pewter_pc_wm31_gate_it1800_s1, 0/100 successes) on Pewter City (map 2)
Marks the primary failure terminal tile:
- (25, 12) in Viridian Forest
- (13, 26) door tile in Pewter City
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

FAILURE_CASES = [
    {
        "leg": "forest",
        "run": "forest_wm31_gate_s1",
        "map_id": 51,
        "title": "forest (Viridian Forest) — Seed 1 (69/100)",
        "expected_trap": (25, 12),
        "trap_label": "Antidote stall (25,12)\n15 of 31 failures end here",
        "trap_offset": (15, -15),
    },
    {
        "leg": "to_pewter_pc",
        "run": "to_pewter_pc_wm31_gate_it1800_s1",
        "map_id": 2,
        "title": "to_pewter_pc (Pewter City) — Seed 1 (0/100)",
        "expected_trap": (13, 26),
        "trap_label": "PC door (13,26)\n64 of 100 end here (no up)",
        "trap_offset": (-20, 20),
    },
]

CAPTION = (
    "Real-game failure traces (20 episodes each, with ±0.25 tile jitter and alpha 0.35) from weaker policy seeds: "
    "(left) Viridian Forest seed 1 (69/100): 31 failures, 15 of them ending at (25,12) beside the Antidote item ball at (25,11), a spot the training data barely covers; "
    "(right) to_pewter_pc seed 1 (0/100) reaching the Pokémon Center door (13,26) but repeatedly failing to press up."
)


def plot():
    style.apply()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), dpi=200)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.86, wspace=0.14)

    rng = np.random.default_rng(999)

    for idx, case in enumerate(FAILURE_CASES):
        ax = axes[idx]
        leg = case["leg"]
        run = case["run"]
        map_id = case["map_id"]
        task = TASKS[leg]

        # Load run data
        run_file = Path(f"work/agent/{leg}_learned_{run}_s200.json")
        with open(run_file) as f:
            agent_data = json.load(f)
        results = agent_data["results"]

        # Filter failure episodes
        failures = [r for r in results if not r.get("success", False)]
        if not failures:
            failures = results

        # Choose 20 failure episodes
        chosen_indices = rng.choice(len(failures), size=min(20, len(failures)), replace=False)

        # Geodesic fields for map
        fields = train_geodesic_fields(task["exits"], extended=True)
        arr = fields[map_id]
        obs = arr < arr.max()
        ys, xs = np.where(obs)

        # Observed tiles
        obs_patches = [Rectangle((x - 0.5, y - 0.5), 1.0, 1.0) for x, y in zip(xs, ys)]
        pc_obs = PatchCollection(obs_patches, facecolor="#e2e8f0", edgecolor="#cbd5e1", linewidth=0.25, zorder=1)
        ax.add_collection(pc_obs)

        # Exit tiles
        exit_tiles = task["exits"].get(map_id, [])
        if exit_tiles:
            exit_patches = [Rectangle((x - 0.5, y - 0.5), 1.0, 1.0) for x, y in exit_tiles]
            pc_exit = PatchCollection(exit_patches, facecolor="#1e293b", edgecolor="#0f172a", linewidth=0.4, zorder=2)
            ax.add_collection(pc_exit)

        # Draw failure traces
        for ep_i in chosen_indices:
            r = failures[ep_i]
            trace = r.get("trace", [])
            pts = [(step[1], step[2]) for step in trace if step[0] == map_id]
            if len(pts) >= 2:
                jx = rng.uniform(-0.25, 0.25)
                jy = rng.uniform(-0.25, 0.25)
                bx, by = style.break_jumps(pts)
                pxs = [x + jx for x in bx]
                pys = [y + jy for y in by]
                ax.plot(pxs, pys, color=style.COLOR_DANGER, lw=1.3, alpha=0.35, zorder=3)

        # Mark the most common failure terminal tile
        trap_x, trap_y = case["expected_trap"]
        ax.scatter(
            [trap_x], [trap_y],
            marker="*", s=140, facecolor="#facc15", edgecolor="#991b1b",
            linewidth=1.2, zorder=6
        )

        # Annotate failure trap
        ox, oy = case["trap_offset"]
        ax.annotate(
            case["trap_label"],
            xy=(trap_x, trap_y),
            xytext=(ox, oy), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
            fontsize=8.0, fontweight="bold", color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#f87171", lw=0.8, alpha=0.9),
            zorder=7,
        )

        pad = 1.0
        ax.set_xlim(xs.min() - pad - 0.5, xs.max() + pad + 0.5)
        ax.set_ylim(ys.max() + pad + 0.5, ys.min() - pad - 0.5)
        ax.set_aspect("equal")
        ax.set_title(case["title"], fontsize=9.0, fontweight="bold", pad=6)
        ax.set_axis_off()

    legend_elements = [
        Patch(facecolor="#e2e8f0", edgecolor="#cbd5e1", label="Observed tile"),
        Patch(facecolor="#1e293b", edgecolor="#0f172a", label="Target exit"),
        Line2D([0], [0], color=style.COLOR_DANGER, lw=2.0, alpha=0.7, label="Failure trace (jitter ±0.25)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#facc15", markeredgecolor="#991b1b",
               markersize=10, label="Common failure terminal tile"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )

    style.save(fig, "fig_leg_failures")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
