"""Figure 6: Total GameBoy presses per quest evaluation run.

Strip plots of total presses across 10 evaluation runs per seed:
- Group 0: rules + path-finding (baseline) from work/gym_rng/ (grey, one seed labeled baseline)
- Group 1: 5 travel legs learned (gym_learned_gateall, seeds 0, 1, 2)
- Group 2: all 10 legs learned (first) (gym_learned_moves, seeds 0, 1, 2)
- Group 3: all 10 legs learned (best) (gym_learned_best, seeds 0, 1, 2)
- Coloured by seed with medians marked
"""
import glob
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from lab.figures import style

GROUPS = [
    {
        "name": "rules + path-finding\n(baseline)",
        "kind": "baseline",
        "dir_pattern": "work/gym_rng/run_seed_*.json",
        "seeds": [("baseline", "#64748b")],
    },
    {
        "name": "5 travel legs\nlearned",
        "kind": "learned",
        "prefix": "gym_learned_gateall",
        "seeds": [
            ("s0", style.COLOR_PRIMARY),
            ("s1", style.COLOR_SECONDARY),
            ("s2", style.COLOR_SUCCESS),
        ],
    },
    {
        "name": "all 10 legs learned\n(first)",
        "kind": "learned",
        "prefix": "gym_learned_moves",
        "seeds": [
            ("s0", style.COLOR_PRIMARY),
            ("s1", style.COLOR_SECONDARY),
            ("s2", style.COLOR_SUCCESS),
        ],
    },
    {
        "name": "all 10 legs learned\n(best)",
        "kind": "learned",
        "prefix": "gym_learned_best",
        "seeds": [
            ("s0", style.COLOR_PRIMARY),
            ("s1", style.COLOR_SECONDARY),
            ("s2", style.COLOR_SUCCESS),
        ],
    },
]

CAPTION = (
    "Distribution of total GameBoy button presses to obtain the Boulder Badge across quest policy configurations: "
    "rules + path-finding baseline (grey), 5 travel legs learned (gateall), and all 10 legs learned (first and best retrains), "
    "with individual 10-run evaluations shown as strip points and seed medians marked."
)


def plot():
    style.apply()

    rng = np.random.default_rng(123)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=200)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.16, top=0.86)

    # Alternate background shading per group
    # Positions of group centers
    group_x_centers = [0.0, 1.35, 2.70, 4.05]
    col_width = 0.22

    for idx, cx in enumerate(group_x_centers):
        span_w = 0.50 if idx == 0 else 0.58
        if idx % 2 == 1:
            ax.axvspan(cx - span_w, cx + span_w, color="#f8fafc", zorder=1)

    for g_idx, grp in enumerate(GROUPS):
        base_x = group_x_centers[g_idx]
        seeds = grp["seeds"]
        n_seeds = len(seeds)

        for s_idx, (s_label, s_color) in enumerate(seeds):
            if n_seeds == 1:
                col_x = base_x
            else:
                col_x = base_x + (s_idx - 1) * col_width

            if grp["kind"] == "baseline":
                files = sorted(glob.glob(grp["dir_pattern"]))
            else:
                prefix = grp["prefix"]
                seed_num = s_idx
                files = sorted(glob.glob(f"work/{prefix}_s{seed_num}/run_seed_*.json"))

            presses = []
            for fpath in files:
                with open(fpath) as f:
                    d = json.load(f)
                presses.append(d["presses_total"])

            presses = np.array(presses)
            median_val = float(np.median(presses))

            # Strip plot points with slight horizontal jitter
            jitter = rng.uniform(-0.035, 0.035, size=len(presses))
            ax.scatter(
                col_x + jitter, presses,
                color=s_color, s=36, alpha=0.75, edgecolor="white", linewidth=0.5,
                zorder=4
            )

            # Mark median with a bold horizontal bar
            bar_w = 0.075
            ax.plot(
                [col_x - bar_w, col_x + bar_w], [median_val, median_val],
                color=s_color, lw=2.6, solid_capstyle="round", zorder=5
            )

            # Label seed under column
            ax.text(
                col_x, 7200, s_label,
                ha="center", va="top", fontsize=7.5, color="#64748b"
            )

    ax.set_xticks(group_x_centers)
    ax.set_xticklabels([grp["name"] for grp in GROUPS], fontsize=8.8, fontweight="bold")
    ax.set_xlim(-0.65, 4.75)
    ax.set_ylim(7000, 13000)
    ax.set_yticks([7000, 8000, 9000, 10000, 11000, 12000, 13000])
    ax.set_yticklabels(["7k", "8k", "9k", "10k", "11k", "12k", "13k"], fontsize=8.5)
    ax.set_ylabel("Total Button Presses to Badge", fontsize=9.0)
    ax.set_title("Quest Execution Length Across Policy Sets (10 runs each)", fontsize=10.0, pad=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748b",
               markersize=7, label="Baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.COLOR_PRIMARY,
               markersize=7, label="Seed 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.COLOR_SECONDARY,
               markersize=7, label="Seed 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.COLOR_SUCCESS,
               markersize=7, label="Seed 2"),
        Line2D([0], [0], color="#334155", lw=2.4, label="Median"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        fontsize=8.5,
        frameon=False,
    )

    style.save(fig, "fig_quest_presses")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
