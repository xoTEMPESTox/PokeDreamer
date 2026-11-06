"""Figure 2: Success rate and median steps per navigation leg.

Dot plot comparing:
- Gate runs across seeds 0, 1, and 2
- Greedy geodesic baseline as a hollow marker
- Read from work/agent/eval_<run>.log and work/agent/<leg>_greedy_geo_s200.json
"""
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from lab.figures import style

LEGS = [
    ("to_route2", "to_route2"),
    ("to_gate", "to_gate"),
    ("forest", "forest"),
    ("to_pewter", "to_pewter"),
    ("to_gym", "to_gym"),
    ("to_grass", "to_grass"),
    ("to_viridian_pc", "to_viridian_pc"),
    ("leave_viridian_pc", "leave_viridian_pc"),
    ("to_pewter_pc", "to_pewter_pc"),
    ("leave_pewter_pc", "leave_pewter_pc"),
]

CAPTION = (
    "Evaluation success rate (/100) and median steps across seeds 0, 1, and 2 of the "
    "learned gate policies compared against the greedy geodesic baseline (hollow markers)."
)


def get_run_name(leg: str, seed: int) -> str:
    if leg == "to_pewter_pc":
        return f"{leg}_wm31_gate_it1800_s{seed}"
    elif leg == "leave_pewter_pc":
        return f"{leg}_wm31_gate_mc_s{seed}"
    else:
        return f"{leg}_wm31_gate_s{seed}"


def parse_eval_log(log_path: Path):
    lines = [l.strip() for l in log_path.read_text().splitlines() if l.strip()]
    last = lines[-1]
    m = re.search(r"(\d+)/100 successes;\s*median steps\s+([\d\.]+|None)", last)
    if not m:
        raise ValueError(f"Could not parse eval log line: {last}")
    succ = int(m.group(1))
    steps_str = m.group(2)
    steps = float(steps_str) if steps_str != "None" else None
    return succ, steps


def plot():
    style.apply()

    data = []
    for leg, display_name in LEGS:
        # Greedy baseline
        greedy_file = Path(f"work/agent/{leg}_greedy_geo_s200.json")
        with open(greedy_file) as f:
            g_data = json.load(f)
        g_succ = g_data.get("successes", 0)
        g_steps = g_data.get("median_steps_when_successful", None)

        # Seeds 0, 1, 2
        seed_res = []
        for s in (0, 1, 2):
            run_name = get_run_name(leg, s)
            log_path = Path(f"work/agent/eval_{run_name}.log")
            succ, steps = parse_eval_log(log_path)
            seed_res.append((succ, steps))

        data.append({
            "leg": leg,
            "display": display_name,
            "greedy": (g_succ, g_steps),
            "seeds": seed_res,
        })

    fig, (ax_succ, ax_steps) = plt.subplots(1, 2, figsize=(8.0, 4.4), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.14, top=0.86, wspace=0.12)

    y_positions = np.arange(len(LEGS))
    # Reverse so top-down reads route2 -> pewter
    y_positions = y_positions[::-1]

    # Vertical offsets to prevent overlap of identical points (e.g. at 100%)
    offsets = [-0.18, -0.06, 0.06, 0.18]
    seed_colors = [style.COLOR_PRIMARY, style.COLOR_SECONDARY, style.COLOR_SUCCESS]

    for idx, item in enumerate(data):
        y = y_positions[idx]

        # Faint horizontal row guide
        ax_succ.axhline(y, color="#edf2f7", lw=1.2, zorder=1)
        ax_steps.axhline(y, color="#edf2f7", lw=1.2, zorder=1)

        # Greedy baseline marker
        g_succ, g_steps = item["greedy"]
        ax_succ.scatter(
            g_succ, y + offsets[0], marker="o", s=45, facecolors="none",
            edgecolors="#4a5568", linewidths=1.4, zorder=4
        )
        if g_steps is not None:
            ax_steps.scatter(
                g_steps, y + offsets[0], marker="o", s=45, facecolors="none",
                edgecolors="#4a5568", linewidths=1.4, zorder=4
            )

        # Seeds 0, 1, 2
        for s, (s_succ, s_steps) in enumerate(item["seeds"]):
            ax_succ.scatter(
                s_succ, y + offsets[s + 1], marker="o", s=38, color=seed_colors[s],
                alpha=0.9, zorder=5
            )
            if s_steps is not None:
                ax_steps.scatter(
                    s_steps, y + offsets[s + 1], marker="o", s=38, color=seed_colors[s],
                    alpha=0.9, zorder=5
                )
            else:
                # 0 successes (e.g. to_pewter_pc seed 1): mark at 0 with an 'x'
                ax_steps.scatter(
                    0, y + offsets[s + 1], marker="x", s=40, color=style.COLOR_DANGER,
                    lw=1.5, zorder=5
                )

    # Configure ax_succ
    ax_succ.set_xlim(-4, 106)
    ax_succ.set_xticks([0, 20, 40, 60, 80, 100])
    ax_succ.set_xlabel("Successes / 100", fontsize=9.0)
    ax_succ.set_title("Evaluation Success Rate", fontsize=9.5, pad=8)
    ax_succ.set_yticks(y_positions)
    ax_succ.set_yticklabels([item["display"] for item in data], fontsize=9.0)

    # Configure ax_steps
    ax_steps.set_xlim(-20, 720)
    ax_steps.set_xticks([0, 150, 300, 450, 600])
    ax_steps.set_xlabel("Median Steps", fontsize=9.0)
    ax_steps.set_title("Median Steps When Successful", fontsize=9.5, pad=8)
    ax_steps.spines["left"].set_visible(False)

    # Unified legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#4a5568",
               markerfacecolor="none", markeredgewidth=1.4, markersize=7, label="Greedy baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=seed_colors[0],
               markersize=7, label="Seed 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=seed_colors[1],
               markersize=7, label="Seed 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=seed_colors[2],
               markersize=7, label="Seed 2"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )

    style.save(fig, "fig_leg_success")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
