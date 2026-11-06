"""Figure 5: Measured GPU compute ledger and Path A allocation.

Two horizontal stacked bars:
1. Measured GPU hours by kind (world model 9.8, imagination RL 26.1, upsampler 0.8, LLM System 2 3.9; total 40.6 h)
2. Path A stretch split (planned 6.3 / scope added 21.4 / rework 13.5 h; total 41.2 h)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from lab.figures import style

# Numbers from agent_notes/PROGRESS.md 2026-11-05 21:20
GPU_BY_KIND = [
    ("World Model", 9.8, style.COLOR_PRIMARY),       # #2b5c8f
    ("Imagination RL", 26.1, style.COLOR_SECONDARY),  # #d95f02
    ("Upsampler", 0.8, style.COLOR_SUCCESS),         # #1b9e77
    ("LLM System 2", 3.9, style.COLOR_PURPLE),       # #7570b3
]
TOTAL_KIND = sum(v for _, v, _ in GPU_BY_KIND)  # 40.6 h

GPU_BY_SPLIT = [
    ("Planned work", 6.3, style.COLOR_PRIMARY),       # #2b5c8f
    ("Scope added", 21.4, style.COLOR_AMBER),         # #e6ab02
    ("Rework", 13.5, style.COLOR_DANGER),             # #d62728
]
TOTAL_SPLIT = sum(v for _, v, _ in GPU_BY_SPLIT)  # 41.2 h

CAPTION = (
    "Measured GPU compute accounting for the project: (top) 40.6 total hours broken down by component kind, "
    "dominated by imagination RL; (bottom) Path A stretch allocation across planned work (6.3 h), added scope "
    "(21.4 h), and debugging rework (13.5 h)."
)


def plot():
    style.apply()

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.18, top=0.84)

    # For horizontal bars, switch grid to x only
    ax.grid(True, axis="x", color="#e2e8f0", lw=0.6, alpha=0.8)

    y_pos = [1.2, 0.0]
    bar_height = 0.50

    # -------------------------------------------------------------
    # Bar 1: By Kind
    # -------------------------------------------------------------
    left = 0.0
    for name, val, color in GPU_BY_KIND:
        frac = val / TOTAL_KIND
        bar = ax.barh(y_pos[0], val, bar_height, left=left, color=color, edgecolor="white", lw=1.2, zorder=3)

        # Labels inside segments
        cx = left + val / 2
        if val >= 5.0:
            ax.text(cx, y_pos[0], f"{name}\n{val:.1f} h ({frac:.0%})",
                    ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", zorder=4)
        elif name == "LLM System 2":
            ax.text(cx, y_pos[0], f"LLM S2\n{val:.1f} h",
                    ha="center", va="center", color="white", fontsize=7.0, fontweight="bold", zorder=4)
        else:
            # Narrow segment: callout above with generous headroom
            ax.annotate(
                f"{name}\n{val:.1f} h",
                xy=(cx, y_pos[0] + bar_height / 2),
                xytext=(cx - 1.5, y_pos[0] + bar_height / 2 + 0.22),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=0.8),
                ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#1e293b",
                zorder=5
            )
        left += val

    # -------------------------------------------------------------
    # Bar 2: Path A Stretch Split
    # -------------------------------------------------------------
    left = 0.0
    for name, val, color in GPU_BY_SPLIT:
        frac = val / TOTAL_SPLIT
        bar = ax.barh(y_pos[1], val, bar_height, left=left, color=color, edgecolor="white", lw=1.2, zorder=3)

        cx = left + val / 2
        if val >= 10.0:
            ax.text(cx, y_pos[1], f"{name}\n{val:.1f} h ({frac:.0%})",
                    ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", zorder=4)
        else:
            ax.text(cx, y_pos[1], f"Planned\n{val:.1f} h ({frac:.0%})",
                    ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=4)
        left += val

    ax.set_ylim(-0.55, 1.95)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        f"Measured by Kind\n({TOTAL_KIND:.1f} h total)",
        f"Path A Stretch Split\n({TOTAL_SPLIT:.1f} h total)",
    ], fontsize=9.0, fontweight="bold")

    ax.set_xlim(0, 48)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("GPU Hours (RTX 4090 / CUDA)", fontsize=9.0)
    ax.set_title("PokeDreamer v3 GPU Compute Ledger", fontsize=10.0, pad=12)

    # Reference dotted lines at totals
    ax.axvline(TOTAL_KIND, color="#94a3b8", linestyle=":", lw=1.0, zorder=2)

    style.save(fig, "fig_gpu_ledger")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
