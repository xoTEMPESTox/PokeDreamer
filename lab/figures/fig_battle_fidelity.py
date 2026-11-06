"""Figure 4: World model battle fidelity evaluation.

Two panels:
1. Battle-end timing categories as stacked bars (real probe ceiling vs imagined)
2. Battle screen pixel MSE vs horizon k with 95% CI bands for imagined and copy-last baseline
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from lab.figures import style

CAPTION = (
    "Battle fidelity evaluation of wm31: (left) battle-end timing categories showing 71.9% hallucinated "
    "early ends in imagination vs 0.9% for the real probe ceiling; (right) battle-screen pixel MSE vs horizon k "
    "with 95% bootstrap CIs, where copy-last surpasses imagination for k >= 4."
)


def plot():
    style.apply()

    data_file = Path("work/eval/battle_fidelity_wm31_ext.json")
    with open(data_file) as f:
        d = json.load(f)

    fig, (ax_bar, ax_mse) = plt.subplots(1, 2, figsize=(8.0, 4.4), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.86, wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: Battle-end timing categories as stacked bars
    # -------------------------------------------------------------
    b_ceil = d["battle_end"]["probe_ceiling_real_frames"]
    b_img = d["battle_end"]["imagined_frames"]

    categories = [
        ("within_pm3", "Within ±3 steps (accurate)", style.COLOR_SUCCESS),
        ("hallucinated_end", "Hallucinated end (>3 early)", style.COLOR_DANGER),
        ("later_or_never", "Later or never (>3 late)", style.COLOR_SECONDARY),
        ("neither", "Neither ends in H=32", "#64748b"),
    ]

    x_labels = ["Real Probe\n(Ceiling)", "Imagined\nRollout"]
    x_pos = np.array([0, 1])
    bar_width = 0.52

    bottoms = np.zeros(2)
    for key, label, color in categories:
        vals = np.array([b_ceil[key], b_img[key]])
        bars = ax_bar.bar(x_pos, vals, bar_width, bottom=bottoms, label=label, color=color, zorder=3)

        # Print percentage text for segments with enough vertical space
        for idx, (b, val) in enumerate(zip(bars, vals)):
            if val >= 0.08:
                ax_bar.text(
                    b.get_x() + b.get_width() / 2,
                    bottoms[idx] + val / 2,
                    f"{val:.1%}",
                    ha="center", va="center",
                    color="white", fontsize=8.0, fontweight="bold",
                    zorder=4
                )
        bottoms += vals

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(x_labels, fontsize=9.0)
    ax_bar.set_ylabel("Fraction of Battle Windows", fontsize=9.0)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_bar.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_bar.set_title("Battle-End Timing", fontsize=10.0, pad=8)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5, frameon=False)

    # -------------------------------------------------------------
    # Panel 2: Pixel MSE vs horizon k with 95% CI bands
    # -------------------------------------------------------------
    mse_curves = d["pixel_mse"]["curves"]
    m_img = mse_curves["imagined"]
    m_copy = mse_curves["copy_last"]

    ks = np.arange(1, len(m_img["mean"]) + 1)

    # Imagined curve
    ax_mse.plot(ks, m_img["mean"], color=style.COLOR_PRIMARY, lw=1.8, label="Imagined (wm31)", zorder=4)
    ax_mse.fill_between(ks, m_img["lo"], m_img["hi"], color=style.COLOR_PRIMARY, alpha=0.2, zorder=3)

    # Copy-last curve
    ax_mse.plot(ks, m_copy["mean"], color=style.COLOR_SECONDARY, lw=1.8, linestyle="--", label="Copy-last frame", zorder=4)
    ax_mse.fill_between(ks, m_copy["lo"], m_copy["hi"], color=style.COLOR_SECONDARY, alpha=0.2, zorder=3)

    # Crossover annotation
    crossover_k = 3.5
    ax_mse.axvline(crossover_k, color="#94a3b8", linestyle=":", lw=1.2, zorder=2)
    ax_mse.annotate(
        "Copy-last beats\nimagination (k ≥ 4)",
        xy=(crossover_k, 0.088),
        xytext=(crossover_k + 2.5, 0.065),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0),
        fontsize=7.8, fontweight="bold", color="#334155",
        zorder=5
    )

    ax_mse.set_xlabel("Horizon k (steps)", fontsize=9.0)
    ax_mse.set_ylabel("Inside-Battle Pixel MSE", fontsize=9.0)
    ax_mse.set_xlim(1, 32)
    ax_mse.set_xticks([1, 4, 8, 16, 24, 32])
    ax_mse.set_ylim(0.0, 0.25)
    ax_mse.set_title("Battle-Screen Pixel MSE vs Horizon", fontsize=10.0, pad=8)
    ax_mse.legend(loc="upper left", fontsize=8.0, frameon=False)

    style.save(fig, "fig_battle_fidelity")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
