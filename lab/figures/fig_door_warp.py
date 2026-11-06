"""Figure 3: Imagined vs real map-change rates at doors and mats.

Heatmap comparison per location (Pewter PC door, Viridian PC door, Pewter PC mat, Viridian PC mat):
- Two heatmaps side-by-side per location: "Imagined" and "Real game"
- One row per tile and one column per button (Down, Left, Right, Up)
- Rates printed in every cell ("n/a" if n == 0)
- Red outline for cells where imagined - real >= 0.3 (hallucinated warps paying reward)
- Shared 0..1 color scale
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

from lab.figures import style

BUTTONS = ["down", "left", "right", "up"]
BUTTON_LABELS = ["Down", "Left", "Right", "Up"]

# Locations and their tiles with descriptive role names
LOCATIONS = [
    {
        "name": "Pewter PC Door",
        "key_prefix": "pewter_pc door",
        "tiles": [
            ("(13,26)", "(13,26) door tile", True),
            ("(12,26)", "(12,26) beside", False),
            ("(14,26)", "(14,26) beside", False),
            ("(13,27)", "(13,27) below", False),
        ],
    },
    {
        "name": "Viridian PC Door (Control)",
        "key_prefix": "viridian_pc door",
        "tiles": [
            ("(23,26)", "(23,26) door tile", True),
            ("(22,26)", "(22,26) beside", False),
            ("(24,26)", "(24,26) beside", False),
            ("(23,27)", "(23,27) below", False),
        ],
    },
    {
        "name": "Pewter PC Mat",
        "key_prefix": "pewter_pc mat",
        "tiles": [
            ("(3,7)", "(3,7) mat", True),
            ("(4,7)", "(4,7) mat", True),
            ("(2,7)", "(2,7) beside", False),
            ("(5,7)", "(5,7) beside", False),
            ("(3,6)", "(3,6) above", False),
            ("(4,6)", "(4,6) above", False),
        ],
    },
    {
        "name": "Viridian PC Mat (Control)",
        "key_prefix": "viridian_pc mat",
        "tiles": [
            ("(3,7)", "(3,7) mat", True),
            ("(4,7)", "(4,7) mat", True),
            ("(2,7)", "(2,7) beside", False),
            ("(5,7)", "(5,7) beside", False),
            ("(3,6)", "(3,6) above", False),
            ("(4,6)", "(4,6) above", False),
        ],
    },
]

CAPTION = (
    "Imagined vs real map-change rates per action across Pokémon Center door and mat tile groups; "
    "cells outlined in red indicate imagined map changes that never happen in the real game (imagined − real ≥ 0.3), "
    "which erroneously paid training reward."
)


def render_heatmap(ax, matrix, valid_mask, outlines, row_labels, col_labels, title, show_yticklabels=True):
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    # Grid lines separating cells
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5)

    if show_yticklabels:
        ax.set_yticklabels(row_labels, fontsize=7.5)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=4)

    # Minor grid to frame cells neatly
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="major", visible=False)

    # Print rate in every cell
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if valid_mask[r, c]:
                val = matrix[r, c]
                text_color = "white" if val > 0.55 else "#0f172a"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.2, fontweight="bold", color=text_color, zorder=5)
            else:
                ax.text(c, r, "n/a", ha="center", va="center",
                        fontsize=7.0, color="#94a3b8", zorder=5)

            # Red outline where imagined - real >= 0.3
            if outlines[r, c]:
                rect = Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False,
                                 edgecolor="#dc2626", linewidth=2.2, zorder=10)
                ax.add_patch(rect)

    return im


def plot():
    style.apply()

    data_file = Path("work/eval/door_warp_wm31.json")
    with open(data_file) as f:
        d = json.load(f)

    # 2 rows (Doors: 4 tiles, Mats: 6 tiles) x 2 location columns
    fig = plt.figure(figsize=(9.8, 6.0), dpi=200)
    gs = gridspec.GridSpec(
        2, 5,
        height_ratios=[4, 6],
        width_ratios=[4, 4, 2.4, 4, 4],
        left=0.13, right=0.91, bottom=0.08, top=0.88,
        wspace=0.15, hspace=0.38
    )

    im_ref = None

    for loc_idx, loc in enumerate(LOCATIONS):
        row_idx = loc_idx // 2
        col_base = (loc_idx % 2) * 3

        prefix = loc["key_prefix"]
        tile_tuples = loc["tiles"]
        n_rows = len(tile_tuples)
        n_cols = len(BUTTONS)

        mat_im = np.zeros((n_rows, n_cols), dtype=float)
        mat_re = np.zeros((n_rows, n_cols), dtype=float)
        val_re = np.zeros((n_rows, n_cols), dtype=bool)
        outlines = np.zeros((n_rows, n_cols), dtype=bool)

        row_names = [label for _, label, _ in tile_tuples]

        for r_i, (t_coord, _, _) in enumerate(tile_tuples):
            key = f"{prefix} {t_coord}"
            tile_data = d["tiles"].get(key, {})
            im_dict = tile_data.get("imagined", {})
            re_dict = tile_data.get("real", {})

            for c_i, b in enumerate(BUTTONS):
                im_val = im_dict.get(b, 0.0)
                mat_im[r_i, c_i] = im_val

                re_btn = re_dict.get(b, {})
                re_n = re_btn.get("n", 0)
                re_warp = re_btn.get("warp", None)

                if re_n > 0 and re_warp is not None:
                    mat_re[r_i, c_i] = re_warp
                    val_re[r_i, c_i] = True
                    diff = im_val - re_warp
                else:
                    val_re[r_i, c_i] = False
                    diff = im_val

                if diff >= 0.3:
                    outlines[r_i, c_i] = True

        ax_im = fig.add_subplot(gs[row_idx, col_base])
        ax_re = fig.add_subplot(gs[row_idx, col_base + 1])

        im_ref = render_heatmap(
            ax_im, mat_im, np.ones_like(mat_im, dtype=bool), outlines,
            row_names, BUTTON_LABELS, f"{loc['name']}\nImagined",
            show_yticklabels=True
        )

        render_heatmap(
            ax_re, mat_re, val_re, np.zeros_like(outlines),
            row_names, BUTTON_LABELS, f"{loc['name']}\nReal Game",
            show_yticklabels=False
        )

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.93, 0.18, 0.016, 0.64])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label("Map-Change Rate", fontsize=9.0, labelpad=6)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8.0)

    # Legend proxy indicating red outline meaning
    legend_rect = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#dc2626", lw=2.2)
    fig.legend(
        [legend_rect],
        ["Imagined − Real ≥ 0.3 (hallucinated warps paying spurious reward)"],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        fontsize=8.5,
        frameon=False,
    )

    style.save(fig, "fig_door_warp")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    caption = main()
    print(caption)
