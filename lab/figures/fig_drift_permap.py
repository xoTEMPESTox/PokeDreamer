"""Figure: Imagination drift and map hallucination per leg map.

From work/eval/wm31_test_ext_s3_a0_ext_permap.npz.
Two rows:
1. Position error on the right map vs k (1..32, log2 x ticks) for:
   Viridian City (1), Route 2 (13), Viridian Forest (51), Pewter City (2)
   - imagined (solid, CI band)
   - no-change (dashed, CI band)
   - probe ceiling (dotted)
2. Wrong-map rate vs k for maps 1, 13, 51, 2, 41 (Viridian PC), 58 (Pewter PC)
   in one panel, one line per map, CI bands (mean only for maps with < 3 episodes).
Shaded region k <= 10 in all panels labeled 'imagination RL horizon'.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from lab.eval.metrics import bootstrap_ci
from lab.figures import style

CAPTION = (
    "Imagination drift and map hallucination per leg map across rollout horizon k: in Viridian Forest (map 51), "
    "position error on the correct map climbs to 15.65 tiles at k=8 and >20 tiles at k=32—far exceeding the probe's "
    "own error on real forest frames (about 2 tiles), confirming error originates in world model dynamics rather than "
    "measurement—while Pewter PC (map 58) suffers severe map hallucination, read as an incorrect map 76% of the time "
    "at k=8 (vs 20% in Viridian PC)."
)

TOP_MAPS = [
    (1, "Viridian City (1)"),
    (13, "Route 2 (13)"),
    (51, "Viridian Forest (51)"),
    (2, "Pewter City (2)"),
]

BOTTOM_MAPS = [
    (1, "Viridian City (1)", style.PALETTE[0]),
    (13, "Route 2 (13)", style.PALETTE[1]),
    (51, "Viridian Forest (51)", style.PALETTE[2]),
    (2, "Pewter City (2)", style.PALETTE[3]),
    (41, "Viridian PC (41)", style.PALETTE[4]),
    (58, "Pewter PC (58)", style.PALETTE[5]),
]


def load_drift_data(npz_path):
    d = np.load(npz_path)
    map0 = d["map0"]
    eps = d["episode"]

    valid = ~np.isnan(d["pos_err_img"])
    right = valid & (d["img_map"] == d["true_map"])
    err_right = np.where(right, d["pos_err_img"], np.nan)
    nochange_right = np.where(right, d["pos_err_nochange"], np.nan)
    wrong_map = np.where(valid, (d["img_map"] != d["true_map"]).astype(float), np.nan)

    probe_valid = ~np.isnan(d["pos_err_real"]) & (d["real_map"] == d["true_map"])
    probe_ceiling = np.where(probe_valid, d["pos_err_real"], np.nan)

    metrics_by_map = {}
    all_map_ids = set([m for m, _ in TOP_MAPS] + [m for m, _, _ in BOTTOM_MAPS])

    for m in all_map_ids:
        sel = (map0 == m)
        eps_m = np.unique(eps[sel])
        n_eps = len(eps_m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ep_err_img = np.array([np.nanmean(err_right[sel & (eps == e)], axis=0) for e in eps_m])
            ep_nochange = np.array([np.nanmean(nochange_right[sel & (eps == e)], axis=0) for e in eps_m])
            ep_wrong = np.array([np.nanmean(wrong_map[sel & (eps == e)], axis=0) for e in eps_m])
            ep_probe = np.array([np.nanmean(probe_ceiling[sel & (eps == e)], axis=0) for e in eps_m])

        entry = {"n_eps": n_eps}
        for key, arr in [("img", ep_err_img), ("nochange", ep_nochange),
                         ("wrong", ep_wrong), ("probe", ep_probe)]:
            if n_eps >= 3:
                m_val, lo, hi = bootstrap_ci(arr)
            else:
                m_val = np.nanmean(arr, axis=0)
                lo, hi = None, None
            entry[key] = (m_val, lo, hi)

        metrics_by_map[m] = entry

    return metrics_by_map


def plot():
    style.apply()

    data_file = Path("work/eval/wm31_test_ext_s3_a0_ext_permap.npz")
    metrics = load_drift_data(data_file)

    fig = plt.figure(figsize=(9.2, 5.8), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.38, wspace=0.28,
                           left=0.07, right=0.96, top=0.91, bottom=0.08)

    ks = np.arange(1, 33)
    tick_ks = [1, 2, 4, 8, 16, 32]

    # -------------------------------------------------------------
    # Top Row: Position Error on Right Map vs k (4 panels)
    # -------------------------------------------------------------
    top_axes = []
    for idx, (m, name) in enumerate(TOP_MAPS):
        ax = fig.add_subplot(gs[0, idx])
        top_axes.append(ax)

        # Shading for RL horizon
        ax.axvspan(1, 10, color="#f1f5f9", alpha=0.9, zorder=0)

        # Imagined error
        m_img, lo_img, hi_img = metrics[m]["img"]
        ax.plot(ks, m_img, color=style.COLOR_PRIMARY, lw=1.6, linestyle="-",
                label="Imagined", zorder=3)
        if lo_img is not None and hi_img is not None:
            ax.fill_between(ks, lo_img, hi_img, color=style.COLOR_PRIMARY, alpha=0.2, zorder=2)

        # No-change baseline
        m_nc, lo_nc, hi_nc = metrics[m]["nochange"]
        ax.plot(ks, m_nc, color=style.COLOR_SECONDARY, lw=1.5, linestyle="--",
                label="No-change", zorder=3)
        if lo_nc is not None and hi_nc is not None:
            ax.fill_between(ks, lo_nc, hi_nc, color=style.COLOR_SECONDARY, alpha=0.18, zorder=2)

        # Probe ceiling
        m_pr, _, _ = metrics[m]["probe"]
        ax.plot(ks, m_pr, color="#475569", lw=1.4, linestyle=":",
                label="Probe ceiling", zorder=3)

        # Scale & ticks
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 32)
        ax.set_xticks(tick_ks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title(name, fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlabel("Horizon k", fontsize=8.0)

        # Shading label in each panel
        y_lim = ax.get_ylim()
        y_max = max(y_lim[1], 5.0)
        ax.set_ylim(bottom=0, top=y_max)
        ax.text(3.16, y_max * 0.90, "imagination\nRL horizon\n(k ≤ 10)", ha="center", va="top",
                fontsize=6.8, color="#64748b", zorder=4)

        if idx == 0:
            ax.set_ylabel("Position error (tiles)", fontsize=8.5)

    # Top row shared legend
    top_legend_elements = [
        Line2D([0], [0], color=style.COLOR_PRIMARY, lw=1.6, linestyle="-", label="Imagined (95% CI)"),
        Line2D([0], [0], color=style.COLOR_SECONDARY, lw=1.5, linestyle="--", label="No-change baseline (95% CI)"),
        Line2D([0], [0], color="#475569", lw=1.4, linestyle=":", label="Probe ceiling (real frames)"),
        Patch(facecolor="#f1f5f9", edgecolor="#cbd5e1", label="Imagination RL horizon (k ≤ 10)"),
    ]
    fig.legend(
        handles=top_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.99),
        ncol=4,
        fontsize=8.0,
        frameon=False,
    )

    # -------------------------------------------------------------
    # Bottom Row: Wrong-Map Rate vs k (1 panel across gs[1, :])
    # -------------------------------------------------------------
    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.axvspan(1, 10, color="#f1f5f9", alpha=0.9, zorder=0)

    for m, name, color in BOTTOM_MAPS:
        m_wr, lo_wr, hi_wr = metrics[m]["wrong"]
        n_eps = metrics[m]["n_eps"]

        if n_eps >= 3:
            ax_bot.plot(ks, m_wr, color=color, lw=1.6, linestyle="-",
                        label=f"{name} ({n_eps} eps)", zorder=3)
            if lo_wr is not None and hi_wr is not None:
                ax_bot.fill_between(ks, lo_wr, hi_wr, color=color, alpha=0.18, zorder=2)
        else:
            # Mean only, dashed thin, say so in legend
            ax_bot.plot(ks, m_wr, color=color, lw=1.3, linestyle="--",
                        label=f"{name} ({n_eps} eps, mean only, <3 eps)", zorder=3)

    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xlim(1, 32)
    ax_bot.set_xticks(tick_ks)
    ax_bot.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_xlabel("Horizon k (rollout steps)", fontsize=8.5)
    ax_bot.set_ylabel("Wrong-map rate", fontsize=8.5)
    ax_bot.set_title("Wrong-Map Rate vs Horizon k (Map Hallucination / Classification Drift)",
                      fontsize=9.0, fontweight="bold", pad=4)

    # Label RL horizon in bottom panel
    ax_bot.text(1.15, 0.94, "imagination RL horizon (k ≤ 10)", ha="left", va="top",
                fontsize=8.0, color="#64748b", zorder=4)

    ax_bot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.0, frameon=False)

    style.save(fig, "fig_drift_permap")
    plt.close(fig)
    return CAPTION


def main():
    return plot()


if __name__ == "__main__":
    c = main()
    print("Caption:", c)
