"""Shared style module for PokeDreamer v3 publication figures.

Provides:
- apply(): sets matplotlib rcParams for clean, consistent 800px-wide figures.
- save(fig, name): writes 200 dpi PNG and SVG to work/figures/<name>.*.
- PALETTE: fixed 6-colour palette.
"""
from pathlib import Path
import matplotlib as mpl

# Fixed 6-colour palette defined once
PALETTE = [
    "#2b5c8f",  # 0: Classic deep blue (primary / learned / seed 0)
    "#d95f02",  # 1: Burnt orange (secondary / copy-last / seed 1)
    "#1b9e77",  # 2: Muted emerald green (success / seed 2)
    "#d62728",  # 3: Crimson red (failure / misread / rework)
    "#7570b3",  # 4: Muted slate purple (special / S2 / baseline)
    "#e6ab02",  # 5: Dark amber / mustard
]

# Semantic color helpers
COLOR_PRIMARY = PALETTE[0]
COLOR_SECONDARY = PALETTE[1]
COLOR_SUCCESS = PALETTE[2]
COLOR_DANGER = PALETTE[3]
COLOR_PURPLE = PALETTE[4]
COLOR_AMBER = PALETTE[5]


def apply():
    """Apply consistent publication rcParams: white background, no top/right spines,
    9-10 pt sans fonts, fixed 6-colour palette, thin grid on y only."""
    mpl.rcParams.update({
        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#718096",
        "axes.linewidth": 0.8,

        # Typography (9-10 pt sans fonts)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,

        # Grid (thin grid on y only)
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "grid.linestyle": "-",

        # Palette
        "axes.prop_cycle": mpl.cycler(color=PALETTE),

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save(fig, name):
    """Write figure to work/figures/<name>.png (200 dpi, tight) and .svg."""
    out_dir = Path("work/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def break_jumps(pts, max_step=1.5):
    """(x, y) points -> x list and y list with NaN wherever consecutive points are more than max_step tiles
    apart (a warp, a respawn, or a gap left by filtering to one map), so a plotted trace never draws a walk
    that did not happen."""
    xs, ys = [], []
    for i, (x, y) in enumerate(pts):
        if i and abs(x - pts[i - 1][0]) + abs(y - pts[i - 1][1]) > max_step:
            xs.append(float("nan")); ys.append(float("nan"))
        xs.append(x); ys.append(y)
    return xs, ys
