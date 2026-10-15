"""Shared figure style: IEEE column widths, >= 7 pt text, TrueType fonts (no Type 3), validated colour-blind-safe palette
(slots 1-4 of the reference categorical order; every series also differs by line style / marker)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_IN = 3.5          # IEEE single column (88.9 mm)
TEXT_IN = 7.16        # IEEE text width (double column)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "ink": "#0b0b0b",
     "muted": "#52514e", "grid": "#d9d8d4"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "xtick.color": C["muted"],
    "ytick.color": C["muted"], "lines.linewidth": 1.4, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name):
    from pathlib import Path
    out = Path(__file__).resolve().parent
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
