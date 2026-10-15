"""Fig. 1 — how a one-sided temperature bound becomes a rating lower bound (illustration on the frozen rating map:
Drake ACSR, T_max 75 degC, KAUS latitude / elevation, declared inputs). Regenerate: python figs/fig1_rating_map.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "figs"))
from style import COL_IN, C, plt, save  # noqa: E402
from aarguard.rating import DRAKE, ampacity, station_inputs  # noqa: E402

st = pd.read_csv(REPO / "data" / "stations.csv").set_index("icao").loc["KAUS"]
inp = station_inputs(st.lat, st.elev_m)
T = np.linspace(0, 45, 451)
day = ampacity(T, DRAKE, 75.0, inp, True)
night = ampacity(T, DRAKE, 75.0, inp, False)
t_hat, q_hat = 33.0, 3.0            # illustrative forecast and margin
t_plus = t_hat + q_hat
r_hat, r_lo = float(ampacity(t_hat, DRAKE, 75.0, inp, True)), float(ampacity(t_plus, DRAKE, 75.0, inp, True))

fig, ax = plt.subplots(figsize=(COL_IN, 2.0))
ax.grid(False)
ax.plot(T, night, color=C["blue"], ls="--", label="Night (no solar heating)")
ax.plot(T, day, color=C["orange"], ls="-", label="Day")
for x, r in ((t_hat, r_hat), (t_plus, r_lo)):
    ax.plot([x, x], [600, r], color=C["muted"], lw=0.7, ls=":")
    ax.plot([0, x], [r, r], color=C["muted"], lw=0.7, ls=":")
    ax.plot(x, r, "o", ms=4, color=C["ink"], zorder=5)
ax.text(1.0, r_hat + 10, f"Forecast rating ({r_hat:.0f} A)", fontsize=7.5, color=C["ink"], va="bottom")
ax.text(1.0, r_lo - 12, f"Rating with margin ({r_lo:.0f} A)", fontsize=7.5, color=C["ink"], va="top")
ax.annotate("", xy=(t_plus, 648), xytext=(t_hat, 648), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.8))
ax.text((t_hat + t_plus) / 2, 668, r"$\hat q$", ha="center", va="bottom", fontsize=8)
ax.text(t_hat - 0.6, 606, r"$\hat T$", ha="right", va="bottom", fontsize=8)
ax.text(t_plus + 0.6, 606, r"$\hat T + \hat q$", ha="left", va="bottom", fontsize=8)
ax.set_xlabel(r"Ambient temperature $T$ (°C)")
ax.set_ylabel("Steady-state rating (A)")
ax.set_xlim(0, 45); ax.set_ylim(600, 1250)
ax.legend(loc="upper right")
save(fig, "fig1_rating_map")
print(f"R_hat={r_hat:.0f} A, R_lo={r_lo:.0f} A, drop={100 * (1 - r_lo / r_hat):.1f} %")
