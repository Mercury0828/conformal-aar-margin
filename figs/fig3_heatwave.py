"""Fig. 3 — the longest statewide heat wave of the test period (Q4 run 1, 2023-07-09 to 2023-09-10, KBPT, 25–48 h band,
Drake / 75 degC, alpha = 5 %): realized-temperature AAR, raw forecast AAR, conformal lower bound and static seasonal rating
at the daily peak-heat valid time 21 UTC (15:00 LST) of the frozen 3-hourly trace; gaps are kept. Data: results/Q4_trace_test.csv. Regenerate: python figs/fig3_heatwave.py"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "figs"))
from style import COL_IN, C, plt, save  # noqa: E402

t = pd.read_csv(REPO / "results" / "Q4_trace_test.csv", parse_dates=["valid"])
t = t[t.run == 1].sort_values("valid")
n_all, miss_all = len(t), int((t.R_lo > t.R_obs_Drake_75 + 1e-9).sum())
t = t[t.valid.dt.hour == 21]
fig, ax = plt.subplots(figsize=(COL_IN, 1.9))
ax.plot(t.valid, t.R_stat_Drake_75, color=C["muted"], ls=":", lw=1.0, label="Static rating")
ax.plot(t.valid, t.R_hat_Drake_75, color=C["orange"], ls="--", lw=0.9, label="Forecast rating")
ax.plot(t.valid, t.R_obs_Drake_75, color=C["blue"], ls="-", lw=0.9, label="Realized-temperature rating")
ax.plot(t.valid, t.R_lo, color=C["aqua"], ls="-.", lw=1.3, label="Conformal rating")
miss = t[t.R_lo > t.R_obs_Drake_75 + 1e-9]
ax.plot(miss.valid, miss.R_lo, "v", ms=3.5, color=C["ink"], label="Conformal above realized", zorder=5)
ax.set_ylabel("Rating (A)")
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_xlim(t.valid.min(), t.valid.max())
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.36), handlelength=1.8, columnspacing=0.8, fontsize=7.2)
save(fig, "fig3_heatwave")
print(len(t), "points at 21 UTC;", len(miss), "with R_lo > R there;", miss_all, "of", n_all, "over the full 3-hourly trace")
