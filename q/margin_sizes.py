"""Descriptive (not pre-registered, written after the confirmatory run): size of the frozen split conformal margin q_hat
in degC per lead band and alpha on the test period, and the running example of the paper (KAUS, 25-48 h band, valid hours
12:00-17:59 LST in July 2023). Same frozen method and data as q/run_q.py; no new claim.
Usage: python q/margin_sizes.py test -> results/margin_sizes_<period>.csv, results/margin_example_<period>.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

period = sys.argv[1]
ev, hist = C.load(period)
rows, ex = [], []
for alpha in (0.01, 0.05):
    q = np.asarray(C.bounds(ev, hist, alpha), dtype=float)
    for band in C.BAND_ORDER:
        v = q[(ev["band"].values == band) & np.isfinite(q)]
        rows.append({"alpha": alpha, "band": band, "n": len(v), "p10": np.percentile(v, 10), "median": np.median(v),
                     "p90": np.percentile(v, 90)})
    lst = ev["valid_lst"]
    m = ((ev["station"] == "KAUS") & (ev["band"] == "25-48") & (lst.dt.year == 2023) & (lst.dt.month == 7)
         & (lst.dt.hour >= 12) & (lst.dt.hour < 18)).values
    v = q[m]
    ex.append({"alpha": alpha, "n": int(m.sum()), "median_q": np.median(v), "min_q": v.min(), "max_q": v.max(),
               "median_T_hat": float(np.median(ev.loc[m, "T_hat"])), "covered": float(np.mean(
                   ev.loc[m, "T"].values <= ev.loc[m, "T_hat"].values + v + 1e-9))})
pd.DataFrame(rows).to_csv(C.RES / f"margin_sizes_{period}.csv", index=False)
pd.DataFrame(ex).to_csv(C.RES / f"margin_example_{period}.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(pd.DataFrame(ex).round(2).to_string(index=False))
