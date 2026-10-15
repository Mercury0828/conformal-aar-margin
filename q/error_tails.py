"""Descriptive (not pre-registered) mechanism evidence: tail shape of the temperature forecast errors per lead band.
Usage: python q/error_tails.py test -> results/error_tails_<period>.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
period = sys.argv[1]
ev, _ = C.load(period)
rows = []
for b in C.BAND_ORDER:
    e = ev.loc[ev.band == b, "e"]; z = (e - e.mean()) / e.std()
    rows.append({"band": b, "n": len(e), "mean_e": e.mean(), "sd_e": e.std(), "skew": stats.skew(e),
                 "excess_kurtosis": stats.kurtosis(e), "z_q95": np.quantile(z, 0.95), "z_q99": np.quantile(z, 0.99),
                 "normal_q95": 1.6449, "normal_q99": 2.3263})
pd.DataFrame(rows).to_csv(C.RES / f"error_tails_{period}.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
