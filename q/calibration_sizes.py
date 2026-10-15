"""Descriptive / exploratory (not pre-registered; written after the confirmatory run):
(1) how often the frozen split margin falls back from the 90-day to the 730-day window, the calibration size n actually
used and the rank position k relative to n (k = n means the margin is the sample maximum), per lead band and alpha;
(2) a diagnostic Gaussian prediction bound on the identical calibration errors, mean + t_{n-1,1-alpha} s sqrt(1 + 1/n),
which adds the finite-sample correction that the rolling Gaussian (B3r-n, z quantile) lacks. Exploratory: it is not a
pre-registered comparator and does not replace B3r-n. Same frozen method and data as q/run_q.py.
Usage: python q/calibration_sizes.py test -> results/calibration_sizes_<period>.csv, results/gauss_tpred_<period>.csv"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from aarguard import bound  # noqa: E402

period = sys.argv[1]
ev, hist = C.load(period)
groups = {k: g.sort_values("valid") for k, g in hist.groupby(["station", "band", "cell"])}
rows_n, rows_g = [], []
for alpha in C.ALPHAS:
    n90 = pd.Series(0, index=ev.index); n_used = pd.Series(0, index=ev.index)
    k_eq_n = pd.Series(False, index=ev.index); q_t = pd.Series(np.nan, index=ev.index)
    for key, tg in ev.groupby(["station", "band", "cell"]):
        g = groups[key]
        cv, ca, ce = bound._naive(g["valid"].values), bound._naive(g["t_obs"].values), g["e"].values
        for tau, idx in tg.groupby("issue").groups.items():
            t64 = bound._naive(np.array([tau]))[0]
            e90 = bound.available_errors(cv, ca, ce, t64, 90)
            e = bound.selected_errors(cv, ca, ce, t64, alpha, 90, bound.W_MAX_DAYS)
            n = len(e); k = math.ceil((n + 1) * (1 - alpha))
            n90.loc[idx] = len(e90); n_used.loc[idx] = n; k_eq_n.loc[idx] = (k == n)
            if n > 1:
                q_t.loc[idx] = e.mean() + stats.t.ppf(1 - alpha, n - 1) * e.std(ddof=1) * math.sqrt(1 + 1 / n)
    fb = (np.ceil((n90 + 1) * (1 - alpha)) > n90)
    r_lo = C.rating_at(ev, (ev["T_hat"] + q_t).values, *C.PRIMARY)
    r_obs = ev["R_obs_Drake_75"].values
    r_hat = ev["R_hat_Drake_75"].values
    for b in C.BAND_ORDER:
        m = (ev["band"] == b).values
        rows_n.append({"alpha": alpha, "band": b, "cases": int(m.sum()), "fallback_share": float(fb[m].mean()),
                       "n_min": int(n_used[m].min()), "n_median": float(n_used[m].median()), "n_max": int(n_used[m].max()),
                       "n90_median": float(n90[m].median()), "share_k_eq_n": float(k_eq_n[m].mean())})
        rows_g.append({"alpha": alpha, "band": b, "cov_tpred": float((r_lo[m] <= r_obs[m] + 1e-9).mean()),
                       "CC_tpred": float((r_hat[m] - r_lo[m]).mean() / r_obs[m].mean())})
out_n, out_g = pd.DataFrame(rows_n), pd.DataFrame(rows_g)
out_n.to_csv(C.RES / f"calibration_sizes_{period}.csv", index=False)
out_g.to_csv(C.RES / f"gauss_tpred_{period}.csv", index=False)
print(out_n.round(3).to_string(index=False))
print(out_g.round(4).to_string(index=False))
