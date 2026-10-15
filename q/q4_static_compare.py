"""Descriptive (not pre-registered, written after the confirmatory run): on the frozen Q4 heat-wave traces, how often the
static seasonal rating, the forecast rating and the conformal rating exceeded the realized-temperature rating, and their
mean values (Drake / 75 degC, 25-48 h band, alpha = 5 %). Reads results/Q4_trace_test.csv only.
Usage: python q/q4_static_compare.py -> results/Q4_static_compare_test.csv"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
t = pd.read_csv(RES / "Q4_trace_test.csv")
t = t[t["R_obs_Drake_75"].notna()]
rows = []
for run, x in t.groupby("run"):
    r = x["R_obs_Drake_75"] + 1e-9
    rows.append({"run": run, "station": x["station"].iloc[0], "first_valid": x["valid"].min(), "last_valid": x["valid"].max(),
                 "n": len(x), "static_above": int((x["R_stat_Drake_75"] > r).sum()),
                 "forecast_above": int((x["R_hat_Drake_75"] > r).sum()), "conformal_above": int((x["R_lo"] > r).sum()),
                 "mean_static": x["R_stat_Drake_75"].mean(), "mean_conformal": x["R_lo"].mean(),
                 "mean_forecast": x["R_hat_Drake_75"].mean(), "mean_realized": x["R_obs_Drake_75"].mean(),
                 "share_conformal_above_static": float((x["R_lo"] > x["R_stat_Drake_75"]).mean())})
    pk = x[pd.to_datetime(x["valid"]).dt.hour == 21]  # 15:00 LST, the hours drawn in the heat-wave figure
    rp = pk["R_obs_Drake_75"] + 1e-9
    rows[-1].update({"peak_n": len(pk), "peak_static_above": int((pk["R_stat_Drake_75"] > rp).sum()),
                     "peak_conformal_above": int((pk["R_lo"] > rp).sum()),
                     "peak_share_conformal_below_static": float((pk["R_lo"] < pk["R_stat_Drake_75"]).mean())})
out = pd.DataFrame(rows)
out.to_csv(RES / "Q4_static_compare_test.csv", index=False)
print(out.round(3).to_string(index=False))
