"""Descriptive (not pre-registered, written after the confirmatory run): case-weighted exposure of the forecast rating by
season, pooled over stations and bands, from the frozen registered table results/Q1_station_band_season_test.csv.
Usage: python q/exposure_by_season.py -> results/Q1_season_pooled_test.csv"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
t = pd.read_csv(RES / "Q1_station_band_season_test.csv")
w = lambda g, c: float((g[c] * g["n"]).sum() / g["n"].sum())  # noqa: E731
out = pd.DataFrame([{"season": s, "n": int(g["n"].sum()), "rate": w(g, "rate"), "share_gt5": w(g, "share_gt5"),
                     "mean_e": w(g, "mean_e")} for s, g in t.groupby("season")])
out.to_csv(RES / "Q1_season_pooled_test.csv", index=False)
print(out.round(4).to_string(index=False))
