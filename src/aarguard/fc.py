"""M-fc: archived MOS station forecasts from the IEM archive.

Columns used: `runtime` (issue time tau, UTC), `ftime` (valid time t, UTC), `tmp` (deg F, integer). Lead = t - tau in
hours. Temperatures are converted to deg C once, here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .periods import drop_test

DATA = Path(os.environ.get("AARGUARD_RAW", Path(__file__).resolve().parents[2] / "data" / "raw")) / "forecasts"


def f_to_c(f):
    return (np.asarray(f, dtype=float) - 32.0) * 5.0 / 9.0


def read_mos_raw(model: str, station: str, years: range, _counts_only: bool = False) -> pd.DataFrame:
    """Archived forecasts. Rows with issue or valid time in the test period are removed unless `_counts_only` (used
    only by functions that return counts / lead structure, never values)."""
    frames = []
    for y in years:
        f = DATA / model / f"{station}_{y}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=["runtime", "ftime", "tmp"])
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["issue", "valid", "lead", "T_hat_F", "T_hat"])
    d = pd.concat(frames, ignore_index=True)
    d["issue"] = pd.to_datetime(d["runtime"], utc=True)
    d["valid"] = pd.to_datetime(d["ftime"], utc=True)
    d["lead"] = ((d["valid"] - d["issue"]).dt.total_seconds() / 3600.0).round().astype(int)
    d = d.rename(columns={"tmp": "T_hat_F"})[["issue", "valid", "lead", "T_hat_F"]]
    d = d.dropna(subset=["T_hat_F"]).drop_duplicates(["issue", "valid"])
    d["T_hat"] = f_to_c(d["T_hat_F"])
    d = d.sort_values(["issue", "valid"]).reset_index(drop=True)
    if _counts_only:
        return d.drop(columns=["T_hat_F", "T_hat"])
    return drop_test(d, ("valid", "issue")).reset_index(drop=True)


def forecasts(model: str, station: str, issue_hour: int = 12, years: range = range(2018, 2026)) -> pd.DataFrame:
    """Forecasts at the frozen issue hour, test period removed (valid and issue)."""
    d = read_mos_raw(model, station, years)
    d = d[d["issue"].dt.hour == issue_hour]
    return drop_test(d, ("valid", "issue")).reset_index(drop=True)


def issue_counts(model: str, station: str, issue_hour: int, years: range) -> pd.Series:
    """Number of distinct issues at `issue_hour` per calendar year. Count only (usable in the test period)."""
    d = read_mos_raw(model, station, years, _counts_only=True)
    d = d[d["issue"].dt.hour == issue_hour]
    return d.groupby(d["issue"].dt.year)["issue"].nunique()
