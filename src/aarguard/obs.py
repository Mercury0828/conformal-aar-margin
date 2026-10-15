"""M-obs: NOAA ISD hourly temperature at the forecast valid hours.

For valid hour t (UTC), the observation is the FM-15 (routine METAR) report with time in [t - 15 min, t + 5 min],
nearest to t. TMP quality codes {0, 1, 4, 5, 9} are accepted when the value is not +9999; codes {2, 3, 6, 7} are
rejected. Values are in 0.1 degC.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .periods import drop_test

DATA = Path(os.environ.get("AARGUARD_RAW", Path(__file__).resolve().parents[2] / "data" / "raw")) / "noaa"
ACCEPT_QC = {"0", "1", "4", "5", "9"}
WINDOW_BEFORE = pd.Timedelta(minutes=15)
WINDOW_AFTER = pd.Timedelta(minutes=5)


def read_isd(station: str, years: range, _counts_only: bool = False) -> pd.DataFrame:
    """All QC-passed FM-15 temperatures (UTC, degC). Test-period reports are removed unless `_counts_only` (used only
    by coverage_counts, which returns a count and never a value)."""
    shipped = Path(__file__).resolve().parents[2] / "data" / "obs_hourly" / f"{station}.parquet"
    if not any((DATA / f"{station}_{y}.csv").exists() for y in years) and shipped.exists():
        out = pd.read_parquet(shipped)  # QC-passed FM-15 temperatures shipped with the repository
        out = out[out["time"].dt.year.isin(list(years))].reset_index(drop=True)
        return out if _counts_only else drop_test(out, ("time",)).reset_index(drop=True)
    frames = []
    for y in years:
        f = DATA / f"{station}_{y}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype=str)
        rt = d.iloc[:, 3].str.strip()  # first REPORT_TYPE column
        d = d.loc[rt == "FM-15", ["DATE", "TMP"]]
        val = d["TMP"].str.split(",", expand=True)
        v = val[0].str.strip()
        q = val[1].str.strip()
        ok = q.isin(ACCEPT_QC) & (v != "+9999")
        out = pd.DataFrame({"time": pd.to_datetime(d.loc[ok, "DATE"], utc=True),
                            "T": v[ok].astype(int) / 10.0})
        frames.append(out)
    if not frames:
        return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"), "T": pd.Series(dtype=float)})
    out = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return out if _counts_only else drop_test(out, ("time",)).reset_index(drop=True)


def at_valid_hours(obs: pd.DataFrame, valid: pd.Series, with_time: bool = False):
    """Observed temperature for each valid hour per the [t-15 min, t+5 min] nearest-report rule (NaN if none).
    With `with_time`, also the selected report's time (needed for availability: a report after tau never calibrates)."""
    if len(obs) == 0:
        nan = pd.Series(np.nan, index=valid.index)
        return (nan, pd.Series(pd.NaT, index=valid.index, dtype="datetime64[ns, UTC]")) if with_time else nan
    q = pd.DataFrame({"valid": valid.reset_index(drop=True), "_i": np.arange(len(valid))}).sort_values("valid")
    o = obs.rename(columns={"time": "t_obs"})
    back = pd.merge_asof(q, o, left_on="valid", right_on="t_obs", direction="backward", tolerance=WINDOW_BEFORE)
    fwd = pd.merge_asof(q, o, left_on="valid", right_on="t_obs", direction="forward", tolerance=WINDOW_AFTER)
    d_back = (back["valid"] - back["t_obs"]).dt.total_seconds()
    d_fwd = (fwd["t_obs"] - fwd["valid"]).dt.total_seconds()
    use_fwd = fwd["t_obs"].notna() & (back["t_obs"].isna() | (d_fwd < d_back))
    T = np.where(use_fwd, fwd["T"], back["T"])
    t_rep = pd.Series(np.where(use_fwd, fwd["t_obs"], back["t_obs"]), index=q["_i"].values).sort_index()
    res = pd.Series(T, index=q["_i"].values).sort_index()
    out = pd.Series(res.values, index=valid.index)
    if with_time:
        return out, pd.Series(pd.to_datetime(t_rep.values, utc=True), index=valid.index)
    return out


def observed(station: str, valid: pd.Series, years: range = range(2018, 2026)) -> tuple[pd.Series, pd.Series]:
    """(T, report time) at each valid hour; test-period hours stay NaN unless the test period is formally open."""
    vv = drop_test(pd.DataFrame({"valid": valid}))
    obs = read_isd(station, years)
    T = pd.Series(np.nan, index=valid.index)
    t_rep = pd.Series(pd.NaT, index=valid.index, dtype="datetime64[ns, UTC]")
    a, b = at_valid_hours(obs, vv["valid"], with_time=True)
    T.loc[vv.index] = a.values
    t_rep.loc[vv.index] = b
    return T, t_rep


def coverage_counts(station: str, valid_hours: pd.DatetimeIndex, years: range = range(2018, 2026)) -> int:
    """Number of the given valid hours with a QC-passed observation. Count only (usable in the test period)."""
    obs = read_isd(station, years, _counts_only=True)
    s = at_valid_hours(obs, pd.Series(valid_hours))
    return int(s.notna().sum())
