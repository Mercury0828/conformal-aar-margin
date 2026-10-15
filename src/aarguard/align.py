"""Aligned forecast / observation / rating table (no test-period rows; see periods.py).

One row per (station, issue, valid): lead, band, cell, T_hat, T, e = T - T_hat, daytime flag, and R_c for the frozen
conductor cases. Cached as parquet under data/aligned/ (override with AARGUARD_ALIGNED).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import fc, obs
from .periods import TEST_OPEN, drop_test
from .rating import CONDUCTORS, T_MAX_CASES, ampacity, station_inputs, sun_elevation_deg

RUNS = Path(os.environ.get("AARGUARD_ALIGNED", Path(__file__).resolve().parents[2] / "data" / "aligned"))
REPO = Path(__file__).resolve().parents[2]
BANDS = [(1, 6), (7, 24), (25, 48), (49, 96), (97, 10_000)]


def band_label(lead: pd.Series) -> pd.Series:
    out = pd.Series("", index=lead.index, dtype=object)
    for lo, hi in BANDS:
        name = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        out[(lead >= lo) & (lead <= hi)] = name
    return out


def season(ts_local: pd.Series) -> pd.Series:
    m = ts_local.dt.month
    return pd.Series(np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["DJF", "MAM", "JJA"],
                               "SON"), index=ts_local.index)


def stations() -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "stations.csv").set_index("icao")


def rating_cols(df: pd.DataFrame, temp_col: str, lat: float, elev: float, prefix: str) -> pd.DataFrame:
    inp = station_inputs(lat, elev)
    for cname, c in CONDUCTORS.items():
        for t_max in T_MAX_CASES:
            df[f"{prefix}_{cname}_{int(t_max)}"] = ampacity(df[temp_col].values, c, t_max, inp, df["daytime"].values)
    return df


def build(model: str, station: str, issue_hour: int = 12, cache: bool = True) -> pd.DataFrame:
    tag = "OPEN" if TEST_OPEN else "closed"
    path = RUNS / f"aligned_{model}_{issue_hour:02d}Z_{station}_{tag}.parquet"
    if cache and path.exists():
        return drop_test(pd.read_parquet(path), ("valid", "issue"))
    meta = stations().loc[station]
    d = fc.forecasts(model, station, issue_hour)
    d["T"], d["t_obs"] = obs.observed(station, d["valid"])
    d = drop_test(d, ("valid", "issue"))
    d["station"] = station
    d["e"] = d["T"] - d["T_hat"]
    local = d["valid"] + pd.Timedelta(hours=int(meta.lst_offset_h))
    d["valid_lst"] = local.dt.tz_localize(None)
    d["season"] = season(d["valid_lst"])
    d["hblock"] = (d["valid_lst"].dt.hour // 6).astype(int)
    d["cell"] = d["season"] + "-" + d["hblock"].astype(str)
    d["band"] = band_label(d["lead"])
    d["daytime"] = sun_elevation_deg(d["valid"], meta.lat, meta.lon) > 0
    d = rating_cols(d, "T_hat", meta.lat, meta.elev_m, "R_hat")
    d = rating_cols(d, "T", meta.lat, meta.elev_m, "R_obs")
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    d.to_parquet(tmp)
    tmp.replace(path)
    return d


# --- Deterministic alignment checks ----------------------------
GFS_LEADS = set(range(6, 61, 3)) | {66, 72}  # MDL TPB 05-03: every 3 h, then every 6 h to 72 h


class AlignmentError(AssertionError):
    pass


def check_grid(d: pd.DataFrame, issue_hour: int = 12, leads: set[int] = GFS_LEADS) -> None:
    """Issue hour, 3-hourly valid grid and lead set as documented for the product (detects valid-time shifts)."""
    if not (d["issue"].dt.hour == issue_hour).all():
        raise AlignmentError("issue time differs from the frozen issue hour")
    if not ((d["valid"].dt.hour % 3 == 0) & (d["valid"].dt.minute == 0)).all():
        raise AlignmentError("valid time off the 3-hourly grid")
    lead = ((d["valid"] - d["issue"]).dt.total_seconds() / 3600)
    if not lead.isin(leads).all() or not (lead == d["lead"]).all():
        raise AlignmentError("lead inconsistent with valid - issue or outside the documented lead set")


def check_units(d: pd.DataFrame) -> None:
    """deg C plausibility of forecasts against observations (detects a deg F / deg C swap)."""
    e = (d["T"] - d["T_hat"]).dropna()
    if len(e) and (e.abs().median() > 5.0 or d["T_hat"].max() > 55 or d["T_hat"].min() < -40):
        raise AlignmentError("forecast temperatures implausible in deg C (unit swap?)")
