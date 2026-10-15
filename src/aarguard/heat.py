"""Heat-wave days on a complete local-standard-time calendar.

A date with no accepted observation breaks a run. A heat-wave day belongs to >= 3 consecutive calendar dates whose
observed daily maximum >= p95_s. Heat-wave cases additionally require observed T > p95_s at the evaluated valid time.
"""
from __future__ import annotations

import pandas as pd


def heatwave_days(obs: pd.DataFrame, p95: float, lst_offset_h: int) -> tuple[set, int]:
    """obs: QC-passed hourly observations (time UTC, T). Returns (heat-wave dates, number of calendar dates missing)."""
    if len(obs) == 0:
        return set(), 0
    d = (obs["time"] + pd.Timedelta(hours=lst_offset_h)).dt.tz_localize(None).dt.normalize()
    dmax = obs.groupby(d.values)["T"].max()
    cal = pd.date_range(dmax.index.min(), dmax.index.max(), freq="D")
    dmax = dmax.reindex(cal)
    hot = (dmax >= p95).fillna(False)          # a missing date is not hot and breaks the run
    run = hot.groupby((hot != hot.shift()).cumsum()).transform("sum")
    return set(cal[hot & (run >= 3)]), int(dmax.isna().sum())
