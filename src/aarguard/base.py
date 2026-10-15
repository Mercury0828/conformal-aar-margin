"""M-base: baselines B1–B4.

B1 raw forecast AAR (q = 0). B2 fixed margin m_b per band. B3 coverage-matched Gaussian margin mu_{s,b} + kappa_b
sigma_{s,b}. B3n nominal Gaussian margin mu_{s,b} + z_{1-alpha} sigma_{s,b}. B4 static seasonal rating.
Margins are tuned on the validation year only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import obs
from .align import season, stations
from .periods import HISTORY_START, VAL_START
from .rating import CONDUCTORS, T_MAX_CASES, ampacity, station_inputs


def order_stat_quantile(x: np.ndarray, alpha: float) -> float:
    """Smallest value m with empirical share(x <= m) >= 1 - alpha: the k = ceil(n(1-alpha)) order statistic."""
    x = np.sort(np.asarray(x, dtype=float))
    k = math.ceil(len(x) * (1 - alpha))
    return float(x[k - 1])


def tune_b2(val: pd.DataFrame, alpha: float) -> pd.Series:
    return val.groupby("band")["e"].apply(lambda e: order_stat_quantile(e.values, alpha)).rename("m_b")


def tune_b3(val: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.Series]:
    mom = val.groupby(["station", "band"])["e"].agg(mu="mean", sigma="std")
    z = val.join(mom, on=["station", "band"])
    if (mom["sigma"] <= 0).any() or mom["sigma"].isna().any():
        raise ValueError("B3: a station x band has sigma <= 0 or undefined")
    z["zs"] = (z["e"] - z["mu"]) / z["sigma"]
    kappa = z.groupby("band")["zs"].apply(lambda s: order_stat_quantile(s.values, alpha)).rename("kappa_b")
    return mom, kappa


def margins(df: pd.DataFrame, alpha: float, m_b: pd.Series, mom: pd.DataFrame, kappa: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["q_B1"] = 0.0
    out["q_B2"] = df["band"].map(m_b).values
    j = df[["station", "band"]].join(mom, on=["station", "band"])
    out["q_B3"] = (j["mu"] + df["band"].map(kappa) * j["sigma"]).values
    out["q_B3n"] = (j["mu"] + norm.ppf(1 - alpha) * j["sigma"]).values
    if out.isna().any().any() or (j["sigma"] <= 0).any():
        raise ValueError("degenerate baseline margin (NaN or sigma <= 0): define the treatment before reporting")
    return out


def static_design_temps(icaos: list[str] | None = None) -> pd.DataFrame:
    """Per station x season: p99 of hourly observed temperature, 2018-01-01 -> 2021-05-31 (LST seasons)."""
    st = stations()
    rows = []
    for icao in (icaos or list(st.index)):
        o = obs.read_isd(icao, range(2018, 2022))
        if len(o) == 0:
            continue
        o = o[(o["time"] >= HISTORY_START) & (o["time"] < VAL_START)]
        lst = (o["time"] + pd.Timedelta(hours=int(st.loc[icao, "lst_offset_h"]))).dt.tz_localize(None)
        o = o.assign(season=season(lst).values)
        for se, g in o.groupby("season"):
            rows.append({"station": icao, "season": se, "T_design": float(g["T"].quantile(0.99)), "n": len(g)})
    return pd.DataFrame(rows)


def static_rating(df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """B4 for every row of an aligned table (daytime solar always, conservative)."""
    st = stations()
    j = df[["station", "season"]].merge(design, on=["station", "season"], how="left")
    out = pd.DataFrame(index=df.index)
    for cname, c in CONDUCTORS.items():
        for t_max in T_MAX_CASES:
            vals = np.empty(len(df))
            for icao in df["station"].unique():
                m = (df["station"] == icao).values
                inp = station_inputs(st.loc[icao, "lat"], st.loc[icao, "elev_m"])
                vals[m] = ampacity(j.loc[m, "T_design"].values, c, t_max, inp, True)
            out[f"R_stat_{cname}_{int(t_max)}"] = vals
    return out
