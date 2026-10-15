"""Shared loader and metrics for the confirmatory analyses Q1–Q5.

`load(period)` returns the aligned table for the evaluation period ("val" for the dry run, "test" for the confirmatory run,
which requires the test period to be formally open), the full history for calibration, heat-wave flags,
static ratings (B4) and the frozen validation-year baseline parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from aarguard import align, base, bound  # noqa: E402
from aarguard import periods  # noqa: E402
from aarguard.evalstats import block_ids, paired_ratio_difference, ratio_bootstrap, ratio_bootstrap_full  # noqa: E402
from aarguard.rating import CONDUCTORS, T_LOW, ampacity, station_inputs  # noqa: E402

RES = REPO / "results"
PERIODS = {"val": (periods.VAL_START, periods.VAL_END), "test": (periods.TEST_START, periods.TEST_END)}
B6_CHANGE = pd.Timestamp("2023-06-06 12:00", tz="UTC")   # NWS SCN 23-49 (MAV equations updated)
BAND_ORDER = ["1-6", "7-24", "25-48", "49-96"]
ALPHAS = (0.01, 0.05, 0.10)
CASES = [("Drake", 75.0), ("Drake", 100.0), ("Linnet", 75.0), ("Linnet", 100.0)]
PRIMARY = ("Drake", 75.0)
COUNTS: dict = {}


def stations() -> list[str]:
    sel = pd.read_csv(RES / "D1_selected_stations.csv")
    return list(sel.loc[sel["selected"], "icao"])


def load(period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if period == "test" and not periods.TEST_OPEN:
        raise RuntimeError("test period closed: set AARGUARD_TEST_OPEN=1 to open it")
    full = pd.concat([align.build("GFS", s) for s in stations()], ignore_index=True)
    hist = full.dropna(subset=["e", "t_obs"])
    a, b = PERIODS[period]
    ev = full[(full["valid"] >= a) & (full["valid"] < b)].copy()
    COUNTS.update({"issued_cases": len(ev), "missing_obs_cases": int(ev["T"].isna().sum())})
    ev = ev.dropna(subset=["T"]).copy()
    ev["day"] = block_ids(ev["valid"], "day", a)
    ev["week"] = block_ids(ev["valid"], "week", a)
    ev["date_lst"] = ev["valid_lst"].dt.normalize()
    ev["after_B6"] = ev["issue"] >= B6_CHANGE
    ev = ev.join(heatwave_flags(full, ev))
    design = pd.read_csv(RES / "D5_static_design_temps.csv")
    ev = ev.join(base.static_rating(ev.reset_index(drop=True), design).set_index(ev.index))
    return ev, hist


def heatwave_flags(full: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Heat-wave definition: runs of >= 3 LST days with daily max observed T >= p95_s (JJA 2018–2020); HW hours: T > p95_s."""
    p95 = pd.read_csv(RES / "D1_heatwave.csv").set_index("icao")["p95_JJA_2018_2020"]
    out = pd.DataFrame(index=ev.index, data={"HW": False})
    from aarguard import obs
    for s in ev["station"].unique():
        o = obs.read_isd(s, range(2018, 2026))
        from aarguard.heat import heatwave_days
        hw_days, _ = heatwave_days(o, p95[s], int(align.stations().loc[s, "lst_offset_h"]))
        m = ev["station"] == s
        out.loc[m, "HW"] = ev.loc[m, "date_lst"].isin(hw_days) & (ev.loc[m, "T"] > p95[s])
        out.loc[m, "HW_day"] = ev.loc[m, "date_lst"].isin(hw_days)
    out["HW_day"] = out["HW_day"].fillna(False).astype(bool)
    return out


def rating_at(df: pd.DataFrame, temps: np.ndarray, cname: str, t_max: float) -> np.ndarray:
    st = align.stations()
    out = np.zeros(len(df))
    for icao in df["station"].unique():
        m = (df["station"] == icao).values
        inp = station_inputs(st.loc[icao, "lat"], st.loc[icao, "elev_m"])
        t = np.where(np.isneginf(temps[m]), T_LOW, temps[m])
        r = ampacity(np.where(np.isfinite(t), t, 1e3), CONDUCTORS[cname], t_max, inp, df["daytime"].values[m])
        out[m] = np.where(np.isfinite(t), r, 0.0)
    return out


def bounds(ev: pd.DataFrame, hist: pd.DataFrame, alpha: float, w: int = 90, cells: str = "cell",
           aci_gamma: float | None = None) -> pd.Series:
    col = "cell"
    if cells == "season":
        ev = ev.assign(season_cell=ev["season"]); hist = hist.assign(season_cell=hist["season"]); col = "season_cell"
    if aci_gamma is None:
        return bound.split_bounds(hist, ev, alpha, w, cell_col=col)
    return bound.aci_bounds(hist, ev, alpha, aci_gamma, w, cell_col=col)


def frozen_baselines(ev: pd.DataFrame, hist: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """B1, B2, B3, B3n from the validation-year parameters (results/D5_*); B3r-n / B3r-c rolling Gaussian on the identical calibration sets."""
    from scipy.stats import norm
    tuned = pd.read_csv(RES / "D5_tuned.csv")
    t = tuned[np.isclose(tuned["alpha"], alpha)].set_index("band")
    mom = pd.read_csv(RES / f"D5_mu_sigma_alpha{alpha}.csv").set_index(["station", "band"])
    out = pd.DataFrame(index=ev.index)
    out["B1"] = 0.0
    out["B2"] = ev["band"].map(t["m_b"]).values
    j = ev[["station", "band"]].join(mom, on=["station", "band"])
    out["B3"] = (j["mu"] + ev["band"].map(t["kappa_b"]) * j["sigma"]).values
    out["B3n"] = (j["mu"] + norm.ppf(1 - alpha) * j["sigma"]).values
    mu, sd = rolling_moments(ev, hist, alpha)
    rg = pd.read_csv(RES / "D5_rolling_gaussian.csv")
    kap = rg[(rg["method"] == "B3r-c") & np.isclose(rg["alpha"], alpha)].set_index("band")["kappa"]
    out["B3r-n"] = (mu + norm.ppf(1 - alpha) * sd).values
    out["B3r-c"] = (mu + ev["band"].map(kap) * sd).values
    assert out.notna().all().all(), "undefined baseline margin"
    return out


def rolling_moments(ev: pd.DataFrame, hist: pd.DataFrame, alpha: float, w: int = 90, wmax: int = 730):
    """mu, sd of exactly the errors split AARGUARD uses for each target at this alpha."""
    mu = pd.Series(np.nan, index=ev.index); sd = pd.Series(np.nan, index=ev.index)
    groups = {k: g.sort_values("valid") for k, g in hist.groupby(["station", "band", "cell"])}
    for key, tg in ev.groupby(["station", "band", "cell"]):
        g = groups[key]
        cv, ca, ce = bound._naive(g["valid"].values), bound._naive(g["t_obs"].values), g["e"].values
        for tau, rows in tg.groupby("issue"):
            t64 = bound._naive(np.array([tau]))[0]
            e = bound.selected_errors(cv, ca, ce, t64, alpha, w, wmax)
            mu.loc[rows.index] = e.mean()
            sd.loc[rows.index] = e.std(ddof=1) if len(e) > 1 else np.nan
    if not (mu.notna().all() and (sd > 0).all()):
        raise ValueError("undefined or non-positive rolling sd: analysis stops")
    return mu, sd


def metrics(ev: pd.DataFrame, r_lo: np.ndarray, case=PRIMARY, block: str = "day") -> dict:
    key = f"{case[0]}_{int(case[1])}"
    r_obs, r_hat, r_stat = ev[f"R_obs_{key}"].values, ev[f"R_hat_{key}"].values, ev[f"R_stat_{key}"].values
    cov = (r_lo <= r_obs + 1e-9).astype(float)
    one = pd.Series(1.0, index=ev.index)
    c = ratio_bootstrap_full(pd.Series(cov, index=ev.index), one, ev[block])
    cc = ratio_bootstrap_full(pd.Series(r_hat - r_lo, index=ev.index), pd.Series(r_obs, index=ev.index), ev[block])
    gain = r_hat > r_stat
    rg = ratio_bootstrap_full(pd.Series((r_lo - r_stat) * gain, index=ev.index),
                              pd.Series((r_hat - r_stat) * gain, index=ev.index), ev[block])
    return {"n": len(ev), "n_days": ev["day"].nunique(), "cov": c[0], "cov_lo": c[1], "cov_hi": c[2], "cov_p5": c[3],
            "cov_se": c[4], "violation": 1 - c[0], "CC": cc[0], "CC_lo": cc[1], "CC_hi": cc[2], "RG": rg[0],
            "RG_lo": rg[1], "RG_hi": rg[2], "zero_bound_rate": float(np.mean(r_lo == 0))}


def paired(ev: pd.DataFrame, r_a: np.ndarray, r_b: np.ndarray, case=PRIMARY, block: str = "day") -> dict:
    key = f"{case[0]}_{int(case[1])}"
    r_obs = ev[f"R_obs_{key}"].values; r_hat = ev[f"R_hat_{key}"].values; one = np.ones(len(ev))
    dc = paired_ratio_difference((r_a <= r_obs + 1e-9), one, (r_b <= r_obs + 1e-9), one, ev[block])
    dcc = paired_ratio_difference(r_hat - r_a, r_obs, r_hat - r_b, r_obs, ev[block])
    return {"d_cov": dc[0], "d_cov_lo": dc[1], "d_cov_hi": dc[2], "d_CC": dcc[0], "d_CC_lo": dcc[1], "d_CC_hi": dcc[2]}


def tol(alpha: float) -> float:
    """Q2 SE-independent tolerance."""
    return (1 - alpha) - (0.010 if alpha >= 0.10 else 0.005)


def save(df: pd.DataFrame, name: str, period: str) -> Path:
    p = RES / f"{name}_{period}.csv"
    df.to_csv(p, index=False)
    return p


def manifest(period: str) -> dict:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    return {"period": period, "commit": commit, "test_open": periods.TEST_OPEN,
            "choice": json.loads((RES / "D4_choice.json").read_text()).get("A7_main_configuration")}
